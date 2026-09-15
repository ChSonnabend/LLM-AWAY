from __future__ import annotations

from dataclasses import dataclass
import atexit
import json
import shlex
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import AppConfig


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str
    model: str
    raw_prompt_chars: int | None = None


class Backend:
    def infer(self, request: InferenceRequest) -> str:
        raise NotImplementedError


class SlurmSshBackend(Backend):
    def __init__(self, config: AppConfig):
        self.config = config

    def infer(self, request: InferenceRequest) -> str:
        cmd = self.build_command(request)
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise BackendError(f"remote inference failed with exit code {completed.returncode}: {detail}")
        return completed.stdout.strip()

    def build_command(self, request: InferenceRequest) -> list[str]:
        cfg = self.config
        payload = {
            "prompt": request.prompt,
            "model": request.model,
            "remote_workdir": cfg.remote.workdir,
            "partition": cfg.slurm.partition,
            "exclusive": cfg.slurm.exclusive,
            "node_class": cfg.slurm.node_class,
            "custom_options": cfg.slurm.custom_options,
            "debug": cfg.slurm.debug,
            "llamacpp": {
                "backend": cfg.llamacpp.backend,
                "rocm_arch": cfg.llamacpp.rocm_arch,
                "visible_devices": cfg.llamacpp.visible_devices,
                "build_before_run": cfg.llamacpp.build_before_run,
                "show_config_before_run": cfg.llamacpp.show_config_before_run,
                "list_devices_before_run": cfg.llamacpp.list_devices_before_run,
                "run_cli": cfg.llamacpp.run_cli,
                "inference_timeout_seconds": cfg.llamacpp.inference_timeout_seconds,
                "extra_args": cfg.llamacpp.extra_args,
            },
        }
        remote = "bash -lc " + shlex.quote(f"{cfg.remote.runner} --json")
        return [
            "ssh",
            "-o",
            f"ConnectTimeout={cfg.ssh.connect_timeout_seconds}",
            cfg.ssh.destination,
            remote,
        ], json.dumps(payload)


class SubprocessSlurmSshBackend(SlurmSshBackend):
    def infer(self, request: InferenceRequest) -> str:
        cmd, stdin = self.build_command(request)
        attempts = max(1, self.config.ssh.retries)
        failures: list[str] = []

        for attempt in range(1, attempts + 1):
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
            )
            stdout, _ = process.communicate(stdin)
            if process.returncode == 0:
                return stdout.strip()

            detail = stdout.strip()
            failures.append(f"attempt {attempt}/{attempts}: exit {process.returncode}: {detail}")
            if process.returncode != 255 or attempt == attempts:
                break
            print(
                f"llm-epn: SSH failed with exit 255; retrying in {self.config.ssh.retry_delay_seconds}s",
                file=sys.stderr,
            )
            time.sleep(self.config.ssh.retry_delay_seconds)

        raise BackendError("remote inference failed:\n" + "\n".join(failures))


class SlurmServerBackend(Backend):
    def __init__(self, config: AppConfig):
        self.config = config
        self.local_port = config.gateway.local_port or self.find_free_port()
        self._tunnel: subprocess.Popen | None = None
        self._tunnel_target: str | None = None
        self._owned_job_id: str | None = None
        self._log_offset = 0
        self._ready_model: str | None = None
        self._ready_lock = threading.Lock()
        atexit.register(self.close)

    def infer(self, request: InferenceRequest) -> str:
        started = time.time()
        self.ensure_ready(request.model)
        ready_at = time.time()
        text = self.completion(request.prompt)
        completed_at = time.time()
        print(
            "llm-epn: request timing "
            f"ready={ready_at - started:.2f}s "
            f"completion={completed_at - ready_at:.2f}s "
            f"total={completed_at - started:.2f}s "
            f"prompt_chars={len(request.prompt)}"
            + (
                f" raw_prompt_chars={request.raw_prompt_chars}"
                if request.raw_prompt_chars is not None and request.raw_prompt_chars != len(request.prompt)
                else ""
            ),
            file=sys.stderr,
        )
        return text

    def ensure_ready(self, model: str) -> None:
        with self._ready_lock:
            if self._ready_model == model and self._tunnel and self._tunnel.poll() is None and self.http_ready(model):
                return

            before = self.serverctl("status", model)
            state = self.serverctl("ensure", model)
            if not before.get("active") and state.get("active") and state.get("job_id"):
                self._owned_job_id = str(state["job_id"])
                print(f"llm-epn: submitted Slurm job {self._owned_job_id}", file=sys.stderr)
            deadline = time.time() + self.config.gateway.startup_timeout_seconds

            while time.time() < deadline:
                state = self.serverctl("status", model)
                if self._owned_job_id:
                    self.stream_log(model)
                if state.get("job_id") and not state.get("active"):
                    log_path = state.get("log_path", "<unknown>")
                    raise BackendError(f"server job exited before becoming ready; remote log: {log_path}")
                host = state.get("host")
                if host:
                    self.ensure_tunnel(host)
                    if self.http_ready(model):
                        self._ready_model = model
                        return
                time.sleep(self.config.gateway.poll_interval_seconds)

            log_path = state.get("log_path", "<unknown>")
            raise BackendError(f"server did not become ready before timeout; remote log: {log_path}")

    def serverctl(self, command: str, model: str, extra: dict | None = None) -> dict:
        cfg = self.config
        payload = {
            "model": model,
            "remote_workdir": cfg.remote.workdir,
            "state_dir": cfg.remote.state_dir,
            "partition": cfg.slurm.partition,
            "exclusive": cfg.slurm.exclusive,
            "node_class": cfg.slurm.node_class,
            "custom_options": cfg.slurm.custom_options,
            "server_port": cfg.gateway.server_port,
            "llamacpp": {
                "backend": cfg.llamacpp.backend,
                "rocm_arch": cfg.llamacpp.rocm_arch,
                "visible_devices": cfg.llamacpp.visible_devices,
                "build_before_run": cfg.llamacpp.build_before_run,
                "model_name": cfg.llamacpp.model_name or model,
                "server_cli": cfg.llamacpp.server_cli,
                "model_path": cfg.llamacpp.model_path,
                "context_size": cfg.llamacpp.context_size,
                "server_extra_args": cfg.llamacpp.server_extra_args,
                "server_command": cfg.llamacpp.server_command,
            },
        }
        if extra:
            payload.update(extra)
        remote = "bash -lc " + shlex.quote(f"{cfg.remote.serverctl} {command} --json")
        cmd = [
            "ssh",
            *self.ssh_options(),
            cfg.ssh.destination,
            remote,
        ]
        attempts = max(1, cfg.ssh.retries)
        for attempt in range(1, attempts + 1):
            completed = subprocess.run(cmd, input=json.dumps(payload), text=True, capture_output=True, check=False)
            if completed.returncode == 0:
                try:
                    return json.loads(completed.stdout)
                except json.JSONDecodeError as exc:
                    raise BackendError(f"serverctl {command} returned non-JSON output: {completed.stdout}") from exc

            detail = completed.stderr.strip() or completed.stdout.strip()
            if completed.returncode != 255 or attempt == attempts:
                raise BackendError(f"serverctl {command} failed with exit {completed.returncode}: {detail}")
            print(
                f"llm-epn: serverctl SSH failed with exit 255; retrying in {cfg.ssh.retry_delay_seconds}s",
                file=sys.stderr,
            )
            time.sleep(cfg.ssh.retry_delay_seconds)

        raise BackendError(f"serverctl {command} failed unexpectedly")

    def stream_log(self, model: str) -> None:
        if not self.config.gateway.stream_startup_log:
            return
        try:
            result = self.serverctl("log", model, {"log_offset": self._log_offset})
        except BackendError:
            return
        self._log_offset = int(result.get("offset") or self._log_offset)
        data = result.get("data") or ""
        if data:
            sys.stderr.write(data)
            sys.stderr.flush()

    def ensure_tunnel(self, host: str) -> None:
        target = f"{self.local_port}:{host}:{self.config.gateway.server_port}"
        if self._tunnel and self._tunnel.poll() is None and self._tunnel_target == target:
            return
        if self._tunnel and self._tunnel.poll() is None:
            self._tunnel.terminate()

        local = f"{self.local_port}:{host}:{self.config.gateway.server_port}"
        self._tunnel = subprocess.Popen(
            [
                "ssh",
                *self.ssh_options(exit_on_forward_failure=True),
                "-N",
                "-L",
                local,
                self.config.ssh.destination,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._tunnel_target = target
        time.sleep(0.2)
        if self._tunnel.poll() is not None:
            raise BackendError(f"failed to open SSH tunnel on local port {self.local_port}")

    def http_ready(self, model: str) -> bool:
        try:
            with urlopen(self.local_url("/health"), timeout=2) as response:
                if response.status != 200:
                    return False
                health = json.loads(response.read().decode("utf-8"))
                if health.get("status") != "ok":
                    return False
        except (HTTPError, URLError, TimeoutError, ConnectionResetError, OSError, json.JSONDecodeError):
            return False

        try:
            with urlopen(self.local_url("/v1/models"), timeout=2) as response:
                if response.status != 200:
                    return False
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ConnectionResetError, OSError, json.JSONDecodeError):
            return False

        entries = []
        entries.extend(data.get("data") or [])
        entries.extend(data.get("models") or [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            names = {str(entry.get("id") or ""), str(entry.get("name") or ""), str(entry.get("model") or "")}
            names.update(str(alias) for alias in entry.get("aliases") or [])
            if model in names:
                return True
        return False

    def completion_payload(self, prompt: str, max_tokens: int | None = None) -> bytes:
        if max_tokens is None:
            max_tokens = self.config.llamacpp.max_tokens
        payload = json.dumps(
            {
                "model": self.config.model.name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        return payload

    def completion(self, prompt: str, max_tokens: int | None = None) -> str:
        payload = self.completion_payload(prompt, max_tokens=max_tokens)
        request = Request(
            self.local_url("/v1/chat/completions"),
            data=payload,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.llamacpp.inference_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except Exception as exc:
            raise BackendError(f"llama-server completion request failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip()
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return str(message.get("content") or choices[0].get("text") or data)
        return str(data.get("content") or data.get("response") or data)

    def local_url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.local_port}{path}"

    def ssh_options(self, exit_on_forward_failure: bool = False) -> list[str]:
        options = [
            ("ConnectTimeout", str(self.config.ssh.connect_timeout_seconds)),
            ("ControlMaster", "auto"),
            ("ControlPersist", "10m"),
            ("ControlPath", "/tmp/llm-epn-ssh-%C"),
            ("ServerAliveInterval", "30"),
        ]
        if exit_on_forward_failure:
            options.append(("ExitOnForwardFailure", "yes"))
        flattened: list[str] = []
        for key, value in options:
            flattened.extend(["-o", f"{key}={value}"])
        return flattened

    def close(self) -> None:
        if self._tunnel and self._tunnel.poll() is None:
            self._tunnel.terminate()
        should_cancel = self.config.gateway.cancel_on_exit and (
            self._owned_job_id or (self.config.gateway.cancel_reused_on_exit and self._ready_model)
        )
        if should_cancel:
            try:
                job_label = self._owned_job_id or "reused server"
                print(f"llm-epn: canceling Slurm job for {job_label}", file=sys.stderr)
                self.serverctl("cancel", self.config.model.name)
            except Exception as exc:
                print(f"llm-epn: failed to cancel Slurm job: {exc}", file=sys.stderr)
            finally:
                self._owned_job_id = None
                self._ready_model = None

    @staticmethod
    def find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def make_backend(config: AppConfig) -> Backend:
    if config.backend_type == "slurm":
        return SubprocessSlurmSshBackend(config)
    if config.backend_type == "slurm_server":
        return SlurmServerBackend(config)
    raise BackendError(f"unsupported backend type: {config.backend_type}")

from __future__ import annotations

from dataclasses import dataclass, asdict
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
from .protocol import normalize_chat_messages, native_tool_definitions


class BackendError(RuntimeError):
    pass


def gpus_for_cfg(cfg: AppConfig) -> int:
    """GPU count for the active host (from [hosts.NAME].gpus); 0 = cluster default."""
    return cfg.slurm.gpus


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str
    model: str
    raw_prompt_chars: int | None = None
    messages: list[dict] | None = None
    tools: list[dict] | None = None
    reasoning_effort: str | None = None


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
            "mi50_fallback": cfg.slurm.mi50_fallback,
            "custom_options": cfg.slurm.custom_options,
            "debug": cfg.slurm.debug,
            "gpus": cfg.slurm.gpus,
            "nodes": cfg.slurm.nodes,
            "llamacpp": {
                "backend": cfg.llamacpp.backend,
                "models_dir": cfg.llamacpp.models_dir,
                "installation_dir": cfg.llamacpp.installation_dir,
                "container": cfg.llamacpp.container,
                "container_source": cfg.llamacpp.container_source,
                "container_runtime": cfg.llamacpp.container_runtime,
                "rocm_arch": cfg.llamacpp.rocm_arch,
                "visible_devices": cfg.llamacpp.visible_devices,
                "build_before_run": cfg.llamacpp.build_before_run,
                "show_config_before_run": cfg.llamacpp.show_config_before_run,
                "list_devices_before_run": cfg.llamacpp.list_devices_before_run,
                "run_cli": cfg.llamacpp.run_cli,
                "model_name": cfg.llamacpp.model_name or cfg.model.name,
                "inference_timeout_seconds": cfg.llamacpp.inference_timeout_seconds,
                "extra_args": cfg.llamacpp.extra_args,
                "mtp": cfg.llamacpp.mtp,
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
                f"llm-away: SSH failed with exit 255; retrying in {self.config.ssh.retry_delay_seconds}s",
                file=sys.stderr,
            )
            time.sleep(self.config.ssh.retry_delay_seconds)

        raise BackendError("remote inference failed:\n" + "\n".join(failures))


class SlurmServerBackend(Backend):
    native_tools = True
    def __init__(self, config: AppConfig):
        self.config = config
        # Direct mode runs the server on the connected host itself: it never submits a
        # Slurm allocation, so there is no scheduler job to name; the remote helper
        # reports the local server process PID in "job_id" instead.
        self.direct_mode = config.backend_type == "direct"
        self.local_port = config.gateway.local_port or self.find_free_port()
        self.job_port: int | None = None
        self.slurm_nodes: int | None = None
        self._tunnel: subprocess.Popen | None = None
        self._tunnel_target: str | None = None
        self._owned_job_id: str | None = None
        self._log_offset = 0
        self._ready_model: str | None = None
        self._ready_lock = threading.Lock()
        self._closing = threading.Event()
        self._reused_job = False
        atexit.register(self.close)

    def infer(self, request: InferenceRequest) -> str:
        started = time.time()
        self.ensure_ready(self.config.model.name)
        ready_at = time.time()
        text = self.completion(request.prompt, messages=request.messages, tools=request.tools, reasoning_effort=request.reasoning_effort)
        completed_at = time.time()
        print(
            "llm-away: request timing "
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
            if self._closing.is_set():
                raise BackendError("backend is shutting down")
            if self._ready_model == model and self._tunnel and self._tunnel.poll() is None and self.http_ready(model):
                return

            before = self.serverctl("status", model)
            if self._closing.is_set():
                raise BackendError("backend is shutting down")
            # A requested remote port identifies a specific server allocation: reuse it
            # when the running job already serves that port, otherwise start a fresh
            # allocation so multiple agents can run on separate Slurm nodes.
            if self.job_port is not None and int(before.get("server_port") or 0) != int(self.job_port):
                state = self.serverctl("ensure", model, {"force_new_job": True, "server_port": self.job_port})
                self._owned_job_id = str(state.get("job_id") or "")
                self._reused_job = False
                self._log_offset = 0
                if self._owned_job_id:
                    if self.direct_mode:
                        print(f"llm-away: direct server process {self._owned_job_id} on remote port {self.job_port} (no Slurm job submitted)", file=sys.stderr)
                    else:
                        print(f"llm-away: submitted new Slurm job {self._owned_job_id} (remote port {self.job_port})", file=sys.stderr)
            else:
                state = self.serverctl("ensure", model)
                self._reused_job = bool(before.get("active") and state.get("active"))
                if not before.get("active") and state.get("active") and state.get("job_id"):
                    self._owned_job_id = str(state["job_id"])
                    self._log_offset = 0
                    if self.direct_mode:
                        print(f"llm-away: direct access to {self.config.ssh.destination}; no Slurm job is allocated "
                              f"(server process {self._owned_job_id} started directly on the host)", file=sys.stderr)
                    else:
                        print(f"llm-away: submitted Slurm job {self._owned_job_id}", file=sys.stderr)
            deadline = time.time() + self.config.gateway.startup_timeout_seconds

            last_phase = None
            while time.time() < deadline and not self._closing.is_set():
                state = self.serverctl("status", model)
                phase = state.get("slurm_state")
                progress = (phase, state.get("reason"))
                if phase and progress != last_phase:
                    job_id = state.get("job_id")
                    if self.direct_mode:
                        print(f"llm-away: direct server process {job_id}: {phase} ({state.get('reason') or 'no reason reported'})", file=sys.stderr)
                    else:
                        print(f"llm-away: Slurm job {job_id}: {phase} ({state.get('reason') or 'no reason reported'})", file=sys.stderr)
                    last_phase = progress
                if self._owned_job_id and phase not in ("PENDING", "CONFIGURING"):
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
                self._closing.wait(self.config.gateway.poll_interval_seconds)

            if self._closing.is_set():
                raise BackendError("backend is shutting down")
            log_path = state.get("log_path", "<unknown>")
            raise BackendError(f"server did not become ready before timeout; Slurm state: {state.get('slurm_state') or 'unknown'}, reason: {state.get('reason') or 'unknown'}; remote log: {log_path}")

    def serverctl(self, command: str, model: str, extra: dict | None = None) -> dict:
        cfg = self.config
        payload = {
            "model": model,
            "remote_workdir": cfg.remote.workdir,
            "state_dir": cfg.remote.state_dir,
            "kubernetes": asdict(cfg.kubernetes),
            "partition": cfg.slurm.partition,
            "exclusive": cfg.slurm.exclusive,
            "node_class": cfg.slurm.node_class,
            "mi50_fallback": cfg.slurm.mi50_fallback,
            "nodes": self.slurm_nodes or cfg.slurm.nodes,
            "gpus": gpus_for_cfg(cfg),
            "custom_options": cfg.slurm.custom_options,
            "server_port": (extra or {}).get("server_port", self.job_port) or cfg.gateway.server_port,
            "llamacpp": {
                "backend": cfg.llamacpp.backend,
                "models_dir": cfg.llamacpp.models_dir,
                "installation_dir": cfg.llamacpp.installation_dir,
                "container": cfg.llamacpp.container,
                "container_source": cfg.llamacpp.container_source,
                "container_runtime": cfg.llamacpp.container_runtime,
                "rocm_arch": cfg.llamacpp.rocm_arch,
                "visible_devices": cfg.llamacpp.visible_devices,
                "build_before_run": cfg.llamacpp.build_before_run,
                "model_name": cfg.llamacpp.model_name or model,
                "server_cli": cfg.llamacpp.server_cli,
                "model_path": cfg.llamacpp.model_path,
                "context_size": cfg.llamacpp.context_size,
                "server_extra_args": cfg.llamacpp.server_extra_args,
                "mtp": cfg.llamacpp.mtp,
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
        if cfg.ssh.connection == "local":
            cmd = ["bash", "-lc", shlex.quote(cfg.remote.serverctl) + " " + shlex.quote(command) + " --json"]
        if command == "ensure" and cfg.llamacpp.container:
            print("llm-away: preparing remote container (first launch downloads it; later launches reuse it)", file=sys.stderr, flush=True)
        attempts = max(1, cfg.ssh.retries)
        for attempt in range(1, attempts + 1):
            completed = subprocess.run(cmd, input=json.dumps(payload), text=True, capture_output=True, check=False, start_new_session=True)
            if completed.returncode == 0:
                try:
                    return json.loads(completed.stdout)
                except json.JSONDecodeError as exc:
                    raise BackendError(f"serverctl {command} returned non-JSON output: {completed.stdout}") from exc

            detail = completed.stderr.strip() or completed.stdout.strip()
            if completed.returncode != 255 or attempt == attempts:
                raise BackendError(f"serverctl {command} failed with exit {completed.returncode}: {detail}")
            print(
                f"llm-away: serverctl SSH failed with exit 255; retrying in {cfg.ssh.retry_delay_seconds}s",
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
        remote_port = self.job_port if self.job_port is not None else self.config.gateway.server_port
        if self.config.ssh.connection == "local":
            self._local_endpoint = (host, remote_port)
            return
        target = f"{self.local_port}:{host}:{remote_port}"
        if self._tunnel and self._tunnel.poll() is None and self._tunnel_target == target:
            return
        if self._tunnel and self._tunnel.poll() is None:
            self._tunnel.terminate()
            try:self._tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:self._tunnel.kill();self._tunnel.wait()

        local = f"{self.local_port}:{host}:{remote_port}"
        # Older multiplexed tunnels left their forward on the persistent master.
        # Cancel only this exact forward, never the shared SSH master itself.
        try:
            subprocess.run(['ssh',*self.ssh_options(),'-O','cancel','-L',local,
                            self.config.ssh.destination],stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,timeout=10)
        except subprocess.TimeoutExpired:pass
        self._tunnel = subprocess.Popen(
            [
                "ssh",
                # OpenSSH uses the first option value; these must precede defaults.
                "-o", "ControlMaster=no", "-o", "ControlPath=none",
                *self.ssh_options(exit_on_forward_failure=True),
                "-N",
                "-L",
                local,
                self.config.ssh.destination,
            ],
            stdout=subprocess.DEVNULL,
            stderr=None,
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

    def completion_payload(self, prompt: str, max_tokens: int | None = None, messages: list[dict] | None = None, tools=None, reasoning_effort=None) -> bytes:
        if max_tokens is None:
            max_tokens = self.config.llamacpp.max_tokens
        data = {
                "model": self.config.model.name,
                "messages": normalize_chat_messages(messages) if messages is not None else [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
        effort=reasoning_effort or self.config.codex.reasoning_effort
        if self.config.model.name.lower().startswith('glm'):
            effort={'medium':'high','xhigh':'max','minimal':'low','none':'low'}.get(effort,effort)
            if effort not in ('low','high','max'):raise BackendError('Unsupported GLM reasoning effort: '+str(effort))
            data['chat_template_kwargs']={'reasoning_effort':effort}
        data['reasoning_effort']=effort
        if tools:
            data.update(tools=native_tool_definitions(tools),tool_choice='auto',parallel_tool_calls=False)
        return json.dumps(data).encode('utf-8')

    def completion(self, prompt: str, max_tokens: int | None = None, messages: list[dict] | None = None, tools=None, reasoning_effort=None) -> str:
        payload = self.completion_payload(prompt, max_tokens=max_tokens, messages=messages, tools=tools, reasoning_effort=reasoning_effort)
        request = Request(
            self.local_url("/v1/chat/completions"),
            data=payload,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.llamacpp.inference_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read(16384).decode("utf-8", "replace")
            try:
                error = json.loads(detail).get("error", {})
                if isinstance(error, dict):
                    detail = str(error.get("message") or detail)
            except (ValueError, AttributeError):
                pass
            raise BackendError(f"llama-server HTTP {exc.code}: {detail or exc.reason}") from exc
        except Exception as exc:
            raise BackendError(f"llama-server completion request failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip()
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            if message.get("tool_calls"):
                calls = []
                for call in message["tool_calls"]:
                    function = call.get("function", {})
                    calls.append("<tool_call>" + json.dumps(function) + "</tool_call>")
                return str(message.get("content") or "") + "\n".join(calls)
            return str(
                message.get("content")
                or message.get("reasoning_content")
                or choices[0].get("text")
                or data
            )
        return str(data.get("content") or data.get("response") or data)

    def local_url(self, path: str) -> str:
        if hasattr(self, '_local_endpoint'):
            host, port = self._local_endpoint
            return f"http://{host}:{port}{path}"
        return f"http://127.0.0.1:{self.local_port}{path}"

    def ssh_options(self, exit_on_forward_failure: bool = False) -> list[str]:
        options = [
            ("ConnectTimeout", str(self.config.ssh.connect_timeout_seconds)),
            ("ControlMaster", "auto"),
            ("ControlPersist", "10m"),
            ("ControlPath", "/tmp/llm-away-ssh-%C"),
            ("ServerAliveInterval", "30"),
        ]
        if exit_on_forward_failure:
            options.append(("ExitOnForwardFailure", "yes"))
        flattened: list[str] = []
        for key, value in options:
            flattened.extend(["-o", f"{key}={value}"])
        return flattened

    def close(self) -> None:
        self._closing.set()
        # Wait for an in-flight submission to record its job ID before canceling.
        with self._ready_lock:
            if self._tunnel and self._tunnel.poll() is None:
                self._tunnel.terminate()
                try:
                    self._tunnel.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._tunnel.kill()
                    self._tunnel.wait()
            should_cancel = self.config.gateway.cancel_on_exit and (
                self._owned_job_id or (self.config.gateway.cancel_reused_on_exit and (self._reused_job or self._ready_model))
            )
            if should_cancel:
                try:
                    job_label = self._owned_job_id or "reused server"
                    if self.direct_mode:
                        print(f"llm-away: stopping direct server process {job_label} (no Slurm job to cancel)", file=sys.stderr)
                    else:
                        print(f"llm-away: canceling Slurm job for {job_label}", file=sys.stderr)
                    extra = {"job_id": self._owned_job_id} if self._owned_job_id else None
                    self.serverctl("cancel", self.config.model.name, extra)
                except Exception as exc:
                    if self.direct_mode:
                        print(f"llm-away: failed to stop direct server: {exc}", file=sys.stderr)
                    else:
                        print(f"llm-away: failed to cancel Slurm job: {exc}", file=sys.stderr)
                    return  # Keep the job reference so atexit can retry.
                self._owned_job_id = None
                self._ready_model = None
                self._reused_job = False

    @staticmethod
    def find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


class KubernetesBackend(SlurmServerBackend):
    """Keep kubectl forwarding and the tunnel on the same SSH login host."""
    def ensure_tunnel(self, host: str) -> None:
        if self._tunnel and self._tunnel.poll() is None and self._tunnel_target == host:
            return
        if self._tunnel and self._tunnel.poll() is None:
            self._tunnel.terminate()
        k = self.config.kubernetes
        cmd = ["kubectl"]
        if k.context:
            cmd.extend(["--context", k.context])
        port = self.job_port or self.config.gateway.server_port
        cmd.extend(["--namespace", k.namespace, "port-forward", "--address", "127.0.0.1",
                    "pod/" + host, str(self.local_port) + ":" + str(port)])
        if self.config.ssh.connection == "local":
            self._tunnel = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None, text=True)
            self._tunnel_target = host
            return
        self._tunnel = subprocess.Popen([
            "ssh", "-o", "ControlMaster=no", "-o", "ControlPath=none",
            *self.ssh_options(exit_on_forward_failure=True),
            "-L", f"{self.local_port}:127.0.0.1:{self.local_port}",
            self.config.ssh.destination, " ".join(shlex.quote(arg) for arg in cmd)],
            stdout=subprocess.DEVNULL, stderr=None, text=True)
        self._tunnel_target = host


def make_backend(config: AppConfig) -> Backend:
    if config.backend_type == "slurm":
        return SubprocessSlurmSshBackend(config)
    if config.backend_type in ("slurm_server", "direct"):
        return SlurmServerBackend(config)
    if config.backend_type == "kubernetes":
        return KubernetesBackend(config)
    raise BackendError(f"unsupported backend type: {config.backend_type}")

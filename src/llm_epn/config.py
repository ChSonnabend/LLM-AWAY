from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import ast

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass(frozen=True)
class ModelConfig:
    name: str = "epn-llamacpp"


@dataclass(frozen=True)
class SshConfig:
    host: str = "epnh"
    user: str = ""
    connect_timeout_seconds: int = 60
    retries: int = 3
    retry_delay_seconds: int = 5

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


@dataclass(frozen=True)
class RemoteConfig:
    runner: str = "$HOME/.local/bin/llm-epn-slurm-run"
    serverctl: str = "$HOME/.local/bin/llm-epn-serverctl"
    workdir: str = "/scratch/csonnabe/cern-fellowship/misc/lamacpp-llm"
    state_dir: str = "$HOME/.cache/llm-epn"


@dataclass(frozen=True)
class SlurmConfig:
    partition: str = "prod"
    exclusive: bool = True
    node_class: str = ""
    custom_options: list[str] = field(default_factory=list)
    debug: bool = False


@dataclass(frozen=True)
class GatewayConfig:
    local_port: int = 0
    server_port: int = 8080
    startup_timeout_seconds: int = 900
    poll_interval_seconds: int = 5
    idle_timeout_minutes: int = 20
    cancel_on_exit: bool = True
    cancel_reused_on_exit: bool = True
    stream_startup_log: bool = True
    max_prompt_chars: int = 20000
    prompt_keep_tail_chars: int = 16000
    prompt_keep_head_chars: int = 4000


@dataclass(frozen=True)
class LlamaCppConfig:
    backend: str = "rocm"
    rocm_arch: str = "auto"
    visible_devices: str = "0,1,2,3,4,5,6,7"
    build_before_run: bool = True
    show_config_before_run: bool = False
    list_devices_before_run: bool = False
    run_cli: str = "bin/run-cli"
    inference_timeout_seconds: int = 120
    max_tokens: int = 256
    extra_args: list[str] = field(default_factory=lambda: ["-n", "64"])
    model_name: str = "qwen3-coder-next-f16-1m"
    server_cli: str = "bin/run-server"
    model_path: str = ""
    context_size: int = 8192
    server_extra_args: list[str] = field(default_factory=list)
    server_command: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ssh: SshConfig = field(default_factory=SshConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    llamacpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    backend_type: str = "slurm_server"


def _merge(defaults: dict, values: dict) -> dict:
    merged = dict(defaults)
    merged.update(values or {})
    return merged


def _parse_scalar(value: str):
    value = value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") or value.startswith('"') or value.startswith("'"):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        return value


def _loads_toml(text: str) -> dict:
    if tomllib is not None:
        return tomllib.loads(text)

    parsed: dict[str, dict] = {}
    current: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            current = parsed.setdefault(section, {})
            continue
        if current is None or "=" not in line:
            raise ValueError(f"unsupported TOML line: {raw_line}")
        key, value = line.split("=", 1)
        current[key.strip()] = _parse_scalar(value)
    return parsed


def load_config(path: str | Path) -> AppConfig:
    raw = _loads_toml(Path(path).read_text(encoding="utf-8"))
    backend = raw.get("backend", {})
    return AppConfig(
        server=ServerConfig(**_merge(ServerConfig().__dict__, raw.get("server", {}))),
        model=ModelConfig(**_merge(ModelConfig().__dict__, raw.get("model", {}))),
        ssh=SshConfig(**_merge(SshConfig().__dict__, raw.get("ssh", {}))),
        remote=RemoteConfig(**_merge(RemoteConfig().__dict__, raw.get("remote", {}))),
        slurm=SlurmConfig(**_merge(SlurmConfig().__dict__, raw.get("slurm", {}))),
        gateway=GatewayConfig(**_merge(GatewayConfig().__dict__, raw.get("gateway", {}))),
        llamacpp=LlamaCppConfig(**_merge(LlamaCppConfig().__dict__, raw.get("llamacpp", {}))),
        backend_type=backend.get("type", "slurm_server"),
    )

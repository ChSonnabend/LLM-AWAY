from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    name: str = "away-llamacpp"


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
class HostConfig:
    """One remote compute host: its SSH alias and how jobs are submitted there."""
    name: str = ""
    ssh_host: str = ""
    ssh_user: str = ""
    partition: str = ""
    exclusive: bool = True
    node_class: str = ""
    mi50_fallback: bool = False
    # Number of full nodes per allocation. 1 is the current single-node behavior;
    # >1 requests a multi-node allocation (needs a multi-node-capable remote wrapper
    # and Slurm IB config to actually span the model across nodes).
    nodes: int = 1
    custom_options: list[str] = field(default_factory=list)
    debug: bool = False
    # GPU count per allocation. 0 lets Slurm/the remote wrapper decide (default);
    # set to e.g. 4 for `--gres=gpu:4` partial-node allocations (H100/H200 hosts).
    gpus: int = 0
    remote_workdir: str = ""
    runner: str = ""
    serverctl: str = ""
    state_dir: str = ""
    backend: str = ""
    rocm_arch: str = ""
    visible_devices: str | None = None

    @property
    def destination(self) -> str:
        return f"{self.ssh_user}@{self.ssh_host}" if self.ssh_user else self.ssh_host

    @property
    def label(self) -> str:
        return self.name or self.ssh_host or "default"


@dataclass(frozen=True)
class RemoteConfig:
    runner: str = "$HOME/.local/bin/llm-away-slurm-run"
    serverctl: str = "$HOME/.local/bin/llm-away-serverctl"
    workdir: str = "/scratch/csonnabe/cern-fellowship/misc/LLM-AWAY-remote"
    state_dir: str = "$HOME/.cache/llm-away"


@dataclass(frozen=True)
class SlurmConfig:
    gpus: int = 0
    partition: str = "prod"
    exclusive: bool = True
    node_class: str = ""
    mi50_fallback: bool = False
    # Number of full nodes per allocation. 1 is the current single-node behavior;
    # >1 requests a multi-node allocation (needs a multi-node-capable remote wrapper
    # and Slurm IB config to actually span the model across nodes).
    nodes: int = 1
    custom_options: list[str] = field(default_factory=list)
    debug: bool = False

    def __post_init__(self):
        if self.gpus < 0:
            raise ValueError("slurm.gpus must be >= 0")
        if self.nodes < 1:
            raise ValueError("slurm.nodes must be >= 1")


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
    max_prompt_chars: int = 0
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
    inference_timeout_seconds: int = 900
    max_tokens: int = 16384
    extra_args: list[str] = field(default_factory=lambda: ["-n", "64"])
    model_name: str = "qwen3.8-flash-next-125b-ultralite-37g"
    server_cli: str = "bin/run-server"
    model_path: str = ""
    context_size: int = 45000
    server_extra_args: list[str] = field(default_factory=list)
    server_command: list[str] = field(default_factory=list)
    mtp: str = "auto"

    def __post_init__(self):
        if self.mtp not in ("auto", "on", "off"):
            raise ValueError("llamacpp.mtp must be auto, on, or off")
        if self.mtp != "auto":
            if self.model_path or self.server_command:
                raise ValueError("llamacpp.mtp overrides require a managed model (empty model_path and server_command)")
            controls = {"--spec-type", "--spec-draft-model", "--model-draft", "-md"}
            if any(arg.split("=", 1)[0] in controls for arg in self.extra_args + self.server_extra_args):
                raise ValueError("Use llamacpp.mtp or manual speculative type/model arguments, not both")

@dataclass(frozen=True)
class CodexConfig:
    provider_display_name: str = ""
    account_email: str = ""
    instructions: str = ""
    sandbox_mode: str = "workspace-write"
    approval_policy: str = "on-request"
    context_window: int = 65536
    auto_compact_token_limit: int = 45000
    tool_output_token_limit: int = 2000


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ssh: SshConfig = field(default_factory=SshConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    llamacpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    codex: CodexConfig = field(default_factory=CodexConfig)
    backend_type: str = "slurm_server"
    hosts: dict[str, HostConfig] = field(default_factory=dict)

    def host_configs(self) -> list[HostConfig]:
        """Configured hosts first (in [hosts] order); the default host last."""
        result = [host for host in self.hosts.values() if host.name or host.ssh_host]
        if not any(host.ssh_host == self.ssh.host for host in result):
            result.append(HostConfig(name="default", ssh_host=self.ssh.host, ssh_user=self.ssh.user))
        return result

    def host_config(self, name: str | None = None) -> HostConfig:
        name = (name or "").strip().lower()
        for host in self.host_configs():
            if host.label.lower() == name or host.name.lower() == name or host.ssh_host.lower() == name:
                return host
        known = ", ".join(host.label for host in self.host_configs()) or "<none>"
        raise ValueError(f"unknown host {name!r}; configured hosts: {known}")

    def with_host(self, name: str | None = None) -> "AppConfig":
        """Return a copy whose top-level ssh/slurm settings come from one host."""
        if not (name or "").strip():
            return self
        host = self.host_config(name)
        if host.name == "default" and host.ssh_host == self.ssh.host:
            return self
        return replace(
            self,
            ssh=replace(self.ssh, host=host.ssh_host, user=host.ssh_user),
            remote=replace(self.remote,
                workdir=host.remote_workdir or self.remote.workdir,
                runner=host.runner or self.remote.runner,
                serverctl=host.serverctl or self.remote.serverctl,
                state_dir=host.state_dir or self.remote.state_dir),
            llamacpp=replace(self.llamacpp,
                backend=host.backend or self.llamacpp.backend,
                rocm_arch=host.rocm_arch or self.llamacpp.rocm_arch,
                visible_devices=self.llamacpp.visible_devices if host.visible_devices is None else host.visible_devices),
            slurm=replace(
                self.slurm,
                partition=host.partition or self.slurm.partition,
                exclusive=host.exclusive,
                node_class=host.node_class,
                mi50_fallback=host.mi50_fallback,
                nodes=host.nodes,
                gpus=host.gpus,
                custom_options=list(host.custom_options),
                debug=host.debug,
            ),
        )


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
    hosts: dict[str, HostConfig] = {}
    for key, values in raw.items():
        if not key.startswith("hosts.") or not isinstance(values, dict):
            continue
        values = dict(values)
        name = values.pop("name", key[len("hosts."):])
        values.pop("name", None)
        hosts[name] = HostConfig(**_merge(HostConfig(name=name).__dict__, values))
    host_table = raw.get("hosts")
    if isinstance(host_table, dict):
        for name, values in host_table.items():
            if not isinstance(values, dict):
                continue
            values = dict(values)
            display = values.pop("name", name)
            values.pop("name", None)
            hosts[display] = HostConfig(**_merge(HostConfig(name=display).__dict__, values))
    return AppConfig(
        server=ServerConfig(**_merge(ServerConfig().__dict__, raw.get("server", {}))),
        model=ModelConfig(**_merge(ModelConfig().__dict__, raw.get("model", {}))),
        ssh=SshConfig(**_merge(SshConfig().__dict__, raw.get("ssh", {}))),
        remote=RemoteConfig(**_merge(RemoteConfig().__dict__, raw.get("remote", {}))),
        slurm=SlurmConfig(**_merge(SlurmConfig().__dict__, raw.get("slurm", {}))),
        hosts=hosts,
        gateway=GatewayConfig(**_merge(GatewayConfig().__dict__, raw.get("gateway", {}))),
        llamacpp=LlamaCppConfig(**_merge(LlamaCppConfig().__dict__, raw.get("llamacpp", {}))),
        codex=CodexConfig(**_merge(CodexConfig().__dict__, raw.get("codex", {}))),
        backend_type=backend.get("type", "slurm_server"),
    )

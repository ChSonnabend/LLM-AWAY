from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict
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
    name: str = "model"


@dataclass(frozen=True)
class SshConfig:
    connection: str = "ssh"
    host: str = "epnh"
    user: str = ""
    connect_timeout_seconds: int = 60
    retries: int = 3
    retry_delay_seconds: int = 5

    @property
    def destination(self) -> str:
        if self.connection == "local":
            return "local machine"
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
    container: str = ""
    container_source: str = ""
    container_runtime: str = "apptainer"
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
    # Empty preserves resource locations for already-created sessions.
    resource_state_dir: str = ""
    runner: str = "$HOME/.local/bin/llm-away-slurm-run"
    serverctl: str = "$HOME/.local/bin/llm-away-serverctl"
    workdir: str = "/scratch/csonnabe/LLM-AWAY/remote"
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
    startup_timeout_seconds: int = 1800
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
    models_dir: str = ""
    installation_dir: str = ""
    container: str = ""
    container_source: str = ""
    container_runtime: str = "apptainer"
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
    context_size: int = 100000
    server_extra_args: list[str] = field(default_factory=list)
    model_batch_defaults: dict[str, list[int]] = field(default_factory=lambda: {
        'glm-5.3-flash-q4': [2048, 1024], 'glm-5.3-flash-q8': [2048, 512]})
    model_host_batch_defaults: dict[str, dict[str, list[int]]] = field(default_factory=lambda: {
        'hydra': {'glm-5.3-flash-q4': [2048, 512], 'glm-5.3-flash-q8': [2048, 512]}})
    server_command: list[str] = field(default_factory=list)
    mtp: str = "auto"
    mtp_draft_tokens: int | None = None

    def __post_init__(self):
        if self.mtp not in ("auto", "on", "off"):
            raise ValueError("llamacpp.mtp must be auto, on, or off")
        if self.mtp != "auto":
            if self.model_path or self.server_command:
                raise ValueError("llamacpp.mtp overrides require a managed model (empty model_path and server_command)")
            controls = {"--spec-type", "--spec-draft-model", "--model-draft", "-md"}
            if any(arg.split("=", 1)[0] in controls for arg in self.extra_args + self.server_extra_args):
                raise ValueError("Use llamacpp.mtp or manual speculative type/model arguments, not both")
        if self.mtp_draft_tokens is not None and self.mtp_draft_tokens < 1:
            raise ValueError("llamacpp.mtp_draft_tokens must be at least 1")

@dataclass(frozen=True)
class CodexConfig:
    provider_display_name: str = ""
    account_email: str = ""
    instructions: str = ""
    sandbox_mode: str = "workspace-write"
    approval_policy: str = "on-request"
    reasoning_effort: str = "high"
    model_verbosity: str = "medium"
    hide_agent_reasoning: bool = True
    custom_metadata: bool = False
    context_window: int = 65536
    auto_compact_token_limit: int = 100000
    tool_output_token_limit: int = 2000


@dataclass(frozen=True)
class KubernetesConfig:
    context: str = ""
    namespace: str = "default"
    image: str = ""
    pvc: str = ""
    gpu_resource: str = "nvidia.com/gpu"
    cpu: str = "8"
    memory: str = "64Gi"
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: list[dict] = field(default_factory=list)
    priority_class: str = ""
    time_limit_seconds: int = 0


@dataclass(frozen=True)
class AgentConfig:
    cli: str = "auto"


@dataclass(frozen=True)
class ClaudeConfig:
    # Blank shares the concise instructions already configured for Codex.
    instructions: str = ""


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
    agent: AgentConfig = field(default_factory=AgentConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    backend_type: str = "slurm_server"
    hosts: dict[str, HostConfig] = field(default_factory=dict)
    kubernetes: KubernetesConfig = field(default_factory=KubernetesConfig)
    saved_hosts: dict = field(default_factory=dict)
    active_host: str = ""

    def host_configs(self) -> list[HostConfig]:
        """Configured hosts first (in [hosts] order); the default host last."""
        result = [HostConfig(name=name, ssh_host=profile['ssh']['host'])
                  for name, profile in self.saved_hosts.items()]
        result.extend(host for name, host in self.hosts.items() if name not in self.saved_hosts)
        if not any(host.ssh_host == self.ssh.host for host in result):
            result.append(HostConfig(name=self.ssh.host, ssh_host=self.ssh.host, ssh_user=self.ssh.user))
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
        if name == "default":
            name = self.ssh.host
        if name in self.saved_hosts:
            from .host_store import apply_profile
            profile = self.saved_hosts[name]
            return replace(apply_profile(self, profile), active_host='' if profile.get('_builtin') else name)
        host = self.host_config(name)
        if host.name not in self.hosts and host.ssh_host == self.ssh.host:
            return self
        return replace(
            self,
            backend_type="slurm_server",
            active_host="",
            kubernetes=KubernetesConfig(),
            ssh=replace(self.ssh, connection="ssh", host=host.ssh_host, user=host.ssh_user),
            remote=replace(self.remote,
                resource_state_dir="",
                workdir=host.remote_workdir or self.remote.workdir,
                runner=host.runner or self.remote.runner,
                serverctl=host.serverctl or self.remote.serverctl,
                state_dir=host.state_dir or self.remote.state_dir),
            llamacpp=replace(self.llamacpp,
                models_dir="",
                installation_dir="",
                container=host.container,
                container_source=host.container_source,
                container_runtime=host.container_runtime,
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
    cfg = AppConfig(
        server=ServerConfig(**_merge(ServerConfig().__dict__, raw.get("server", {}))),
        model=ModelConfig(**_merge(ModelConfig().__dict__, raw.get("model", {}))),
        ssh=SshConfig(**_merge(SshConfig().__dict__, raw.get("ssh", {}))),
        remote=RemoteConfig(**_merge(RemoteConfig().__dict__, raw.get("remote", {}))),
        slurm=SlurmConfig(**_merge(SlurmConfig().__dict__, raw.get("slurm", {}))),
        hosts=hosts,
        kubernetes=KubernetesConfig(**raw.get("kubernetes", {})),
        gateway=GatewayConfig(**_merge(GatewayConfig().__dict__, raw.get("gateway", {}))),
        llamacpp=LlamaCppConfig(**_merge(LlamaCppConfig().__dict__, raw.get("llamacpp", {}))),
        codex=CodexConfig(**_merge(CodexConfig().__dict__, raw.get("codex", {}))),
        agent=AgentConfig(**raw.get("agent", {})),
        claude=ClaudeConfig(**raw.get("claude", {})),
        backend_type=backend.get("type", "slurm_server"),
    )
    from .host_store import read_store
    store = read_store(path)
    baseline = {key: asdict(getattr(cfg, key)) for key in
                ('ssh', 'remote', 'slurm', 'llamacpp', 'model', 'kubernetes')}
    baseline['backend_type'] = cfg.backend_type
    baseline['_builtin'] = True
    cfg = replace(cfg, saved_hosts={cfg.ssh.host: baseline, **store['hosts']},
                  active_host=store.get('active', ''))
    return cfg.with_host(cfg.active_host) if cfg.active_host else cfg

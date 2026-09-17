from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .backends import InferenceRequest, SlurmServerBackend, make_backend
from .config import CodexConfig, load_config
from .server import serve
from .models import choose_model, choose_mtp, discover_models, mtp_label, save_model


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "model.toml"


def print_codex_config(config_path: str) -> None:
    cfg = load_config(config_path)
    base_url = f"http://{cfg.server.host}:{cfg.server.port}/v1"
    print(
        f'''# Add this provider block to ~/.codex/config.toml.
[model_providers.remote]
name = "{codex_provider_display_name(cfg.codex.provider_display_name)}"
base_url = "{base_url}"
wire_api = "responses"

# Add this profile block to ~/.codex/remote.config.toml.
model_provider = "remote"
model = "{cfg.model.name}"
model_reasoning_effort = "{cfg.codex.reasoning_effort}"
model_catalog_json = "~/.codex/model-catalogs/model.json"
'''
    )


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()


def codex_provider_display_name(configured_name: str = "") -> str:
    return (
        os.environ.get("LLM_REMOTE_CODEX_PROVIDER_NAME")
        or configured_name
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or "AWAY"
    )


def choose_host(options: list[str], current: str) -> str | None:
    """Ask which remote host to run the models on; None means the user canceled."""
    from .prompt import choose_option, interactive_available
    labels = options
    if interactive_available():
        try:
            index = choose_option(labels, "remote host", 0)
        except (ValueError, KeyboardInterrupt) as exc:
            print(f"\nHost selection canceled: {exc}", file=sys.stderr)
            return None
        print(f"llm-away: selected host {options[index]}", file=sys.stderr)
        return options[index]
    if not sys.stdin.isatty():
        return current if current in options else options[0]
    # Non-interactive: fall back to the default host unless exactly one is configured.
    if len(options) == 1:
        return options[0]
    for index, name in enumerate(options, 1):
        marker = " (current)" if name == current else ""
        print(f"  {index}. {name}{marker}")
    while True:
        try:
            answer = input("Choose remote host number or name [Enter keeps default], q to cancel: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHost selection canceled.", file=sys.stderr)
            return None
        if answer.lower() == "q":
            return None
        if not answer:
            return current if current in options else options[0]
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        for name in options:
            if name.lower() == answer.lower():
                return name
        print("Choose one of the listed hosts.")


def quoted(value: str) -> str:
    return json.dumps(value)


def remote_model_catalog(model: str, context_window: int = 262000, settings: CodexConfig | None = None) -> dict:
    settings = settings or CodexConfig()
    return {
        "models": [
            {
                "slug": model,
                "display_name": {"glm-5.3-flash-q4":"GLM-5.3-Flash Q4", "qwen3.8-27b-q4km":"Qwen3.8 27B Q4_K_M"}.get(model,model),
                "description": f"Use the selected remote model (currently {model}).",
                "default_reasoning_level": settings.reasoning_effort,
                "supported_reasoning_levels": [
                    {"effort": "low", "description": "Fast responses"},
                    {"effort": "medium", "description": "Default"},
                    {"effort": "high", "description": "More careful"},
                ],
                "shell_type": "unified_exec",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 100,
                "additional_speed_tiers": [],
                "service_tiers": [],
                "default_service_tier": None,
                "availability_nux": None,
                "upgrade": None,
                "support_verbosity": False,
                "default_verbosity": settings.model_verbosity,
                "truncation_policy": {
                    "mode": "tokens",
                    "limit": settings.tool_output_token_limit,
                },
                "context_window": context_window,
                "max_context_window": context_window,
                "effective_context_window_percent": 75,
                "input_modalities": ["text"],
                "supports_image_detail_original": False,
                "supports_search_tool": False,
                "supports_experimental_context": False,
                "experimental_supported_tools": [],
                "web_search_tool_type": "text_and_image",
                "apply_patch_tool_type": "freeform",
                "node_repl_disabled": True,
                "node_repl_auto_review_required": False,
                "include_apps_usage_instructions": False,
                "include_plugin_usage_instructions": False,
                "include_skills_usage_instructions": False,
                "use_responses_lite": False,
                "default_reasoning_summary": "none",
                "multi_agent_reasoning_effort": "high",
                "multi_agent_version": "v2",
                "model_messages": {
                    "instructions_template": settings.instructions or 'Complete requested edits with the smallest correct change. Batch related reads and SSH commands. Reuse established facts; do not repeat successful inspections without a reason. Once sufficient evidence is available, edit rather than continuing discovery. Preserve unrelated changes and follow repository instructions. After two failed attempts, change approach or report a concrete blocker. Use the provided tools and exact argument schemas; never invent commands or claim execution without tool evidence. Prefer patches for edits. Run focused checks appropriate to the change. Report the result, verification and remaining uncertainty briefly.'
                },
                "comp_hash": "local-model-preset",
            }
        ]
    }


def write_model_catalogs(home: Path, model: str, context_window: int = 262000, settings: CodexConfig | None = None) -> tuple[Path, Path]:
    catalog_dir = home / "model-catalogs"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    remote_path = catalog_dir / "model.json"
    combined_path = catalog_dir / "combined-with-model.json"
    remote_catalog = remote_model_catalog(model, context_window=context_window, settings=settings)
    remote_path.write_text(json.dumps(remote_catalog, indent=2) + "\n", encoding="utf-8")

    combined = remote_catalog
    if combined_path.exists():
        try:
            combined = json.loads(combined_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            combined = remote_catalog
    models = [
        entry
        for entry in combined.get("models", [])
        if entry.get("slug") not in (model, "away") and not str(entry.get("comp_hash", "")).startswith("local-away-")
    ]
    models.append(remote_catalog["models"][0])
    combined["models"] = models
    combined_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    return remote_path, combined_path


def remove_toml_table(text: str, table: str) -> str:
    lines = text.splitlines()
    result: list[str] = []
    skipping = False
    header = f"[{table}]"
    nested_prefix = f"[{table}."
    for line in lines:
        stripped = line.strip()
        if stripped == header or stripped.startswith(nested_prefix):
            while result:
                while result and not result[-1].strip():
                    result.pop()
                if result and result[-1].strip() == "# LLM remote provider":
                    result.pop()
                    continue
                break
            skipping = True
            continue
        if skipping and stripped.startswith("[") and stripped.endswith("]"):
            if stripped == header or stripped.startswith(nested_prefix):
                continue
            skipping = False
        if not skipping:
            result.append(line)
    return "\n".join(result).rstrip() + "\n" if result else ""


def set_root_keys(text: str, values: dict[str, str]) -> str:
    lines = text.splitlines()
    table_index = next((index for index, line in enumerate(lines) if line.strip().startswith("[")), len(lines))
    root = lines[:table_index]
    rest = lines[table_index:]
    keys = set(values)
    kept_root = []
    for line in root:
        stripped = line.strip()
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in keys:
                continue
        kept_root.append(line)

    new_root = [f"{key} = {quoted(value)}" for key, value in values.items()]
    if kept_root:
        new_root.extend(["", *kept_root])
    if rest:
        new_root.extend(["", *rest])
    return "\n".join(new_root).rstrip() + "\n"


def install_codex_config(config_path: str, activate: bool = False) -> None:
    cfg = load_config(config_path)
    home = codex_home()
    home.mkdir(parents=True, exist_ok=True)
    config_file = home / "config.toml"
    profile_file = home / "remote.config.toml"

    remote_catalog, combined_catalog = write_model_catalogs(
        home, cfg.model.name, context_window=min(cfg.codex.context_window, cfg.llamacpp.context_size), settings=cfg.codex
    )
    base_url = f"http://{cfg.server.host}:{cfg.server.port}/v1"

    existing = config_file.read_text(encoding="utf-8") if config_file.exists() else ""
    updated = remove_toml_table(existing, "model_providers.remote")
    if activate:
        updated = set_root_keys(
            updated,
            {
                "model": cfg.model.name,
                "model_provider": "remote",
                "model_reasoning_effort": cfg.codex.reasoning_effort,
                "model_catalog_json": str(remote_catalog),
            },
        )
    provider_block = f'''
# LLM remote provider
[model_providers.remote]
name = "{codex_provider_display_name(cfg.codex.provider_display_name)}"
base_url = "{base_url}"
wire_api = "responses"
'''
    final_config = updated.rstrip() + "\n\n" + provider_block.lstrip()
    if final_config != existing:
        if existing:
            backup = home / f"config.toml.bak-before-llm-away-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            backup.write_text(existing, encoding="utf-8")
            openai_copy = home / "config-openai.toml"
            if not openai_copy.exists():
                openai_copy.write_text(existing, encoding="utf-8")
        config_file.write_text(final_config, encoding="utf-8")

    profile_text = (
        "\n".join(
            [
                'model_provider = "remote"',
                f"model = {quoted(cfg.model.name)}",
                f"model_reasoning_effort = {quoted(cfg.codex.reasoning_effort)}",
                f"model_catalog_json = {quoted(str(remote_catalog))}",
                f"sandbox_mode = {quoted(cfg.codex.sandbox_mode)}",
                f"approval_policy = {quoted(cfg.codex.approval_policy)}",
                f"model_context_window = {min(cfg.codex.context_window, cfg.llamacpp.context_size)}",
                f"model_auto_compact_token_limit = {min(cfg.codex.auto_compact_token_limit, int(min(cfg.codex.context_window, cfg.llamacpp.context_size) * 0.7))}",
                f"tool_output_token_limit = {cfg.codex.tool_output_token_limit}",
                f"hide_agent_reasoning = {str(cfg.codex.hide_agent_reasoning).lower()}",
                f"model_verbosity = {quoted(cfg.codex.model_verbosity)}",
                "",
            ]
        )
    )
    if not profile_file.exists() or profile_file.read_text(encoding="utf-8") != profile_text:
        profile_file.write_text(profile_text, encoding="utf-8")
    if activate:
        shutil.copy2(config_file, home / "config-model.toml")

    print(f"Installed remote Codex provider in {config_file}")
    print(f"Installed remote profile in {profile_file}")
    print(f"Installed remote model catalog in {remote_catalog}")
    if activate:
        print("AWAY is active for Codex clients that read the default config.")
    else:
        print("Global Codex settings preserved. Use codex-away or the resource run command to select a model.")


def visible_devices_for_gpus(gpus: int) -> str:
    """Map a GPU count to a comma-separated list of device indices (0..gpus-1)."""
    if gpus < 1:
        raise ValueError("--gpus must be >= 1")
    return ",".join(str(i) for i in range(gpus))


def resolve_host(cfg, host_arg: str | None, command: str):
    """Apply --host / REMOTE_HOST to cfg; ask interactively when several hosts exist.
    Returns None if the user canceled host selection."""
    host_name = host_arg or os.environ.get("REMOTE_HOST")
    if not host_name and cfg.active_host:
        return cfg
    if not host_name and command in ("serve", "list-models", "select-model", "server-status", "server-cancel", "infer"):
        available = [h.label for h in cfg.host_configs()]
        if len(available) > 1:
            host_name = choose_host(available, cfg.ssh.host)
            if host_name is None:
                return None
    if host_name:
        try:
            cfg = cfg.with_host(host_name)
        except ValueError as exc:
            print(f"llm-away: {exc}", file=sys.stderr)
            return None
        print(f"llm-away: using host {cfg.ssh.destination}", file=sys.stderr)
    return cfg


def main(argv: list[str] | None = None) -> int:
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", default=str(default_config_path()), help="Path to model.toml")
    config_parent.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help="Local provider port for this instance, overriding [server].port (use different ports for parallel agents)",
    )
    config_parent.add_argument(
        "--host",
        default=None,
        metavar="NAME",
        help="Remote host to use (a [hosts] name from model.toml or the default [ssh].host); "
             "overrides REMOTE_HOST. Omit to ask interactively when several hosts are configured.",
    )

    parser = argparse.ArgumentParser(prog="llm-away")
    parser.add_argument("--config", default=str(default_config_path()), help="Path to model.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser("serve", parents=[config_parent], help="Run the local OpenAI-compatible provider (one per concurrent agent)")
    serve_parser.add_argument("--warm", action="store_true", help="Start the remote backend immediately")
    serve_parser.add_argument(
        "--slurm-job",
        type=int,
        default=None,
        metavar="PORT",
        help="Remote port of a second llama-server allocation to use; omit to reuse the default allocation. "
             "When set to a port the running job does not serve, a new Slurm job is submitted on that port.",
    )
    serve_parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        metavar="N",
        help="Full Slurm nodes per allocation, overriding [slurm].nodes (multi-node needs a "
             "multi-node-capable remote wrapper + Slurm IB config; default 1).",
    )
    serve_parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        metavar="N",
        help="GPUs per allocation (e.g. 2-8 on the H200 host), overriding [hosts.NAME].gpus. "
             "Only used by hosts that support partial GPU allocations.",
    )
    sub.add_parser("codex-config", parents=[config_parent], help="Print a Codex config.toml snippet")
    setup_parser = sub.add_parser("configure-local", parents=[config_parent], help="Select your SSH alias and shared AWAY installation")
    setup_parser.add_argument("--ssh-alias", help="AWAY login-node Host alias from ~/.ssh/config")
    setup_parser.add_argument("--remote-workdir")
    setup_parser.add_argument("--restart", action="store_true")
    setup_parser.add_argument("--connection", choices=("ssh", "local"))
    setup_parser.add_argument("--model")
    setup_parser.add_argument("--mtp", choices=("auto", "on", "off"))
    setup_parser.add_argument("--skip-model-selection", action="store_true")
    list_parser = sub.add_parser("list-models", parents=[config_parent], help="Query installed managed models on the SSH host")
    list_parser.add_argument("--json", action="store_true", help="Print model metadata as JSON")
    select_parser = sub.add_parser("select-model", parents=[config_parent], help="Query remote models and save a choice for AWAY")
    select_parser.add_argument("--model", help="Select an installed preset by name without prompting")
    select_parser.add_argument("--mtp", choices=("auto", "on", "off"),
                               help="MTP: follow remote preset (auto), enable, or disable")
    install_parser = sub.add_parser("install-codex-config", parents=[config_parent], help="Install the Codex AWAY provider/profile")
    activation = install_parser.add_mutually_exclusive_group()
    activation.add_argument("--activate", action="store_true", help="Make AWAY the global default, replacing the model, provider, reasoning effort, and catalog settings")
    activation.add_argument("--no-activate", action="store_true", help="Preserve global Codex settings (default; retained for compatibility)")
    sub.add_parser("server-status", parents=[config_parent], help="Show persistent Slurm server status")
    sub.add_parser("server-cancel", parents=[config_parent], help="Cancel the persistent Slurm server job")

    infer_parser = sub.add_parser("infer", parents=[config_parent], help="Run one prompt through the configured backend")
    infer_parser.add_argument("prompt")

    args = parser.parse_args(argv)

    if args.command == "configure-local":
        from .onboarding import configure_local
        try:
            configure_local(args.config, args.ssh_alias, args.remote_workdir,
                            args.restart, args.model, args.mtp, args.skip_model_selection, args.connection)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except (KeyboardInterrupt, EOFError):
            print("Setup canceled.", file=sys.stderr)
            return 1
        return 0

    if args.command in ("list-models", "select-model"):
        try:
            cfg = resolve_host(load_config(args.config), args.host, args.command)
            if cfg is None:
                return 130
            models = discover_models(cfg)
            if args.command == "list-models":
                if args.json:
                    print(json.dumps(models, indent=2))
                elif not models:
                    print("No installed managed models found on the remote host.")
                else:
                    for model in models:
                        print(f"{model['name']}  ({model['size_bytes'] / 1024**3:.1f} GiB)  {model['path']}  [{mtp_label(model)}]")
                return 0
            model = choose_model(models, cfg.llamacpp.model_name, args.model)
            current_mtp = cfg.llamacpp.mtp if model["name"] == cfg.llamacpp.model_name else "auto"
            print(mtp_label(model))
            mtp = choose_mtp(model, current_mtp, args.mtp, interactive=args.model is None)
            save_model(args.config, model, mtp=mtp)
            install_codex_config(args.config)
            print(f"Selected {model['name']}; MTP: {mtp}. Restart the AWAY provider and start a new AWAY Codex session to use it.")
            print("If an allocation is still running, use llm-away server-cancel with this config before restarting to apply MTP changes.")
            return 0
        except (OSError, ValueError, subprocess.SubprocessError, EOFError) as exc:
            print(f"llm-away: {exc}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\nModel selection canceled.", file=sys.stderr)
            return 130

    if args.command == "codex-config":
        print_codex_config(args.config)
        return 0

    if args.command == "install-codex-config":
        install_codex_config(args.config, activate=args.activate)
        return 0

    cfg = resolve_host(load_config(args.config), args.host, args.command)
    if cfg is None:
        return 130
    if args.command == "serve":
        gpus = args.gpus if args.gpus is not None else os.environ.get("REMOTE_GPUS")
        if gpus is not None:
            try:
                count = int(gpus)
                visible_devices_for_gpus(count)  # Validate without overriding Slurm's device mask.
                cfg = dataclasses.replace(cfg, slurm=dataclasses.replace(cfg.slurm, gpus=count),
                                          llamacpp=dataclasses.replace(cfg.llamacpp, visible_devices=""))
            except ValueError as exc:
                print(f"llm-away: {exc}", file=sys.stderr)
                return 2
    if args.port is not None:
        cfg = dataclasses.replace(
            cfg,
            server=dataclasses.replace(cfg.server, port=args.port),
        )
    backend = make_backend(cfg)

    if args.command == "serve":
        if args.slurm_job is not None:
            backend.job_port = args.slurm_job
        if args.nodes is not None:
            backend.slurm_nodes = args.nodes
        serve(cfg, backend, warm=args.warm)
        return 0

    if args.command == "server-status":
        if not isinstance(backend, SlurmServerBackend):
            print("server-status requires backend.type = \"slurm_server\"", file=sys.stderr)
            return 2
        print(backend.serverctl("status", cfg.model.name))
        return 0

    if args.command == "server-cancel":
        if not isinstance(backend, SlurmServerBackend):
            print("server-cancel requires backend.type = \"slurm_server\"", file=sys.stderr)
            return 2
        print(backend.serverctl("cancel", cfg.model.name))
        return 0

    if args.command == "infer":
        print(backend.infer(InferenceRequest(prompt=args.prompt, model=cfg.model.name)))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

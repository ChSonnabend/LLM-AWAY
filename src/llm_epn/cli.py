from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

from .backends import InferenceRequest, SlurmServerBackend, make_backend
from .config import load_config
from .server import serve


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "epn.toml"


def print_codex_config(config_path: str) -> None:
    cfg = load_config(config_path)
    base_url = f"http://{cfg.server.host}:{cfg.server.port}/v1"
    print(
        f'''# Add this provider block to ~/.codex/config.toml.
[model_providers.epn]
name = "{codex_provider_display_name(cfg.codex.provider_display_name)}"
base_url = "{base_url}"
wire_api = "responses"

# Add this profile block to ~/.codex/epn.config.toml.
model_provider = "epn"
model = "{cfg.model.name}"
model_reasoning_effort = "high"
model_catalog_json = "~/.codex/model-catalogs/epn.json"
'''
    )


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()


def codex_provider_display_name(configured_name: str = "") -> str:
    return (
        os.environ.get("LLM_EPN_CODEX_PROVIDER_NAME")
        or configured_name
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or "EPN"
    )


def quoted(value: str) -> str:
    return json.dumps(value)


def epn_model_catalog(model: str) -> dict:
    return {
        "models": [
            {
                "slug": model,
                "display_name": "EPN Qwen3 Coder Next F16 1M",
                "description": "Local EPN Slurm-hosted llama.cpp server via llm-epn gateway.",
                "default_reasoning_level": "medium",
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
                "support_verbosity": True,
                "default_verbosity": "low",
                "truncation_policy": {
                    "mode": "tokens",
                    "limit": 10000,
                },
                "context_window": 131072,
                "max_context_window": 1010000,
                "effective_context_window_percent": 75,
                "input_modalities": ["text"],
                "supports_image_detail_original": False,
                "supports_search_tool": False,
                "supports_experimental_context": False,
                "experimental_supported_tools": [],
                "tool_mode": "code_mode_only",
                "web_search_tool_type": "text_and_image",
                "apply_patch_tool_type": "freeform",
                "node_repl_disabled": False,
                "node_repl_auto_review_required": False,
                "include_apps_usage_instructions": True,
                "include_plugin_usage_instructions": True,
                "include_skills_usage_instructions": False,
                "use_responses_lite": True,
                "default_reasoning_summary": "none",
                "multi_agent_reasoning_effort": "high",
                "multi_agent_version": "v2",
                "model_messages": {
                    "instructions_template": (
                        "You are Qwen3, a coding agent. You and the user share one workspace. "
                        "Help with coding, debugging, editing files, and explaining technical work. "
                        "Be concise, inspect the repository before changing code, and preserve user work. "
                        "When the user asks to inspect, explain, diagnose, review, summarize, or tell what code does, "
                        "use read-only commands and then answer; do not modify files. "
                        "Only edit files when the user explicitly asks for a change. "
                        "When editing, use apply_patch instead of shell heredocs or redirection."
                    )
                },
                "comp_hash": f"local-epn-{model}",
            }
        ]
    }


def write_model_catalogs(home: Path, model: str) -> tuple[Path, Path]:
    catalog_dir = home / "model-catalogs"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    epn_path = catalog_dir / "epn.json"
    combined_path = catalog_dir / "combined-with-epn.json"
    epn_catalog = epn_model_catalog(model)
    epn_path.write_text(json.dumps(epn_catalog, indent=2) + "\n", encoding="utf-8")

    combined = epn_catalog
    if combined_path.exists():
        try:
            combined = json.loads(combined_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            combined = epn_catalog
    models = [entry for entry in combined.get("models", []) if entry.get("slug") != model]
    models.append(epn_catalog["models"][0])
    combined["models"] = models
    combined_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    return epn_path, combined_path


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
                if result and result[-1].strip() == "# LLM EPN provider":
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


def install_codex_config(config_path: str, activate: bool = True) -> None:
    cfg = load_config(config_path)
    home = codex_home()
    home.mkdir(parents=True, exist_ok=True)
    config_file = home / "config.toml"
    profile_file = home / "epn.config.toml"

    epn_catalog, combined_catalog = write_model_catalogs(home, cfg.model.name)
    base_url = f"http://{cfg.server.host}:{cfg.server.port}/v1"

    existing = config_file.read_text(encoding="utf-8") if config_file.exists() else ""
    updated = remove_toml_table(existing, "model_providers.epn")
    if activate:
        updated = set_root_keys(
            updated,
            {
                "model": cfg.model.name,
                "model_provider": "epn",
                "model_reasoning_effort": "high",
                "model_catalog_json": str(combined_catalog),
            },
        )
    provider_block = f'''
# LLM EPN provider
[model_providers.epn]
name = "{codex_provider_display_name(cfg.codex.provider_display_name)}"
base_url = "{base_url}"
wire_api = "responses"
'''
    final_config = updated.rstrip() + "\n\n" + provider_block.lstrip()
    if final_config != existing:
        if existing:
            backup = home / f"config.toml.bak-before-llm-epn-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            backup.write_text(existing, encoding="utf-8")
            openai_copy = home / "config-openai.toml"
            if not openai_copy.exists():
                openai_copy.write_text(existing, encoding="utf-8")
        config_file.write_text(final_config, encoding="utf-8")

    profile_text = (
        "\n".join(
            [
                'model_provider = "epn"',
                f"model = {quoted(cfg.model.name)}",
                'model_reasoning_effort = "high"',
                f"model_catalog_json = {quoted(str(epn_catalog))}",
                "",
            ]
        )
    )
    if not profile_file.exists() or profile_file.read_text(encoding="utf-8") != profile_text:
        profile_file.write_text(profile_text, encoding="utf-8")
    if activate:
        shutil.copy2(config_file, home / "config-epn.toml")

    print(f"Installed EPN Codex provider in {config_file}")
    print(f"Installed EPN profile in {profile_file}")
    print(f"Installed EPN model catalog in {epn_catalog}")
    if activate:
        print("EPN is active for Codex clients that read the default config.")


def main(argv: list[str] | None = None) -> int:
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", default=str(default_config_path()), help="Path to epn.toml")

    parser = argparse.ArgumentParser(prog="llm-epn")
    parser.add_argument("--config", default=str(default_config_path()), help="Path to epn.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser("serve", parents=[config_parent], help="Run the local OpenAI-compatible provider")
    serve_parser.add_argument("--warm", action="store_true", help="Start the remote backend immediately")
    sub.add_parser("codex-config", parents=[config_parent], help="Print a Codex config.toml snippet")
    install_parser = sub.add_parser("install-codex-config", parents=[config_parent], help="Install the Codex EPN provider/profile")
    install_parser.add_argument("--no-activate", action="store_true", help="Install provider/profile without making EPN the default")
    sub.add_parser("server-status", parents=[config_parent], help="Show persistent Slurm server status")
    sub.add_parser("server-cancel", parents=[config_parent], help="Cancel the persistent Slurm server job")

    infer_parser = sub.add_parser("infer", parents=[config_parent], help="Run one prompt through the configured backend")
    infer_parser.add_argument("prompt")

    args = parser.parse_args(argv)

    if args.command == "codex-config":
        print_codex_config(args.config)
        return 0

    if args.command == "install-codex-config":
        install_codex_config(args.config, activate=not args.no_activate)
        return 0

    cfg = load_config(args.config)
    backend = make_backend(cfg)

    if args.command == "serve":
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

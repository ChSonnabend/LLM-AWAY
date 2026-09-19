"""Agent selection and process-local Claude Code configuration."""
from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from .prompt import choose_option


def choose_cli(preference="auto", available=None):
    if preference not in ("auto", "codex", "claude"):
        raise ValueError("CLI must be auto, codex, or claude")
    if available is None:
        available = [name for name in ("codex", "claude") if shutil.which(name)]
    if preference != "auto":
        if preference not in available:
            raise ValueError(f"{preference} is not available on PATH")
        return preference
    if not available:
        raise ValueError("Install Codex or Claude Code and make it available on PATH")
    if len(available) == 1:
        return available[0]
    if not sys.stdin.isatty():
        raise ValueError("Both CLIs are available; specify --cli codex or --cli claude (or LLM_AWAY_CLI)")
    return available[choose_option(available, "Which CLI do you want to use?")]


def run_foreground(path, data, selection):
    if selection.get("cli") != "codex":
        raise ValueError("tmux is required to keep this agent running in the background")
    spec = dict(selection, token=data["token"], model=data["model"], state=str(path),
                base_url="http://127.0.0.1:" + str(data["config"]["server"]["port"]))
    if selection.get("codex_catalog_content"):
        catalog = Path(path) / "codex-model-catalog.json"
        catalog.write_text(selection["codex_catalog_content"], encoding="utf-8")
        spec["codex_catalog"] = str(catalog)
    engine = runpy.run_path(str(Path(__file__).resolve().parents[2] / "remote/bin/resource-agent"))
    command, env = engine["agent_command"](spec, spec["base_url"])
    command.remove("--json")
    command.remove("-")
    command.insert(1, "--no-alt-screen")
    command += selection.get("extra_args", [])
    if selection.get("rag_command"):
        rag = selection["rag_command"]
        command += ["-c", "mcp_servers.project_search.command=" + json.dumps(rag[0]),
                    "-c", "mcp_servers.project_search.args=" + json.dumps(rag[1:])]
    cwd = Path(selection.get("cwd") or path)
    return subprocess.call(command, cwd=str(cwd), env=env)


def claude_launch(cfg, base_url, model, rag_command=None):
    env = dict(os.environ)
    for key in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
                "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        env.pop(key, None)
    env.update(ANTHROPIC_BASE_URL=base_url, ANTHROPIC_AUTH_TOKEN="llm-away",
               ANTHROPIC_MODEL=model, ANTHROPIC_DEFAULT_OPUS_MODEL=model,
               ANTHROPIC_DEFAULT_SONNET_MODEL=model, ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
               CLAUDE_CODE_SUBAGENT_MODEL=model, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
               DISABLE_PROMPT_CACHING="1")
    command = ["claude", "--model", model]
    instructions = cfg.claude.instructions or cfg.codex.instructions
    if instructions:
        command += ["--append-system-prompt", instructions]
    if rag_command:
        command += ["--mcp-config", json.dumps({"mcpServers": {"project_search": {
            "command": rag_command[0], "args": rag_command[1:]}}})]
    return command, env


def main():
    from .config import load_config
    cfg = load_config(sys.argv[1])
    command, env = claude_launch(cfg, sys.argv[2], cfg.model.name)
    os.execvpe(command[0], command + sys.argv[3:], env)


if __name__ == "__main__":
    main()

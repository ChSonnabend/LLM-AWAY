"""Agent selection and process-local Claude Code configuration."""
from __future__ import annotations

import json
import os
import shutil
import sys

from .prompt import choose_option


def choose_cli(preference="auto"):
    if preference not in ("auto", "codex", "claude"):
        raise ValueError("CLI must be auto, codex, or claude")
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

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${LLM_EPN_SSH_HOST:-epnh}"
REMOTE_BIN="${LLM_EPN_REMOTE_BIN:-.local/bin}"

ssh "$HOST" "mkdir -p ~/$REMOTE_BIN"
scp "$ROOT/scripts/remote/llm-epn-slurm-run" "$HOST:~/$REMOTE_BIN/llm-epn-slurm-run"
scp "$ROOT/scripts/remote/llm-epn-serverctl" "$HOST:~/$REMOTE_BIN/llm-epn-serverctl"
ssh "$HOST" "chmod +x ~/$REMOTE_BIN/llm-epn-slurm-run"
ssh "$HOST" "chmod +x ~/$REMOTE_BIN/llm-epn-serverctl"

echo "Installed remote helpers on $HOST:~/$REMOTE_BIN/"

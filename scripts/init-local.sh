#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv

mkdir -p "$HOME/.local/bin"

install_link() {
    local source_path="$1"
    local dest_path="$HOME/.local/bin/$(basename "$source_path")"
    if [[ -e "$dest_path" && ! -L "$dest_path" ]]; then
        echo "Leaving existing $dest_path in place; add $source_path to PATH manually if needed."
        return
    fi
    ln -sfn "$source_path" "$dest_path"
}

install_link "$ROOT/bin/llm-epn"
install_link "$ROOT/bin/epn-agent"
install_link "$ROOT/bin/code-epn"
install_link "$ROOT/bin/codex-epn"

"$ROOT/bin/llm-epn" install-codex-config --config "$ROOT/config/epn.toml"

echo "Local llm-epn environment is ready."
echo
echo "For an already-open local or Remote-SSH VS Code window, start the EPN Codex provider with:"
echo "  epn-agent"
echo
echo "To open VS Code and keep the provider alive until the window closes, run:"
echo "  code-epn /path/to/project"
echo
echo "If ~/.local/bin is not on PATH, use:"
echo "  $ROOT/bin/epn-agent"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
mkdir -p "$HOME/.local/bin"
# Remove only obsolete links installed by this repository.
for name in add-serve res-background; do
    dest="$HOME/.local/bin/$name"
    if [[ -L "$dest" && "$(readlink "$dest")" == "$ROOT/bin/$name" ]]; then
        rm "$dest"
    fi
done
for name in resource-allocator resource-monitor run res-alloc res-mon res-mon-web session-tool res-clean codex-away claude-away; do
    case "$name" in res-alloc) target=resource-allocator ;; res-mon) target=resource-monitor ;; *) target=$name ;; esac
    dest="$HOME/.local/bin/$name"
    if [[ -e "$dest" && ! -L "$dest" ]]; then
        echo "Leaving existing $dest; use $ROOT/bin/$target directly."
        continue
    fi
    ln -sfn "$ROOT/bin/$target" "$dest"
done
echo 'Ready: res-alloc, run --session N, res-mon.'
echo 'To add ~/.local/bin to PATH, run:'
echo '    export PATH="$HOME/.local/bin:$PATH"'

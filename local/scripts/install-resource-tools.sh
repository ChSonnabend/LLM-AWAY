#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$HOME/.local/bin"
for name in resource-allocator resource-monitor run res-alloc res-mon session-tool add-serve res-clean; do
    case "$name" in res-alloc) target=resource-allocator ;; res-mon) target=resource-monitor ;; *) target=$name ;; esac
    dest="$HOME/.local/bin/$name"
    if [[ -e "$dest" && ! -L "$dest" ]]; then
        echo "Leaving existing $dest; use $ROOT/bin/$target directly."
        continue
    fi
    ln -sfn "$ROOT/bin/$target" "$dest"
done
echo 'Ready: res-alloc, run --session N, res-mon. Ensure ~/.local/bin is on PATH.'

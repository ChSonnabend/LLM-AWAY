#!/usr/bin/env bash
set -euo pipefail

ROOT="$(python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve().parent.parent)' "${BASH_SOURCE[0]}")"
cd "$ROOT"

config="${LLM_REMOTE_CONFIG:-$ROOT/config/away.toml}"
selection_args=()
mtp_args=()
connection_args=()
skip_selection=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--config|--mtp|--ssh-alias|--remote-workdir)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "$1 requires a value" >&2
                exit 2
            fi
            if [[ "$1" == --model ]]; then selection_args=(--model "$2");
            elif [[ "$1" == --mtp ]]; then mtp_args=(--mtp "$2");
            elif [[ "$1" == --config ]]; then config="$2";
            else connection_args+=("$1" "$2"); fi
            shift 2
            ;;
        --skip-model-selection) skip_selection=1; shift ;;
        --restart) connection_args+=(--restart); shift ;;
        -h|--help)
            echo "Usage: init-local.sh [--restart] [--config PATH] [--ssh-alias NAME] [--remote-workdir PATH] [--model NAME] [--mtp auto|on|off] [--skip-model-selection]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ "$skip_selection" == 1 && ( ${#selection_args[@]} -gt 0 || ${#mtp_args[@]} -gt 0 ) ]]; then
    echo "--model/--mtp and --skip-model-selection cannot be combined" >&2
    exit 2
fi
if [[ "$skip_selection" == 1 ]]; then connection_args+=(--skip-model-selection); fi
"$ROOT/bin/llm-away" configure-local --config "$config" ${connection_args[@]+"${connection_args[@]}"} ${selection_args[@]+"${selection_args[@]}"} ${mtp_args[@]+"${mtp_args[@]}"}

mkdir -p "$HOME/.local/bin"

install_link() {
    local source_path="$1"
    local dest_path="$HOME/.local/bin/$(basename "$source_path")"
    if [[ -L "$dest_path" && "$(readlink "$dest_path")" == "$source_path" ]]; then
        return
    fi
    if [[ -e "$dest_path" && ! -L "$dest_path" ]]; then
        echo "Leaving existing $dest_path in place; add $source_path to PATH manually if needed."
        return
    fi
    rm -f "$dest_path"
    ln -s "$source_path" "$dest_path"
}

install_link "$ROOT/bin/llm-away"
install_link "$ROOT/bin/away-agent"
install_link "$ROOT/bin/code-away"
install_link "$ROOT/bin/codex-away"

"$ROOT/bin/llm-away" install-codex-config --config "$config" --no-activate

echo "Local llm-away environment is ready."
echo
echo "For an already-open local or Remote-SSH VS Code window, start the AWAY Codex provider with:"
echo "  away-agent"
echo
echo "To open VS Code and keep the provider alive until the window closes, run:"
echo "  code-away /path/to/project"
echo
echo "If ~/.local/bin is not on PATH, use:"
echo "  $ROOT/bin/away-agent"

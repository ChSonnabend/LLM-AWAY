#!/usr/bin/env bash
# Unified, repeatable local setup. Existing installer names delegate here.
set -euo pipefail
AWAY_LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AWAY_REPO_ROOT="$(cd "$AWAY_LOCAL_ROOT/.." && pwd)"
AWAY_BIN_DIR="${HOME}/.local/bin"
AWAY_VSCODE_MODE=auto
AWAY_MODE=all
AWAY_DOCUMENTS=none
AWAY_CODE_COMMAND=""

usage() {
    cat <<'USAGE'
Usage: local/scripts/install.sh [options]

Default: prepare Python, install command links, and install the VS Code inline
helper when a local VS Code CLI, Node.js 20+ and npm are available.

  --with-vscode       Require the VS Code extension; failures return nonzero
  --without-vscode    Install the framework commands only
  --vscode-only       Install only the extension (no Python setup or command links)
  --env-only          Prepare Python only (compatibility mode for scripts/init)
  --documents         Also install optional document/OCR dependencies
  --check-documents   Check document/OCR dependencies without installing them
  --bin-dir PATH      Command link directory (default: ~/.local/bin)
  --code-command PATH VS Code CLI to use, e.g. code-insiders
  -h, --help          Show this help without installing anything

Auto mode skips extension setup in SSH/VS Code Remote shells and when tools are
missing. Use --with-vscode or --vscode-only to explicitly target that environment.
Downloading or cloning the repository alone does not execute this installer.
USAGE
}

while (($#)); do
    case "$1" in
        --with-vscode) AWAY_VSCODE_MODE=required ;;
        --without-vscode) AWAY_VSCODE_MODE=never ;;
        --vscode-only)
            if [[ "$AWAY_MODE" == env ]]; then echo 'Cannot combine --env-only and --vscode-only.' >&2; exit 2; fi
            AWAY_MODE=vscode; AWAY_VSCODE_MODE=required ;;
        --env-only)
            if [[ "$AWAY_MODE" == vscode ]]; then echo 'Cannot combine --env-only and --vscode-only.' >&2; exit 2; fi
            AWAY_MODE=env; AWAY_VSCODE_MODE=never ;;
        --documents|--check-documents)
            if [[ "$AWAY_DOCUMENTS" != none ]]; then echo 'Choose one document option.' >&2; exit 2; fi
            AWAY_DOCUMENTS="$1" ;;
        --bin-dir|--code-command)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then echo "Missing value for $1" >&2; exit 2; fi
            if [[ "$1" == --bin-dir ]]; then AWAY_BIN_DIR="$2"; else AWAY_CODE_COMMAND="$2"; fi
            shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
if [[ "$AWAY_MODE" == vscode && ( "$AWAY_VSCODE_MODE" != required || "$AWAY_DOCUMENTS" != none ) ]] ||
   [[ "$AWAY_MODE" == env && "$AWAY_VSCODE_MODE" != never ]]; then
    echo 'Incompatible installation options.' >&2; exit 2
fi

install_command_links() {
    mkdir -p "$AWAY_BIN_DIR"
    for name in add-serve res-background; do
        dest="$AWAY_BIN_DIR/$name"
        if [[ -L "$dest" && "$(readlink "$dest")" == "$AWAY_LOCAL_ROOT/bin/$name" ]]; then rm "$dest"; fi
    done
    for name in resource-allocator resource-monitor run res-alloc res-mon res-mon-web session-tool res-clean codex-away claude-away; do
        case "$name" in res-alloc) target=resource-allocator ;; res-mon) target=resource-monitor ;; *) target=$name ;; esac
        dest="$AWAY_BIN_DIR/$name"
        if [[ -L "$dest" ]]; then
            if [[ "$(readlink "$dest")" != "$AWAY_LOCAL_ROOT/bin/$target" ]]; then
                echo "Leaving existing link $dest; use $AWAY_LOCAL_ROOT/bin/$target directly."
                continue
            fi
        elif [[ -e "$dest" ]]; then
            echo "Leaving existing $dest; use $AWAY_LOCAL_ROOT/bin/$target directly."
            continue
        fi
        ln -sfn "$AWAY_LOCAL_ROOT/bin/$target" "$dest"
    done
    echo 'Framework commands ready: res-alloc, run --session N, res-mon.'
    printf 'Command directory: %s (add it to PATH if needed).\n' "$AWAY_BIN_DIR"
}

vscode_available() {
    if [[ "$AWAY_VSCODE_MODE" == auto && ( -n "${SSH_CONNECTION:-}" || -n "${SSH_TTY:-}" ) ]]; then
        echo 'Skipping VS Code extension in an SSH shell; run setup on your desktop.'; return 1
    fi
    if [[ -z "$AWAY_CODE_COMMAND" ]]; then
        AWAY_CODE_COMMAND="$(command -v code || true)"
        if [[ -z "$AWAY_CODE_COMMAND" && -x '/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code' ]]; then
            AWAY_CODE_COMMAND='/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code'
        fi
    fi
    if [[ -z "$AWAY_CODE_COMMAND" ]] || ! command -v "$AWAY_CODE_COMMAND" >/dev/null 2>&1; then
        echo 'VS Code CLI not found. Install the code command in PATH or use --code-command.'; return 1
    fi
    AWAY_CODE_COMMAND="$(command -v "$AWAY_CODE_COMMAND")"
    if [[ "$AWAY_VSCODE_MODE" == auto && ( "$AWAY_CODE_COMMAND" == *vscode-server* || "$AWAY_CODE_COMMAND" == *remote-cli* ) ]]; then
        echo 'Skipping the VS Code Remote CLI; run setup on the desktop side.'; return 1
    fi
    if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
        echo 'Extension setup requires Node.js 20+ and npm.'; return 1
    fi
    if ! node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 20 ? 0 : 1)'; then
        echo 'Extension setup requires Node.js 20 or newer.'; return 1
    fi
}

install_vscode() (
    # Called with explicit error checks so auto-mode failures cannot fall through
    # and accidentally install a stale package left by a previous run.
    cd "$AWAY_REPO_ROOT/vscode-inline" || return 1
    version="$(node -p 'require("./package.json").version')" || return 1
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]]; then
        echo 'Invalid extension version in package.json.' >&2; return 1
    fi
    artifacts="$AWAY_LOCAL_ROOT/run/artifacts"
    mkdir -p "$artifacts" || return 1
    package="$artifacts/llm-away-inline-$version.vsix"
    npm ci --include=dev --ignore-scripts || return 1
    npm run package -- --out "$package" || return 1
    "$AWAY_CODE_COMMAND" --install-extension "$package" --force || return 1
    echo "VS Code inline helper $version installed."
    echo 'Reload the VS Code window, then run: LLM-AWAY: Select Inline Helper Session'
)

if [[ "$AWAY_MODE" != vscode ]]; then
    source "$AWAY_LOCAL_ROOT/scripts/env.sh"
    echo "Project environment ready: $VIRTUAL_ENV"
    if [[ "$AWAY_MODE" == all ]]; then install_command_links; fi
    case "$AWAY_DOCUMENTS" in
        --documents) "$VIRTUAL_ENV/bin/python" -m llm_away.document_tools --install ;;
        --check-documents) "$VIRTUAL_ENV/bin/python" -m llm_away.document_tools --check ;;
        *) echo 'Optional PDF/OCR dependencies: local/scripts/install.sh --documents' ;;
    esac
fi

if [[ "$AWAY_VSCODE_MODE" != never ]]; then
    if vscode_available; then
        if ! install_vscode; then
            echo 'VS Code extension installation failed. Retry: local/scripts/install.sh --vscode-only' >&2
            if [[ "$AWAY_VSCODE_MODE" == required ]]; then exit 1; fi
            echo 'Framework setup completed; VS Code extension setup is incomplete.' >&2
        fi
    elif [[ "$AWAY_VSCODE_MODE" == required ]]; then
        exit 1
    else
        echo 'VS Code extension not installed. Add the missing tools and rerun setup, or use --without-vscode for CLI-only setup.'
    fi
fi

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
Explicit VS Code setup installs missing Node.js/npm with Homebrew when available.
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

node_ready() {
    command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 &&
        node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 20 ? 0 : 1)' >/dev/null 2>&1 &&
        npm --version >/dev/null 2>&1
}

prepare_node() {
    local brew_command prefix formula saved_path="$PATH"
    if node_ready; then return 0; fi
    brew_command="$(command -v brew || true)"
    if [[ -z "$brew_command" ]]; then
        for brew_command in /opt/homebrew/bin/brew /usr/local/bin/brew; do
            if [[ -x "$brew_command" ]]; then break; fi
        done
    fi
    if [[ -x "$brew_command" ]]; then
        # Keg-only Node installations need not be linked into the user's PATH.
        for formula in node node@24 node@22 node@20; do
            prefix="$("$brew_command" --prefix "$formula" 2>/dev/null)" || continue
            [[ -x "$prefix/bin/node" && -x "$prefix/bin/npm" ]] || continue
            export PATH="$prefix/bin:$saved_path"
            if node_ready; then return 0; fi
            export PATH="$saved_path"
        done
        if [[ "$AWAY_VSCODE_MODE" == required ]]; then
            echo 'Installing Node.js and npm with Homebrew for VS Code extension setup...'
            if ! "$brew_command" install node; then
                echo 'Homebrew could not prepare Node.js. Fix the Homebrew error above and retry with --vscode-only.' >&2
                return 1
            fi
            prefix="$("$brew_command" --prefix node)" || return 1
            export PATH="$prefix/bin:$saved_path"
            if node_ready; then return 0; fi
            export PATH="$saved_path"
        fi
    fi
    echo 'Extension setup requires working Node.js 20+ and npm. Install Node.js (https://nodejs.org/) or run: brew install node' >&2
    echo 'Then retry: local/scripts/install.sh --vscode-only' >&2
    return 1
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
    prepare_node
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

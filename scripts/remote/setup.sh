#!/usr/bin/env bash
# Run on the Slurm login host; this directory is a self-contained setup bundle.
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workdir="$(cd "$SOURCE_DIR/../.." && pwd)"
bin_dir="${LLM_REMOTE_REMOTE_BIN:-$HOME/.local/bin}"
mode=install
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) mode=check; shift ;;
        --workdir|--bin-dir)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "$1 requires a path" >&2
                exit 2
            fi
            if [[ "$1" == --workdir ]]; then workdir=$2; else bin_dir=$2; fi
            shift 2
            ;;
        -h|--help)
            echo "Usage: setup.sh [--check] [--workdir PATH] [--bin-dir PATH]"
            echo "Run on the Slurm login host to install the adjacent AWAY helpers."
            echo "Default workdir: two directories above this script (the remote runner project)."
            echo "Default bin directory: \$HOME/.local/bin; relative overrides are home-relative."
            echo "Checks prerequisites first; backs up replacements and skips current files."
            echo "--check verifies the setup without changing files or submitting jobs."
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$bin_dir" in /*) ;; *) bin_dir="$HOME/$bin_dir" ;; esac

# Complete preflight before creating any installation files.
for command in python3 sbatch squeue scancel sinfo srun timeout cmp mktemp cp chmod mv; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Missing command: $command. Run setup on the Slurm login host with its environment loaded." >&2
        exit 1
    fi
done
python3 -c 'import sys; sys.exit("Python 3.6 or newer is required") if sys.version_info < (3, 6) else None'
for file in bin/run-server bin/run-cli scripts/lib/llamacpp-env.sh; do
    if [[ ! -r "$workdir/$file" ]]; then
        echo "Missing remote runner file: $workdir/$file (set --workdir if needed)" >&2
        exit 1
    fi
done
for name in llm-away-slurm-run llm-away-serverctl ensure-container.py; do
    if [[ ! -r "$SOURCE_DIR/$name" ]]; then
        echo "Missing bundled helper: $SOURCE_DIR/$name" >&2
        exit 1
    fi
    if [[ -L "$bin_dir/$name" || ( -e "$bin_dir/$name" && ! -f "$bin_dir/$name" ) ]]; then
        echo "Refusing a symlink or non-regular destination: $bin_dir/$name" >&2
        exit 1
    fi
done

install_helper() (
    name=$1
    source="$SOURCE_DIR/$name"
    dest="$bin_dir/$name"
    if [[ -f "$dest" && -x "$dest" ]] && cmp -s -- "$source" "$dest"; then
        echo "Up to date: $dest"
        exit 0
    fi
    if [[ "$mode" == check ]]; then
        echo "Missing, different, or not executable: $dest; rerun setup without --check." >&2
        exit 1
    fi
    mkdir -p -- "$bin_dir"
    temporary=$(mktemp "$bin_dir/.$name.XXXXXXXX")
    trap 'rm -f -- "$temporary"' EXIT
    cp -- "$source" "$temporary"
    chmod 755 "$temporary"
    if [[ -e "$dest" ]]; then
        backup=$(mktemp -d "$dest.backup.XXXXXXXX")
        cp -p -- "$dest" "$backup/original"
        echo "Backup: $backup/original"
    fi
    mv -f -- "$temporary" "$dest"
    echo "Installed: $dest"
)

for name in llm-away-slurm-run llm-away-serverctl ensure-container.py; do
    install_helper "$name"
done
echo "Remote AWAY helpers are ready. Model files, GPU builds, and Slurm access must also be configured; see README."

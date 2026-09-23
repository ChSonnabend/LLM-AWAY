# Shared by every local command; sourcing affects only that command's process.
_AWAY_SOURCE="${BASH_SOURCE[0]}"
_AWAY_ROOT="$(cd -P "$(dirname "$_AWAY_SOURCE")/.." && pwd)"
_AWAY_BOOTSTRAP=""
for _AWAY_CANDIDATE in /usr/bin/python3 python3 python3.12 python3.11; do
    if "$_AWAY_CANDIDATE" -I -c 'import sys; sys.exit(sys.version_info < (3,10))' >/dev/null 2>&1; then
        _AWAY_BOOTSTRAP="$_AWAY_CANDIDATE"; break
    fi
done
if [[ -z "$_AWAY_BOOTSTRAP" ]]; then
    echo 'LLM-AWAY requires Python 3.10+ with venv support.' >&2
    exit 1
fi
_AWAY_PYTHON="$("$_AWAY_BOOTSTRAP" -I "$_AWAY_ROOT/scripts/bootstrap.py")"
export VIRTUAL_ENV="$_AWAY_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME
export PYTHONPATH="$_AWAY_ROOT/src"

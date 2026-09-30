#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
cd "$ROOT"
exec "$VIRTUAL_ENV/bin/python" -m unittest discover -s tests -v "$@"

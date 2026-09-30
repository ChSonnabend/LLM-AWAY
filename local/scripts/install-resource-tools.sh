#!/usr/bin/env bash
# Compatibility entry point: includes automatic local VS Code extension setup.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install.sh" "$@"

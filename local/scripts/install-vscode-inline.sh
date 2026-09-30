#!/usr/bin/env bash
# Compatibility entry point for extension-only installation.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install.sh" --vscode-only "$@"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -n ${MODEL_NAME:-} ]]; then
  exec bin/download-model "$MODEL_NAME" "$@"
fi

exec bin/download-model "$@"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export MODEL_NAME="${MODEL_NAME:-deepseek-v2.5-q4km}"

exec bin/run-cli "$@"

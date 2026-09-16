#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LLAMACPP_BACKEND=rocm
export LLAMACPP_ROCM_ARCH="${LLAMACPP_ROCM_ARCH:-gfx906}"

exec bin/build-llama "$@"

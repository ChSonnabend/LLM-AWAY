#!/usr/bin/env bash

LLAMACPP_DEFAULT_MODEL=${LLAMACPP_DEFAULT_MODEL:-qwen3.8-27b-q4km}

llamacpp_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

llamacpp_die() {
  echo "llamacpp-llm: $*" >&2
  exit 1
}

llamacpp_csv_range() {
  local count=${1:-0}
  local values=()
  local i

  for ((i = 0; i < count; i++)); do
    values+=("$i")
  done

  local IFS=,
  echo "${values[*]}"
}

llamacpp_csv_ones() {
  local count=${1:-0}
  local values=()
  local i

  for ((i = 0; i < count; i++)); do
    values+=(1)
  done

  local IFS=,
  echo "${values[*]}"
}

llamacpp_select_python() {
  local candidate

  if [[ -n ${LLAMACPP_PYTHON:-} && -x ${LLAMACPP_PYTHON} ]]; then
    echo "$LLAMACPP_PYTHON"
    return 0
  fi

  for candidate in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$candidate" > /dev/null 2>&1 &&
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' > /dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  return 1
}

llamacpp_ensure_hf_cli() {
  local root
  root=$(llamacpp_root)

  local candidate
  for candidate in \
    "${LLAMACPP_HF_CLI:-}" \
    "$root/venvs/hf-download/bin/hf" \
    "$root/.venv39/bin/hf" \
    "$root/.venv/bin/hf" \
    "$(command -v hf 2> /dev/null || true)" \
    "$(command -v huggingface-cli 2> /dev/null || true)"; do
    if [[ -n $candidate && -x $candidate ]]; then
      echo "$candidate"
      return 0
    fi
  done

  local python
  python=$(llamacpp_select_python) || llamacpp_die "no Python >= 3.9 found for Hugging Face downloads"

  mkdir -p "$root/venvs"
  "$python" -m venv "$root/venvs/hf-download"
  "$root/venvs/hf-download/bin/python" -m pip install -U pip huggingface_hub hf_xet >&2

  echo "$root/venvs/hf-download/bin/hf"
}

llamacpp_detect_backend() {
  if [[ -n ${LLAMACPP_BACKEND:-} ]]; then
    echo "$LLAMACPP_BACKEND"
    return 0
  fi

  if command -v rocminfo > /dev/null 2>&1 || command -v rocm-smi > /dev/null 2>&1 || command -v hipconfig > /dev/null 2>&1; then
    echo rocm
    return 0
  fi

  if command -v nvidia-smi > /dev/null 2>&1 || command -v nvcc > /dev/null 2>&1; then
    echo cuda
    return 0
  fi

  echo cpu
}

llamacpp_detect_rocm_arch() {
  if [[ -n ${LLAMACPP_ROCM_ARCH:-} ]]; then
    echo "$LLAMACPP_ROCM_ARCH"
    return 0
  fi

  if command -v rocminfo > /dev/null 2>&1; then
    local archs
    archs=$(rocminfo 2> /dev/null | sed -n 's/.*Name:[[:space:]]*\(gfx[0-9a-zA-Z]*\).*/\1/p' | sort -u | paste -sd ';' -)
    if [[ -n $archs ]]; then
      echo "$archs"
      return 0
    fi
  fi

  if command -v hipconfig > /dev/null 2>&1; then
    local archs
    archs=$(hipconfig --amdgpu-target 2> /dev/null || true)
    if [[ -n $archs ]]; then
      echo "$archs"
      return 0
    fi
  fi

  if command -v rocm-smi > /dev/null 2>&1; then
    local names
    names=$(rocm-smi --showproductname 2> /dev/null || true)
    if grep -qi 'MI50\|MI60' <<< "$names"; then
      echo gfx906
      return 0
    fi
    if grep -qi 'MI100' <<< "$names"; then
      echo gfx908
      return 0
    fi
    if grep -qi 'MI2[0-9][0-9]' <<< "$names"; then
      echo gfx90a
      return 0
    fi
    if grep -qi 'MI3[0-9][0-9]' <<< "$names"; then
      echo gfx942
      return 0
    fi
  fi

  echo gfx906
}

llamacpp_gpu_count() {
  local backend=${1:-$(llamacpp_detect_backend)}

  if [[ ${LLAMACPP_GPU_COUNT:-} =~ ^[0-9]+$ ]]; then
    echo "$LLAMACPP_GPU_COUNT"
    return 0
  fi

  if [[ $backend == rocm ]]; then
    if [[ -n ${LLAMACPP_VISIBLE_DEVICES:-} ]]; then
      awk -F, '{print NF}' <<< "$LLAMACPP_VISIBLE_DEVICES"
      return 0
    fi
    if [[ -n ${HIP_VISIBLE_DEVICES:-} ]]; then
      awk -F, '{print NF}' <<< "$HIP_VISIBLE_DEVICES"
      return 0
    fi
    if [[ -n ${ROCR_VISIBLE_DEVICES:-} ]]; then
      awk -F, '{print NF}' <<< "$ROCR_VISIBLE_DEVICES"
      return 0
    fi
  elif [[ $backend == cuda ]]; then
    if [[ -n ${LLAMACPP_VISIBLE_DEVICES:-} ]]; then
      awk -F, '{print NF}' <<< "$LLAMACPP_VISIBLE_DEVICES"
      return 0
    fi
    if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
      awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES"
      return 0
    fi
    if command -v nvidia-smi > /dev/null 2>&1; then
      nvidia-smi -L 2> /dev/null | grep -c '^GPU ' || true
      return 0
    fi
  fi

  echo 0
}

llamacpp_export_visible_devices() {
  local backend=${1:-$(llamacpp_detect_backend)}

  if [[ -n ${SLURM_JOB_ID:-} ]]; then
    if [[ $backend == cuda && -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
      LLAMACPP_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    elif [[ $backend == rocm && -n ${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-}} ]]; then
      LLAMACPP_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-$HIP_VISIBLE_DEVICES}
    fi
  fi
  if [[ -n ${LLAMACPP_VISIBLE_DEVICES:-} ]]; then
    if [[ $backend == rocm ]]; then
      export HIP_VISIBLE_DEVICES="$LLAMACPP_VISIBLE_DEVICES"
      export ROCR_VISIBLE_DEVICES="$LLAMACPP_VISIBLE_DEVICES"
    elif [[ $backend == cuda ]]; then
      export CUDA_VISIBLE_DEVICES="$LLAMACPP_VISIBLE_DEVICES"
    fi
    return 0
  fi
}

llamacpp_apply_mtp_mode() {
  local mode=${LLAMACPP_MTP:-auto}
  case "$mode" in
    auto) return 0 ;;
    off)
      # Do this after model.env is loaded: CLI --spec-type flags accumulate.
      if [[ ${LLAMACPP_SPEC_TYPE:-draft-mtp} == draft-mtp ]]; then
        unset MODEL_DRAFT_GGUF
        LLAMACPP_SPEC_TYPE=none
      fi
      ;;
    on)
      [[ -n ${MODEL_DRAFT_GGUF:-} && ${LLAMACPP_SPEC_TYPE:-draft-mtp} == draft-mtp ]] ||
        llamacpp_die "MTP requested but this preset has no configured MTP draft model"
      local draft_path
      draft_path=$(llamacpp_resolve_existing_path "$MODEL_DRAFT_GGUF" "MTP draft model") || return
      [[ -r $draft_path && -s $draft_path ]] || llamacpp_die "MTP draft model is unreadable or empty: $draft_path"
      ;;
    *) llamacpp_die "LLAMACPP_MTP must be auto, on, or off" ;;
  esac
}

llamacpp_load_model_config() {
  local root
  root=$(llamacpp_root)

  MODEL_NAME=${MODEL_NAME:-${1:-$LLAMACPP_DEFAULT_MODEL}}
  MODEL_CONFIG=${MODEL_CONFIG:-$root/models/$MODEL_NAME/model.env}

  if [[ -f $MODEL_CONFIG ]]; then
    # shellcheck disable=SC1090
    source "$MODEL_CONFIG"
  fi

  MODEL_ALIAS=${MODEL_ALIAS:-$MODEL_NAME}
  llamacpp_apply_mtp_mode
}

llamacpp_resolve_model() {
  local root
  root=$(llamacpp_root)

  local model_path=${MODEL:-${MODEL_GGUF:-models/$MODEL_NAME/model.gguf}}
  if [[ $model_path != /* ]]; then
    model_path="$root/$model_path"
  fi

  [[ -e $model_path ]] || llamacpp_die "model not found: $model_path"
  echo "$model_path"
}

llamacpp_resolve_existing_path() {
  local path=$1
  local label=${2:-path}
  local root
  root=$(llamacpp_root)

  if [[ $path != /* ]]; then
    path="$root/$path"
  fi

  [[ -e $path ]] || llamacpp_die "$label not found: $path"
  echo "$path"
}

llamacpp_build_id() {
  local backend=${1:-$(llamacpp_detect_backend)}
  local arch=${2:-}

  if [[ $backend == rocm ]]; then
    echo "rocm-${arch//[^A-Za-z0-9]/_}"
  else
    echo "$backend"
  fi
}

llamacpp_binary() {
  local tool=$1
  local backend=${2:-$(llamacpp_detect_backend)}
  local root build_id arch
  root=$(llamacpp_root)

  if [[ $backend == rocm ]]; then
    arch=$(llamacpp_detect_rocm_arch)
  fi
  build_id=$(llamacpp_build_id "$backend" "${arch:-}")

  local env_name="LLAMA_${tool^^}"
  if [[ -n ${!env_name:-} ]]; then
    [[ -x ${!env_name} ]] || llamacpp_die "configured binary is missing: ${!env_name}"
    echo "${!env_name}"
    return 0
  fi
  if [[ ${LLAMACPP_IN_CONTAINER:-0} == 1 && -x /app/llama-$tool ]]; then
    echo "/app/llama-$tool"
    return 0
  fi
  local candidate
  for candidate in \
    "${!env_name:-}" \
    "${LLAMACPP_BUILD_DIR:-$root/builds/$build_id}/bin/llama-$tool" \
    "$root/builds/$build_id/bin/llama-$tool" \
    "$root/llama.cpp/build/bin/llama-$tool" \
    "$(command -v "llama-$tool" 2> /dev/null || true)"; do
    if [[ -n $candidate && -x $candidate ]]; then
      echo "$candidate"
      return 0
    fi
  done

  llamacpp_die "llama-$tool not found; run bin/build-llama first"
}

llamacpp_common_run_args() {
  local backend=$1
  local gpu_count=$2

  if [[ $backend != cpu ]]; then
    printf '%s\n' --n-gpu-layers all
    printf '%s\n' --split-mode "${LLAMACPP_SPLIT_MODE:-layer}"
    if [[ -n ${LLAMACPP_TENSOR_SPLIT:-} ]]; then
      printf '%s\n' --tensor-split "$LLAMACPP_TENSOR_SPLIT"
    elif [[ $gpu_count -gt 1 ]]; then
      printf '%s\n' --tensor-split "${LLAMACPP_TENSOR_SPLIT:-$(llamacpp_csv_ones "$gpu_count")}"
    fi
  fi

  printf '%s\n' --ctx-size "${LLAMACPP_CTX_SIZE:-${CTX_SIZE:-4096}}"
  printf '%s\n' --threads "${LLAMACPP_THREADS:-${THREADS:-$(nproc)}}"

  if [[ -n ${MODEL_DRAFT_GGUF:-} ]]; then
    local draft_path
    draft_path=$(llamacpp_resolve_existing_path "$MODEL_DRAFT_GGUF" "draft model")
    printf '%s\n' --spec-draft-model "$draft_path"
    printf '%s\n' --spec-type "${LLAMACPP_SPEC_TYPE:-draft-mtp}"
    printf '%s\n' --spec-draft-n-max "${LLAMACPP_SPEC_DRAFT_N_MAX:-8}"
    if [[ -n ${LLAMACPP_SPEC_DRAFT_N_MIN:-} ]]; then
      printf '%s\n' --spec-draft-n-min "$LLAMACPP_SPEC_DRAFT_N_MIN"
    fi
    if [[ -n ${LLAMACPP_SPEC_DRAFT_P_MIN:-} ]]; then
      printf '%s\n' --spec-draft-p-min "$LLAMACPP_SPEC_DRAFT_P_MIN"
    fi
    if [[ $backend != cpu ]]; then
      printf '%s\n' --spec-draft-ngl "${LLAMACPP_SPEC_DRAFT_N_GPU_LAYERS:-all}"
    fi
  fi

  if [[ -n ${LLAMACPP_EXTRA_ARGS:-} ]]; then
    read -r -a extra_args <<< "$LLAMACPP_EXTRA_ARGS"
    printf '%s\n' "${extra_args[@]}"
  fi
}

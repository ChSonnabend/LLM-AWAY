#!/usr/bin/env bash
# Re-exec bundled commands in the selected development/runtime image.
llamacpp_enter_container() {
  if [[ ${LLAMACPP_IN_CONTAINER:-0} == 1 ]]; then
    # Upstream server images keep their shared libraries beside the binary.
    # Resolve them independently of the project working directory, retaining
    # Apptainer's injected GPU-driver library paths.
    export LD_LIBRARY_PATH="/app${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    return 0
  fi
  [[ -n ${LLAMACPP_CONTAINER:-} && ${LLAMACPP_IN_CONTAINER:-0} != 1 ]] || return 0
  local script=$1; shift
  local image=$LLAMACPP_CONTAINER backend=${LLAMACPP_BACKEND:-} flag name path
  [[ -r $image ]] || { echo "Container not found: $image (see docs/hydra-containers.md)" >&2; exit 2; }
  command -v apptainer >/dev/null || { echo "apptainer is required" >&2; exit 2; }
  case "$backend" in cuda) flag=--nv ;; rocm) flag=--rocm ;; *) echo "Set LLAMACPP_BACKEND=cuda or rocm" >&2; exit 2 ;; esac
  local -a binds=(--bind "$ROOT_DIR") preserved=()
  for path in /lustre /scratch /cvmfs; do
    [[ ! -d $path ]] || binds+=(--bind "$path")
  done
  # Do not inherit host PATH, module functions, or compiler paths.
  while IFS= read -r name; do
    case "$name" in
      LLAMACPP_*|MODEL_*|LLAMA_*|CUDA_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|SLURM_*|JOBS)
        preserved+=("$name=${!name}") ;;
    esac
  done < <(compgen -e)
  local build_id=$backend
  [[ $backend != rocm ]] || build_id="rocm-${LLAMACPP_ROCM_ARCH:-auto}"
  preserved+=("LLAMACPP_BUILD_DIR=${LLAMACPP_BUILD_DIR:-$ROOT_DIR/builds/$build_id-container}")
  exec apptainer exec --cleanenv "$flag" "${binds[@]}" --pwd "$ROOT_DIR" "$image" \
    env "${preserved[@]}" LLAMACPP_IN_CONTAINER=1 /bin/bash --noprofile --norc "$script" "$@"
}

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
  case "$backend" in cuda) flag=--nv ;; rocm) flag=--rocm ;; cpu) flag='' ;; *) echo "Set LLAMACPP_BACKEND=cuda, rocm or cpu" >&2; exit 2 ;; esac
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
  if [[ ${LLAMACPP_CONTAINER_RUNTIME:-apptainer} == docker ]]; then
    local image_id
    image_id=$(python3 "$ROOT_DIR/scripts/remote/ensure-container.py" "$image" "" docker) || exit $?
    local -a docker_args=(run --rm --init --user "$(id -u):$(id -g)" --workdir "$ROOT_DIR" --volume "$ROOT_DIR:$ROOT_DIR")
    if [[ -n ${LLAMACPP_PORT:-} ]]; then
      docker_args+=(--publish "${LLAMACPP_PUBLISH_ADDRESS:+$LLAMACPP_PUBLISH_ADDRESS:}$LLAMACPP_PORT:$LLAMACPP_PORT")
    fi
    if [[ $backend == cuda ]]; then
      [[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || { echo "Docker CUDA requires explicit or scheduler-assigned GPU device IDs" >&2; exit 2; }
      docker_args+=(--gpus "\"device=$CUDA_VISIBLE_DEVICES\"")
      preserved+=(CUDA_VISIBLE_DEVICES= LLAMACPP_VISIBLE_DEVICES=)
    elif [[ $backend == rocm ]]; then
      # ROCm Docker device mapping is not implemented.
      echo "Use Apptainer for ROCm, or Kubernetes with the AMD GPU plugin" >&2
      exit 2
    fi
    exec docker "${docker_args[@]}" --entrypoint /usr/bin/env "$image_id" \
      "${preserved[@]}" LLAMACPP_IN_CONTAINER=1 /bin/bash --noprofile --norc "$script" "$@"
  fi
  command -v apptainer >/dev/null || { echo "apptainer is required" >&2; exit 2; }
  local -a gpu_flags=()
  [[ -z $flag ]] || gpu_flags+=("$flag")
  exec apptainer exec --cleanenv "${gpu_flags[@]}" "${binds[@]}" --pwd "$ROOT_DIR" "$image" \
    env "${preserved[@]}" LLAMACPP_IN_CONTAINER=1 /bin/bash --noprofile --norc "$script" "$@"
}

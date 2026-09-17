# Hydra containers

Do not run `module load ... || true` inside these images. Host Lmod paths are
not available there. The ROCm image contains GCC, Make, CMake and HIP; the
existing cuda_torch_env.sif has CMake/Ninja but lacks GCC and nvcc.

The wrapper uses a clean container environment, binds Lustre/scratch/CVMFS,
preserves Slurm GPU masks, and uses separate `builds/*-container` directories
so a failed host configuration cannot poison the container build.

From the remote project, inside the matching GPU allocation:

```bash
bin/hydra-build rocm
```

For CUDA, first create a separate development image (leaves the Torch image
untouched). Run on a host that supports Apptainer fakeroot/image building:

```bash
apptainer build --fakeroot \
  /lustre/alice/users/csonnab/TPC/TPC_PRODUCTION/Containers/cuda_llamacpp_devel.sif \
  containers/cuda-devel.def
```

If Hydra disallows fakeroot, build it on another compatible Linux host and copy
the SIF to that location. Then, in an H200 allocation:

```bash
bin/hydra-build cuda
```

For direct commands, export LLAMACPP_CONTAINER and LLAMACPP_BACKEND (plus
LLAMACPP_ROCM_ARCH=gfx908 for MI100), then use bin/run-server or bin/run-cli.
The local Hydra profiles pass these settings automatically. Custom
server_command configurations must manage their own container invocation.
No image builds, llama.cpp builds, or inference tests were performed during setup.

Base CUDA image: https://hub.docker.com/r/nvidia/cuda/tags?name=12.8.1-devel-ubuntu22.04

## Prebuilt servers (no compilation)

Upstream images already include `/app/llama-server`. The current upstream ROCm
Dockerfile lists gfx908 (MI100) among its default targets. Runtime compatibility
has not been tested here. Pull on the login host:

```bash
cd /lustre/alice/users/csonnab/TPC/TPC_PRODUCTION/Containers
apptainer pull llama-server-cuda.sif docker://ghcr.io/ggml-org/llama.cpp:server-cuda
apptainer pull llama-server-rocm.sif docker://ghcr.io/ggml-org/llama.cpp:server-rocm
```

Both Hydra profiles now automatically download and reuse these prebuilt images
on agent activation. Manual pulls above are optional. Leave
build_before_run=false. Bundled wrappers automatically find /app/llama-server.
These server images do not include the diagnostic llama-cli command. Pin image
tags/digests once a version works with your model and MTP options.

https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md

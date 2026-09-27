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

## Unsloth training image

Use `remote/containers/unsloth-training-cuda.sif` for fine-tuning and adapter export.
The definition is `remote/containers/unsloth-training-cuda.def`, based on a pinned
linux/amd64 digest of the official Unsloth CUDA image, with pypdf and MarkItDown
added without upgrading the upstream GPU stack. It records resolved packages
in `/opt/llm-away-training/installed-versions.txt` inside the image.

Build on the login host with a local temporary directory:

```bash
APPTAINER_TMPDIR=$(mktemp -d /tmp/llm-away-build.XXXXXX) \
  apptainer build --fakeroot --mksquashfs-args '-processors 4' \
  containers/unsloth-training-cuda.sif containers/unsloth-training-cuda.def
```

Choose Apptainer in the Fine-tuning tab and provide the full remote SIF path.
Leave Training Python blank (the image puts its environment first on PATH),
or specify `/opt/unsloth-venv/bin/python`. GPU-node preflight checks actual
Unsloth imports, CUDA access and the BF16 memory requirement.

`llama-server-cuda.sif` and `llama-server-rocm.sif` are inference images for the
configured NVIDIA and AMD Hydra profiles; they do not supply a training stack.
`cuda-12.8-devel.sif` supplies the documented GLM native build dependencies.
These remain useful even after adding the training image.

The training recipe also installs `causal-conv1d==1.7.0` against the image's
existing PyTorch/CUDA stack (`--no-build-isolation --no-deps`). This supplies
Qwen's optimized causal convolution instead of its slow reference fallback.
Build a **new image filename** while any training job uses the old image; do
not overwrite or delete the active SIF. Select the new image for the next run.
A Git pull updates the recipe, not existing container binaries.

Validate the new image on an idle allocated GPU before using it:

```bash
srun --jobid=YOUR_JOB --overlap --exact --ntasks=1 --cpus-per-task=2 \
  --chdir="$PWD" apptainer exec --nv --bind "$PWD" \
  containers/unsloth-training-cuda-next.sif \
  /opt/unsloth-venv/bin/python training/check_causal_conv1d.py
```

Run these commands from `remote/`. The check compares FP32/BF16 kernel output
with PyTorch's reference convolution and verifies finite backward gradients.


Use `remote/bin/build-container` from any directory for reproducible builds:

```bash
remote/bin/build-container training /shared/path/unsloth-training-cuda-next.sif JOB_ID
remote/bin/build-container cuda /shared/path/llama-server-cuda-next.sif
remote/bin/build-container rocm /shared/path/llama-server-rocm-next.sif
remote/bin/build-container cuda-devel /shared/path/cuda-devel-next.sif
```

The optional job ID GPU-tests training kernels before publishing the image.
The script refuses existing output filenames and saves the embedded definition
beside the new image. It never deletes old images; remove an image only after
checking active allocations and configured host profiles no longer reference it.


### One CUDA image for training and chat serving

`bin/build-container unified containers/llm-away-cuda.sif JOB_ID` combines the
existing training and CUDA inference images, the optimized causal convolution,
and the pinned GLM5-Next server. Build inputs default to
`containers/unsloth-training-cuda.sif`, `containers/llama-server-cuda.sif`, and
`builds/glm5next/build/bin/llama-server`; override them with `TRAINING_IMAGE`,
`INFERENCE_IMAGE`, and `GLM_SERVER`. These inputs are needed only when building;
the final SIF contains the runtimes and the GLM binary. The script GPU-tests both
servers and the convolution kernel before publishing when JOB_ID is supplied.

Choose the same absolute `llm-away-cuda.sif` path in Attach's container field
and Fine-tuning's Apptainer field. Training Python can stay blank. The WebUI and
CLI clients still run on their selected host and connect to llama.cpp's server.
The image is for NVIDIA/CUDA; configured AMD/ROCm hosts need their ROCm runtime.
Do not remove old images while jobs reference them; migrate after validation.

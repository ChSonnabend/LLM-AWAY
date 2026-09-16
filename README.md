# llama.cpp LLM Runner

This directory is a small, model-agnostic wrapper around llama.cpp for running
GGUF models from a terminal or through llama.cpp's local HTTP server. It is meant
to work across ROCm nodes such as MI50 and MI100, CUDA nodes, and CPU-only
fallback environments.

## Layout

```text
bin/                         generic commands
scripts/lib/                 shared detection helpers
models/<model-name>/         lightweight model config and aliases
models/DeepSeek-V2.5-GGUF/   downloaded DeepSeek shards, ignored by git
venvs/hf-download            Hugging Face CLI venv, currently symlinked to .venv39
builds/<backend-arch>/       per-GPU llama.cpp builds
llama.cpp/                   llama.cpp checkout, ignored by git
```

The stable model convention is:

```text
models/<model-name>/model.env
models/<model-name>/model.env
```

For single-file GGUF models, `MODEL_GGUF` can point to any `.gguf` path. For
split GGUF models, `MODEL_GGUF` must point to the real first shard name, such as
`...-00001-of-00004.gguf`. Do not rename a split shard to `model.gguf`; llama.cpp
validates the shard filename pattern.

## Current Model

The default downloaded model is:

- Model name: `qwen3.8-27b-q4km`
- Hugging Face repo: `ggml-org/Qwen3.8-27B-GGUF`
- Main quant: `Qwen3.8-27B-Q4_K_M.gguf`
- MTP draft quant: `mtp-Qwen3.8-27B-Q4_0.gguf`
- Context: `262144` tokens, matching the model training context
- Size on disk: about 20GiB for the main and draft GGUF files

The wrapper defaults to `qwen3.8-27b-q4km` and enables llama.cpp speculative
decoding with `--spec-type draft-mtp`, `--spec-draft-n-max 3`, and the downloaded
MTP draft GGUF. The embedded `llama.cpp` checkout is pinned to upstream
`v0.4.1`, not the earlier Unsloth patch checkout.

## Build llama.cpp

On a GPU node, first make sure the CUDA or ROCm environment is loaded, then run:

```bash
bin/build-llama
```

The build script auto-selects:

- `rocm` when `rocminfo`, `rocm-smi`, or `hipconfig` is present
- `cuda` when NVIDIA tooling is present
- `cpu` otherwise

For MI50, the ROCm architecture is `gfx906`. For MI100, it is `gfx908`. The
script tries to detect this from `rocminfo` or `rocm-smi`; if the cluster hides
that information, set it explicitly:

```bash
LLAMACPP_BACKEND=rocm LLAMACPP_ROCM_ARCH=gfx906 bin/build-llama  # MI50
LLAMACPP_BACKEND=rocm LLAMACPP_ROCM_ARCH=gfx908 bin/build-llama  # MI100
```

For a fresh rebuild of the current stable checkout on both EPN GPU classes:

```bash
cd /scratch/csonnabe/cern-fellowship/misc/lamacpp-llm
LLAMACPP_BACKEND=rocm LLAMACPP_ROCM_ARCH=gfx906 LLAMACPP_CXX_STANDARD=20 bin/build-llama
LLAMACPP_BACKEND=rocm LLAMACPP_ROCM_ARCH=gfx908 LLAMACPP_CXX_STANDARD=20 bin/build-llama
```

Builds go under `builds/`, keyed by backend and architecture, so MI50 and MI100
builds can coexist.

Useful build overrides:

```bash
LLAMACPP_CXX_STANDARD=20 bin/build-llama
LLAMACPP_BUILD_TARGETS="llama-cli llama-bench" bin/build-llama
LLAMACPP_CMAKE_ARGS="-DGGML_NATIVE=OFF" bin/build-llama
```

This checkout may need a newer compiler than GCC 8 for the current llama.cpp
`llama-cli`/`llama-server` targets. If you see template deduction errors in
`tools/server/server-schema.cpp`, load a newer GCC/Clang module on the node and
rerun `bin/build-llama`.

## Show Detected Configuration

Before launching a large model, check what the wrapper sees:

```bash
bin/show-config
```

## List Devices

```bash
bin/list-devices
```

To force a subset of GPUs:

```bash
LLAMACPP_VISIBLE_DEVICES=0,1,2,3 bin/list-devices
```

For ROCm this sets both `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`. For
CUDA it sets `CUDA_VISIBLE_DEVICES`.

On shared ROCm hosts, prefer setting `LLAMACPP_VISIBLE_DEVICES` explicitly. Some
`rocm-smi` versions print more device lines than llama.cpp can actually use, so
the wrapper only auto-generates `--tensor-split` from explicit visible devices or
`LLAMACPP_GPU_COUNT`.

## Terminal Queries

Interactive:

```bash
bin/run-cli
```

One-shot:

```bash
bin/run-cli -p "Explain ROOT RDataFrame in one paragraph."
```

Choose a model by name:

```bash
MODEL_NAME=deepseek-v2.5-q4km bin/run-cli -p "Give me a ROCm debugging checklist."
```

Eight MI50s:

```bash
LLAMACPP_BACKEND=rocm LLAMACPP_ROCM_ARCH=gfx906 \
LLAMACPP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bin/run-cli -p "Hello"
```

Or point directly to any GGUF:

```bash
MODEL=/path/to/model.gguf bin/run-cli -p "Hello"
```

## Local HTTP Server

```bash
bin/run-server
```

Query from another terminal:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v2.5-q4km",
    "messages": [
      {"role": "user", "content": "Give me a concise checklist for debugging ROCm."}
    ],
    "temperature": 0.6,
    "max_tokens": 512
  }'
```

## Download Models

Each managed model gets a `models/<model-name>/model.env` file with at least:

```bash
MODEL_ALIAS=my-model
MODEL_GGUF=models/org-or-user-repo/some-quant/some-model-00001-of-00004.gguf
MODEL_GGUF_SOURCE=models/org-or-user-repo/some-quant/some-model-00001-of-00004.gguf
HF_REPO=org-or-user/repo
HF_INCLUDE=some-quant/*
HF_LOCAL_DIR=models/org-or-user-repo
```

Then download with:

```bash
bin/download-model my-model
```

If `MODEL_GGUF_SOURCE` is set and differs from `MODEL_GGUF`, `bin/download-model`
updates `MODEL_GGUF` as a symlink after the download. For split models, the
target symlink name must keep the `00001-of-000NN.gguf` shard pattern.

The Hugging Face CLI is auto-selected from `venvs/hf-download`, `.venv39`,
`.venv`, or PATH. If none exists, the script creates `venvs/hf-download` with
the newest available Python >= 3.9 and installs `huggingface_hub` plus `hf_xet`.

## Tuning

Defaults are conservative:

- `LLAMACPP_CTX_SIZE=4096`
- `LLAMACPP_SPLIT_MODE=layer`
- all detected GPUs visible
- equal split across visible GPUs

Useful overrides:

```bash
LLAMACPP_CTX_SIZE=2048 bin/run-cli
LLAMACPP_GPU_COUNT=8 bin/run-cli
LLAMACPP_TENSOR_SPLIT=1,1,1,1,1,1,1,1 bin/run-cli
LLAMACPP_SPLIT_MODE=tensor bin/run-cli
```

For DeepSeek-V2.5, keep `LLAMACPP_SPLIT_MODE=layer`. Current llama.cpp tensor
parallel mode does not support the `DeepSeek2` architecture, so layer splitting
is the compatible multi-GPU mode for that model.

The old DeepSeek-specific scripts still exist as compatibility wrappers:

```bash
./download-deepseek-v25-q4km.sh
./build-llama-rocm-gfx906.sh
./run-deepseek-v25-cli.sh
./run-deepseek-v25-server.sh
```

<!-- BEGIN EPN SETUP -->
## EPN Gateway Setup

Provision the EPN gateway helpers once on this Slurm login host. This project's
`scripts/epn/` directory contains `setup.sh`, `llm-epn-serverctl`, and
`llm-epn-slurm-run`, distributed together from the LLM_EPN repository's
`scripts/remote/` directory. There is no client-side init-remote step.

Prerequisites:

- Bash, Python 3.6+, GNU coreutils (including timeout), and sbatch, squeue,
  scancel, sinfo, and srun available in noninteractive login shells.
- Permission to submit jobs to the configured Slurm partition and GPU nodes.
- This runner project, model presets and all GGUF shards, and the EPN state
  directory accessible from both login and compute nodes.
- A compatible llama.cpp build and the site's GPU runtime on compute nodes.
  Follow this README's build/download instructions; run GPU builds inside an
  appropriate allocation. EPN MI50 uses gfx906 and MI100 uses gfx908. Choose a
  model/context that fits the available GPU memory and supports the hardware.
- SSH from the client to this login host, with TCP forwarding to compute nodes.

From the root of this runner project, on the login host:

```bash
bash scripts/epn/setup.sh
bash scripts/epn/setup.sh --check
```

Setup verifies required commands and wrapper files, then installs the two
helpers into ~/.local/bin. It skips identical executable files and backs up
replacements as HELPER.backup.XXXXXXXX/original, preserving permissions and
modification times. Restore with `cp -p -- BACKUP_PATH HELPER_PATH`.
`--check` changes no files and returns nonzero when setup needs attention.
Setup does not download models, build llama.cpp, submit jobs, or change shell
or Codex configuration.

If storing the setup bundle elsewhere, pass `--workdir /path/to/runner`.
Use `--bin-dir /path/to/bin` to change the helper destination; relative bin
paths are home-relative. To update, refresh all three files in scripts/epn/
from the same LLM_EPN revision used by clients, then rerun setup here.

In the client's LLM_EPN config/epn.toml, set:

- [ssh].host: this login host (currently epnh).
- [remote].workdir: the absolute path to this runner project (currently
  /scratch/csonnabe/cern-fellowship/misc/lamacpp-llm).
- [remote].serverctl and runner: $HOME/.local/bin/llm-epn-serverctl and
  $HOME/.local/bin/llm-epn-slurm-run, or your chosen installation paths.
- [remote].state_dir: a shared writable directory; default $HOME/.cache/llm-epn
  is created by the controller when used.
- [slurm] and [llamacpp]: your partition, GPU/node selection, model, build, and
  context settings. The built-in mi50/mi100 node ranges are EPN-specific.

Then on the client, from its LLM_EPN repository:

```bash
./bin/llm-epn list-models
./scripts/init-local.sh
./bin/llm-epn server-status --config config/epn.toml
epn-agent
```

Discovery verifies SSH, wrapper configuration, and readable model files.
Server status verifies the helper/Slurm status path and may create the state
directory. Neither submits a job. epn-agent starts/reuses an allocation and
checks the tunnel and inference; its configured exit behavior cancels the job.
A successful setup check alone does not verify scheduling or GPU compatibility.
<!-- END EPN SETUP -->

## EPN client MTP selection

The updated EPN client shows configured MTP and draft-file availability in model
selection. Select a model interactively with `llm-epn select-model`, or use
`llm-epn select-model --model qwen3.8-27b-q4km --mtp auto|on|off` (choose one mode).
`init-local.sh` also accepts `--mtp`. The default `auto` preserves this remote
preset's MTP setting. `on` requires its configured draft; `off` suppresses the MTP
draft before llama.cpp arguments are assembled. An existing saved off choice is
preserved by the interactive prompt. The local HTTP/Codex API is unchanged.

The wrapper honors LLAMACPP_MTP=auto|on|off through llamacpp_apply_mtp_mode,
called after loading model.env. Both updated EPN helpers pass that setting into
Slurm jobs. Refresh scripts/epn/ from the updated client helper bundle and run
`bash scripts/epn/setup.sh` when upgrading. Other installations need the updated
wrapper library as well; discovery reports mtp.toggle_supported.

Do not append --spec-type none to disable MTP: repeated --spec-type flags
accumulate in the pinned llama.cpp version. Keep manual speculative type/model
arguments separate from the EPN MTP selector. Draft length remains a preset
setting (currently 3). An explicit on choice checks the draft's availability,
not completed builds, GPU compatibility, or speedup.

Finish the llama.cpp rebuild before launching a server. Stop the local provider,
cancel any old allocation with `llm-epn server-cancel --config config/epn.toml`,
then restart. The controller refuses to reuse an active job with a different
saved MTP mode; selection never cancels jobs automatically. Changes to the
remote preset in auto mode also require a new allocation.


# GLM-5.3-Flash on two H200 GPUs

Preset: `glm-5.3-flash-q4`, Unsloth `UD-Q4_K_XL` (~199.7 GB),
750,000-token context, layer split across the allocated GPUs, reasoning enabled.
The preset advertises its context to `run`, including allocations created earlier.
MTP is disabled for this setup. Vision weights are not included.

From the Mac, use the existing two-GPU allocation:

```sh
run --session 2 --model glm-5.3-flash-q4 --mtp off
res-mon --logs 2
```

Use `res-mon --list` to find the session number if it changes. Ctrl+C in the log
viewer only closes the viewer. On exiting the agent, keep the allocation if desired.

Weights live under `remote/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL` on Hydra.
Pinned Hugging Face revision: `621d456e93e926e4b52f85cff5f634358c1828f9`.
The compatible server is `remote/builds/glm5next/build/bin/llama-server`, built
from `unslothai/llama.cpp`, revision `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`
(`glm5next/upstream`, upstream PR 27754). The existing CUDA 12.8 runtime container
is retained. The preset overrides its server binary without replacing `/app`.

Build dependencies are in `remote/containers/cuda-12.8-devel.sif`
(`docker://nvidia/cuda:12.8.1-devel-ubuntu22.04`). Configure with
`GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=90`, `BUILD_SHARED_LIBS=OFF`,
`LLAMA_CURL=OFF`, `LLAMA_OPENSSL=OFF`, `GGML_NATIVE=OFF`, Release mode;
build only `llama-server`. Use an allocated compute node, not a login node.

The server uses FP16 caches and small prompt batches for compatibility and memory
headroom. Loading verifies memory allocation, not speed or accuracy at 750k tokens.
Source: https://unsloth.ai/docs/models/glm-5.3-flash

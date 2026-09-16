# Architecture

Codex should not know whether inference runs on Slurm, Kubernetes, or a long-lived model server. This repo keeps that boundary explicit:

```text
Codex CLI
  -> local HTTP provider on 127.0.0.1
  -> backend adapter / local gateway
  -> ssh epnh
  -> remote serverctl
  -> Slurm server allocation
  -> SSH tunnel
  -> llama.cpp server
```

## Local Provider

`llm-epn serve` exposes:

- `GET /health`
- `POST /v1/chat/completions`
- `POST /v1/responses`

The provider converts Codex/OpenAI-style requests into a plain prompt and sends it to the configured backend.

## Slurm Server Backend

The default backend uses SSH to run `~/.local/bin/llm-epn-serverctl ensure --json` on `epnh`.

The remote host provisions the helpers once using the standalone `scripts/remote/setup.sh` bundle (deployed as `scripts/epn/` in the remote runner project). Client initialization does not install remote files; see the README's Remote Host Setup section for prerequisites and configuration.

The remote controller:

- maps the requested model to a state file under `remote.state_dir`
- submits an `sbatch` job if no active job exists
- writes the allocated hostname once the job starts
- launches `llama-server` or the configured `llamacpp.server_command`

The local backend polls status, opens an SSH tunnel from `gateway.local_port` to the compute node's `gateway.server_port`, waits for readiness, and forwards inference to llama.cpp's HTTP API.

## One-Shot Slurm Backend

The older diagnostic backend uses SSH to run `~/.local/bin/llm-epn-slurm-run --json` on `epnh`.

The remote runner reads a JSON payload and builds a non-interactive `srun` command. The default behavior mirrors the no-argument path of the supplied `node` shell function:

- no class/custom options: let Slurm choose from the partition
- `mi50`: random `epn000` through `epn278`
- `mi100`: random `epn280` through `epn349`, with `EPN_NODE_MI100=1`
- custom options: passed through to `srun`

When `llamacpp.rocm_arch` is `auto`, the remote runner chooses the architecture after Slurm allocation:

- `epn000` through `epn279`: `gfx906` for MI50
- `epn280` through `epn349`: `gfx908` for MI100
- unknown hostnames: first `gfx...` value reported by `rocminfo`, falling back to `gfx908`

## Kubernetes Later

A Kubernetes backend should implement the same local `Backend.infer()` interface as Slurm:

```python
class KubernetesBackend(Backend):
    def infer(self, request: InferenceRequest) -> str:
        ...
```

Good first implementation:

1. Create or reuse a model-serving deployment.
2. Port-forward or route to a service.
3. Submit inference to that service.
4. Keep Codex config pointed at the same local provider.

That keeps Codex switching stable while the remote-side scheduler changes.

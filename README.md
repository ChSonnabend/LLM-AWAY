# LLM EPN

Run Codex against a local HTTP model provider that starts and reuses EPN Slurm-hosted llama.cpp servers through `epnh`.

The framework has three pieces:

1. A local OpenAI-compatible proxy (`llm-epn serve`) that Codex can use as a model provider.
2. A local gateway backend that submits/reuses a Slurm `llama-server` job and opens an SSH tunnel to it.
3. Remote Slurm helpers installed on `epnh`.
4. Backend configuration that keeps Slurm-specific behavior out of the Codex-facing layer, so a Kubernetes backend can be added later without changing Codex setup.

## Quick Start

From this repository:

```bash
./scripts/init-local.sh
```

Run the same command on any Remote-SSH development host where you want the VS Code Codex extension to use EPN. It installs short launcher commands into `~/.local/bin`, writes the EPN Codex provider/profile under `~/.codex`, and creates the EPN model catalog. The setup is offline; it runs from the repository source and does not need PyPI.

Then install the remote runner on the SSH host:

```bash
./scripts/init-remote.sh
```

Start the local provider and warm the remote Slurm server:

```bash
epn-agent
```

Warmup waits for the tunneled llama.cpp server to become reachable, then sends a one-token chat completion so the model is loaded into VRAM before the first Codex turn.

The default config uses the remote wrapper's managed Qwen model:

```toml
[llamacpp]
model_name = "qwen3-coder-next-f16-1m"
server_cli = "bin/run-server"
```

To use another managed model, change `model_name`. To bypass the wrapper, set either:

```toml
[llamacpp]
model_path = "/path/to/model.gguf"
server_cli = "builds/rocm-gfx908/bin/llama-server"
```

or a complete server command:

```toml
[llamacpp]
server_command = ["bin/run-server"]
```

In another terminal, try an inference:

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"epn-llamacpp","messages":[{"role":"user","content":"Hello"}]}' | jq
```

Configure Codex with:

```bash
./bin/llm-epn install-codex-config --config ./config/epn.toml
```

The init script runs this command for you. It updates `$CODEX_HOME/config.toml` or `~/.codex/config.toml`, writes `~/.codex/epn.config.toml`, and creates `~/.codex/model-catalogs/epn.json`.

Use Codex with the provider:

```bash
./bin/codex-epn
codex --no-alt-screen --profile epn
codex exec --profile epn "Summarize this repository"
```

`./bin/codex-epn` starts a fresh local provider by default, even if an older provider is already listening on
port 8765. This makes the launcher own the provider lifecycle: `/exit` and Ctrl+C stop the provider, and the
provider cancels the remote Slurm job on shutdown. Set `LLM_EPN_RESTART_PROVIDER=0` only when you deliberately
want to reuse an existing local provider.

## Important Limits

The proxy implements enough of `/v1/chat/completions` and `/v1/responses` for plain text inference. The default backend now starts a long-lived llama.cpp server under Slurm, forwards prompts to it through an SSH tunnel, and reuses the allocation while it remains active.

The older `slurm` backend uses non-interactive `srun` rather than `srun --pty`, because Codex calls need request/response execution. The default `slurm_server` backend uses `sbatch` to start a persistent server job. By default it lets Slurm choose a valid node from the partition. Set `slurm.node_class` to `mi50`, `mi100`, or a concrete node name when you want a `-w` constraint, or use `slurm.custom_options` for explicit Slurm options.

For ROCm, `llamacpp.rocm_arch = "auto"` maps EPN MI50 nodes (`epn000`-`epn279`) to `gfx906` and MI100 nodes (`epn280`-`epn349`) to `gfx908` inside the Slurm allocation. llama.cpp may still print paths like `ggml-cuda`; in a ROCm build those are HIP/ROCm backend files, not a CUDA requirement.

If Slurm can allocate both MI50 and MI100 nodes, either leave `rocm_arch = "auto"` and set `build_before_run = true`, or pin `slurm.node_class` to one GPU class and keep the matching cached build.

The default `llamacpp.extra_args = ["-n", "64"]` intentionally bounds first-run generation. Increase or remove it once the end-to-end path is stable. `llamacpp.inference_timeout_seconds` is a hard wall-clock guard around the final `run-cli` command, which is executed with stdin closed. If `run-cli` is REPL-only and requires an interactive shell, use a batch CLI entrypoint or switch this backend to a long-lived `llama-server`.

Persistent server commands:

```bash
./bin/llm-epn server-status --config ./config/epn.toml
./bin/llm-epn server-cancel --config ./config/epn.toml
```

With `gateway.cancel_on_exit = true`, a local gateway process cancels the Slurm job when the gateway exits. With `gateway.cancel_reused_on_exit = true`, this also cancels a pre-existing Slurm server that the gateway reused. With `gateway.stream_startup_log = true`, startup logs from a newly submitted remote `llama-server` job are streamed to the gateway terminal while the server is loading.

Traffic to the local model is bounded by `gateway.max_prompt_chars`, `gateway.prompt_keep_head_chars`, and `gateway.prompt_keep_tail_chars`. This keeps Codex's large agent context from overwhelming a local llama.cpp server while preserving the beginning and most recent end of the prompt. `llamacpp.max_tokens` bounds response length.

The proxy includes the Responses `tools` list in the prompt sent to Qwen. Codex owns the actual filesystem/command permissions; the model only chooses among tool names the client exposes. If Qwen emits common file-tool aliases like `read_file`, or an older `exec` shell call, while Codex expects the current `exec_command` shape, the proxy rewrites them to equivalent shell commands with `cmd` arguments. If a client explicitly advertises an `exec` function, that older shape is still preserved.

To use this profile from Codex CLI, VS Code, or the Codex desktop app, select the `epn` profile/model after starting the local gateway. User-level provider config lives in `~/.codex/config.toml`, while profile overrides live in `~/.codex/epn.config.toml`; this matches the current OpenAI Docs profile format. A combined model catalog at `~/.codex/model-catalogs/combined-with-epn.json` makes the EPN model visible to clients that read the global model catalog, but provider routing still requires the `epn` profile or another config that sets `model_provider = "epn"`.

The generated provider display name comes from `[codex].provider_display_name` in `config/epn.toml`, currently `Christian Sonnabend`, so app chrome keeps showing your name instead of `EPN Slurm llama.cpp`. The `[codex].account_email` value documents the intended ChatGPT/Codex account, but app account sign-in is still managed by the ChatGPT app. Set `LLM_EPN_CODEX_PROVIDER_NAME` before running `./scripts/init-local.sh` if you want a temporary label override.

For an EPN-focused VS Code session, make the EPN provider/model active in `~/.codex/config.toml`, then launch:

```bash
./bin/code-epn /path/to/project
```

The launcher starts the local provider if needed, warms the remote Slurm server with a one-token chat completion, and opens VS Code with `--wait`, so closing that VS Code window also tears down the provider it started.

For an already-open local or Remote-SSH VS Code window, run one command in a terminal on the same host:

```bash
epn-agent
```

Then reload the Codex sidebar if it was already open.

If you only want the HTTP gateway without immediately starting the remote Slurm job, run:

```bash
llm-epn serve --config /path/to/LLM_EPN/config/epn.toml
```

## Files

- `config/epn.toml`: default local and remote settings.
- `bin/llm-epn`: executable wrapper.
- `bin/epn-agent`: foreground provider command for already-open VS Code windows.
- `bin/codex-epn`: Codex launcher that owns the local provider lifecycle.
- `bin/code-epn`: VS Code launcher that starts the local provider before opening the editor.
- `src/llm_epn/`: stdlib Python proxy and backend code.
- `scripts/init-local.sh`: installs launcher links and writes Codex setup.
- `scripts/init-remote.sh`: copies the remote Slurm runner to `epnh`.
- `scripts/remote/llm-epn-serverctl`: remote persistent server controller invoked over SSH.
- `scripts/remote/llm-epn-slurm-run`: older one-shot diagnostic runner invoked over SSH.
- `docs/architecture.md`: backend design and Kubernetes extension plan.
- `docs/codex.md`: Codex configuration notes.

## Development

Run local tests:

```bash
./scripts/test.sh
```

Format is intentionally plain Python plus shell so the repo remains easy to push and bootstrap on constrained systems.

# LLM EPN

Run Codex against a local HTTP model provider that starts and reuses EPN Slurm-hosted llama.cpp servers through `epnh`.

The framework has four pieces:

1. A local OpenAI-compatible proxy (`llm-epn serve`) that Codex can use as a model provider.
2. A local gateway backend that submits/reuses a Slurm `llama-server` job and opens an SSH tunnel to it.
3. Remote Slurm helpers installed on `epnh`.
4. Backend configuration that keeps Slurm-specific behavior out of the Codex-facing layer, so a Kubernetes backend can be added later without changing Codex setup.

## Quick Start

Complete [Remote Host Setup](#remote-host-setup) once on the Slurm login host. An already configured remote needs no initialization from each client. Then, from this repository on the client:

```bash
./scripts/init-local.sh
```

Setup first queries the configured SSH host for installed managed models and presents a numbered selection menu, including MTP status and additional draft-file size. Press Enter to keep the current model if it is available. For a configured MTP model with readable draft files, setup also asks whether to use MTP. The choice updates `[model].name`, `[llamacpp].model_name`, and `[llamacpp].mtp` in the project config, with a timestamped backup. Global Codex settings are preserved.

Run the same command on any Remote-SSH development host where you want to use EPN. It installs short launcher commands into `~/.local/bin`, writes the EPN Codex provider/profile under `~/.codex`, and creates the EPN model catalog. Setup runs from the repository source and does not need PyPI. Discovery uses SSH and the remote model library; it needs no installed EPN remote helper and submits no Slurm jobs.

```bash
./bin/llm-epn list-models                         # Query installed presets
./bin/llm-epn select-model                        # Choose again later
./scripts/init-local.sh --ssh-alias MY_EPN_ALIAS --model PRESET_NAME  # Noninteractive selection
./scripts/init-local.sh --ssh-alias MY_EPN_ALIAS --model PRESET_NAME --mtp on  # Explicit MTP choice
./scripts/init-local.sh --skip-model-selection    # Keep configured model; still checks SSH/shared installation
```

Discovery resolves `models/*/model.env` using the remote wrapper library and lists only presets whose main GGUF files (including all split shards) are readable and nonempty. MTP draft files are checked separately; a missing/incomplete draft is shown in the list. This checks file availability, not GPU compatibility or completion of a llama.cpp build. SSH errors, an unavailable requested model, or cancellation stop selection before settings are written. Noninteractive setup requires `--model NAME` or `--skip-model-selection`. Use `--config PATH` for a different project config (`LLM_EPN_CONFIG` is also supported by the init script and launchers).

Start the local provider and warm the remote Slurm server:

```bash
epn-agent
```

Warmup waits for the tunneled llama.cpp server to become reachable, then sends a one-token chat completion so the model is loaded into VRAM before the first Codex turn.

The default config uses the remote wrapper's managed Qwen 3.8 model:

```toml
[llamacpp]
model_name = "qwen3.8-27b-q4km"
mtp = "auto"
server_cli = "bin/run-server"
```

To use another managed model, run `./bin/llm-epn select-model`. This keeps the backend preset and served model alias in sync and refreshes the EPN Codex profile/catalog. Restart the EPN provider and start a new EPN Codex session after switching. Selection preserves your context size, GPU settings, and extra server arguments; check that they suit the chosen model. Managed selection requires empty `model_path` and `server_command`. For custom setups, use `--skip-model-selection` during init and keep `mtp = "auto"`. To bypass the wrapper, set either:

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

The init script runs this command for you. It updates the EPN provider block in `$CODEX_HOME/config.toml` or `~/.codex/config.toml`, writes `epn.config.toml`, and creates `model-catalogs/epn.json` under the same Codex home. Initialization preserves your global model, provider, reasoning effort, and catalog settings, including when rerun. EPN is selected through its profile. Only an explicit `install-codex-config --activate` replaces those global settings; `--no-activate` remains supported and is the default.

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

## EPN preset and the model dropdown

`init-local.sh` installs one stable model entry named **EPN** (`epn`) in the
`epn` profile. It uses the gateway's selected remote model; changing the managed
preset does not change the client model ID. Start the gateway with `epn-agent`
and use `codex --profile epn` (or `codex-epn`). Restart the gateway after changing
its selected model. `/v1/models` advertises the same `epn` alias.

OpenAI remains available through your unchanged default configuration and normal
Codex sessions. The installer does not replace the global catalog or provider.
The combined catalog file is an export artifact, not a provider-routing table.

**Native dropdown limitation:** in the installed Codex/VS Code clients, picking
a model does not select a provider for that individual catalog entry. Adding EPN
to the OpenAI catalog would send it to OpenAI; selecting EPN globally would send
OpenAI model choices to the EPN gateway. Therefore setup does not claim to add a
working EPN entry beside OpenAI models in one shared native dropdown. The EPN
catalog applies to sessions configured with the EPN provider/profile. See the
[official profile configuration](https://learn.chatgpt.com/docs/config-file/config-advanced#profiles).

## Troubleshooting Qwen requests

Startup displays the current Slurm job state and waiting reason (for example,
`PENDING (Resources)`). Each new allocation uses a separate `MODEL-JOBID.log`
file; pending jobs do not replay old startup logs. If an error mentions a
different job ID or an old cancellation time, it is historical output from
the previous shared-log behavior. `server-status` reports the current job and
its log path.

If llama.cpp reports `System message must be at the beginning`, update and
restart the **local gateway**. The bridge now merges system/developer instructions
into one leading system message, preserving the order of conversation turns.
This applies to both Responses and Chat Completions. Upstream HTTP error details
are included in the gateway error instead of only reporting HTTP 500. Increasing
the timeout does not fix this chat-template rejection.

## Multi-token prediction (MTP)

MTP is implemented by the remote llama.cpp server; the local HTTP/Codex protocol is unchanged. The current remote `qwen3.8-27b-q4km` preset configures `MODEL_DRAFT_GGUF`, `LLAMACPP_SPEC_TYPE=draft-mtp`, and `LLAMACPP_SPEC_DRAFT_N_MAX=3`. Discovery reports this configured mode and the draft's disk size, not measured acceleration or VRAM usage. See [llama.cpp's speculative decoding documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) for the underlying feature.

`[llamacpp].mtp` supports three values:

- `"auto"` (default): follow the remote preset, preserving existing remote MTP setup.
- `"on"`: require that preset's configured MTP draft and enable it.
- `"off"`: suppress the preset's MTP draft before the remote wrapper generates llama.cpp arguments, so it is not loaded.

Interactive selection asks `Use MTP? [Y/n]` when the remote preset enables MTP; a saved `off` choice changes the default to `[y/N]`. Enter preserves the effective setting. Noninteractive selection never prompts for MTP: it keeps the saved mode for the same model, or uses `auto` when changing models. Pass an explicit choice for scripts:

```bash
./bin/llm-epn select-model --model qwen3.8-27b-q4km --mtp on
./bin/llm-epn select-model --model qwen3.8-27b-q4km --mtp off
./bin/llm-epn select-model --model qwen3.8-27b-q4km --mtp auto
```

An explicit mode requires updated EPN helpers and a remote wrapper with `llamacpp_apply_mtp_mode`, called by `llamacpp_load_model_config` after loading `model.env`. That function honors `LLAMACPP_MTP=auto|on|off`; the gateway passes it into both persistent-server jobs and the one-shot runner. The current remote wrapper has this support. On older remote projects, update the wrapper and refresh the helper bundle before using `on`/`off`; `auto` retains the existing behavior. Discovery JSON includes `mtp.toggle_supported`. These checks recognize externally configured MTP draft files; they do not infer embedded MTP capability from arbitrary GGUF tensors.

Avoid mixing the MTP selector with manual speculative type/draft-model arguments. In the pinned llama.cpp revision, repeated `--spec-type` flags accumulate, so appending `--spec-type none` is not a reliable off switch. For custom speculative setups, use `mtp = "auto"` and manage their arguments yourself. Draft length and other tuning stay in the remote preset.

Finish rebuilding llama.cpp before starting an MTP-enabled server. If an allocation is still running, stop the provider and run `./bin/llm-epn server-cancel --config config/epn.toml`, then restart it. The controller refuses to reuse an active job with a different saved MTP mode; changing the setting does not cancel jobs automatically. Changing the remote preset while using `auto` also requires a fresh allocation.

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

The default `llamacpp.context_size = 262000` is passed to the remote server as a final command-line override and advertised in the Codex model catalog. `llamacpp.max_tokens = 262000` sets the generation cap; input and output share the server's context window. `gateway.max_prompt_chars = 0` disables character truncation. Set a positive value to re-enable head/tail truncation with `gateway.prompt_keep_head_chars` and `gateway.prompt_keep_tail_chars`. Restart the EPN server allocation and local gateway, and start a fresh Codex session after changing these limits.

Responses requests preserve system, user, and assistant roles when sent to llama.cpp, keeping the latest user question separate from Codex's environment and permission instructions. Tool calls and results retain the bridge's text markup. The legacy one-shot backend still uses a flattened prompt.

The proxy includes the Responses `tools` list in the prompt sent to Qwen. Codex owns the actual filesystem/command permissions; the model only chooses among tool names the client exposes. If Qwen emits common file-tool aliases like `read_file`, or an older `exec` shell call, while Codex expects the current `exec_command` shape, the proxy rewrites them to equivalent shell commands with `cmd` arguments. If a client explicitly advertises an `exec` function, that older shape is still preserved.

The generated Codex model instructions and the gateway prompt both treat inspect/explain/diagnose/review requests as read-only tasks. The model is told to answer after inspection, not modify files, unless the user explicitly asks for a change. When an edit is requested, it is told to use `apply_patch` instead of shell heredocs or redirection.

To use EPN, select the `epn` profile after starting the local gateway. The installer writes the provider to `~/.codex/config.toml` and profile overrides to `~/.codex/epn.config.toml`. It also writes `~/.codex/model-catalogs/combined-with-epn.json` as an export preserving existing non-EPN entries; this file is not selected globally and does not fetch the normal Codex catalog. Explicit `--activate` selects the EPN-only catalog and provider.

The generated provider display name comes from `[codex].provider_display_name` in `config/epn.toml`, currently `Christian Sonnabend`, so app chrome keeps showing your name instead of `EPN Slurm llama.cpp`. The `[codex].account_email` value documents the intended ChatGPT/Codex account, but app account sign-in is still managed by the ChatGPT app. Set `LLM_EPN_CODEX_PROVIDER_NAME` before running `./scripts/init-local.sh` if you want a temporary label override.

For an EPN-focused VS Code session, select the EPN profile in the client, then launch:

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

## Remote Host Setup

Remote setup belongs to the Slurm login host. The old client-side `scripts/init-remote.sh` has been removed. The standalone bundle in `scripts/remote/` supplies the remote initializer and both gateway helpers; keep these three files together when copying them into the remote runner project.

### Prerequisites

- SSH from the client to the login host, including any required jump host. `[ssh].host` defaults to `epnh`. The login host must allow TCP forwarding to allocated compute nodes.
- Bash, Python 3.6+, GNU coreutils (including `timeout`), and Slurm commands `sbatch`, `squeue`, `scancel`, `sinfo`, and `srun` available to noninteractive login shells. The account needs access to the configured partition and GPU nodes.
- A remote llama.cpp wrapper project containing `bin/run-server`, `bin/run-cli`, and `scripts/lib/llamacpp-env.sh`. This is a separate project; the EPN setup bundle does not download it. On the current host it is `/scratch/csonnabe/cern-fellowship/misc/lamacpp-llm`.
- Readable model presets in `models/PRESET/model.env` and their downloaded GGUF files, including all shards. Use the runner project's `bin/download-model PRESET` to download a configured preset.
- A compatible llama.cpp build and the site's ROCm/CUDA environment on the compute nodes. Follow the remote project's README and run `bin/build-llama` inside an appropriate GPU allocation if a build is missing. EPN MI50 requires `gfx906`, MI100 `gfx908`; the selected model must support the GPU and fit its available memory/context.
- The runner project, models, and `remote.state_dir` must be accessible from both the login host and compute nodes. The default state location is `$HOME/.cache/llm-epn`; the controller creates it when used.

### Install once on the remote host

Place the contents of this repository's `scripts/remote/` directory in the remote runner project's `scripts/epn/` directory. For example, from the client repository (adjust the host and path for your account):

```bash
ssh epnh 'mkdir -p /scratch/csonnabe/cern-fellowship/misc/lamacpp-llm/scripts/epn'
scp scripts/remote/setup.sh scripts/remote/llm-epn-serverctl scripts/remote/llm-epn-slurm-run \
  epnh:/scratch/csonnabe/cern-fellowship/misc/lamacpp-llm/scripts/epn/
```

Then **on the remote login host**:

```bash
cd /scratch/csonnabe/cern-fellowship/misc/lamacpp-llm
bash scripts/epn/setup.sh
bash scripts/epn/setup.sh --check
```

Setup checks the required commands and wrapper files, then installs the two helpers into `~/.local/bin`. It does not download models, build llama.cpp, submit jobs, or change shell/Codex configuration. Repeated setup leaves identical executable helpers untouched. Replacements get unique adjacent `HELPER.backup.XXXXXXXX/original` backups with their original permissions; restore one on the remote with `cp -p -- BACKUP_PATH HELPER_PATH`. `--check` makes no changes and returns a nonzero status if prerequisites or installed helpers need attention.

When the bundle is stored elsewhere, pass `--workdir /path/to/runner`. Use `--bin-dir /path/to/bin` to override the installation directory (`LLM_EPN_REMOTE_BIN` is also supported; relative bin paths are home-relative). After a helper update, refresh the bundle from the same EPN revision used by the client and rerun setup **on the remote host**. Normal client initialization does not reinstall remote files.

### Match the client configuration and verify

Set `[ssh].host` and `[remote].workdir` in `config/epn.toml` to your login host and runner project. Defaults for `[remote].serverctl` and `[remote].runner` are `$HOME/.local/bin/llm-epn-serverctl` and `$HOME/.local/bin/llm-epn-slurm-run`; update them if you chose another bin directory. Configure `[slurm].partition`, `node_class`/`custom_options`, and `[llamacpp]` for your cluster, model, GPU build, and context size. The built-in `mi50`/`mi100` node ranges are EPN-specific.

From the client repository:

```bash
./bin/llm-epn list-models                         # Verify SSH, wrapper, and model files
./scripts/init-local.sh                         # Select a model and configure this client
./bin/llm-epn server-status --config config/epn.toml  # Verify helper and Slurm status path
epn-agent                                      # Starts/reuses a job and verifies inference
```

Model discovery and server status do not submit a job; status may create the state directory. A successful setup check alone does not prove GPU compatibility or scheduling permission. `epn-agent` performs the actual allocation, tunnel, and inference check; its configured exit behavior cancels the job.

## Files

- `config/epn.toml`: default local and remote settings.
- `bin/llm-epn`: executable wrapper.
- `bin/epn-agent`: foreground provider command for already-open VS Code windows.
- `bin/codex-epn`: Codex launcher that owns the local provider lifecycle.
- `bin/code-epn`: VS Code launcher that starts the local provider before opening the editor.
- `src/llm_epn/`: stdlib Python proxy and backend code.
- `scripts/init-local.sh`: installs launcher links and writes Codex setup.
- `scripts/remote/setup.sh`: initializer run on the remote login host, deployed as `scripts/epn/setup.sh` in the runner project.
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

With the persistent Slurm backend, `[slurm] mi50_fallback = true` and `node_class = "mi100"` prefer an idle MI100, then an idle MI50, and otherwise queue on a valid MI100. This requires `[llamacpp] rocm_arch = "auto"` and a working remote gfx906 build for your model. Disable the fallback for models requiring MI100. Existing allocations are reused; the fallback applies to new jobs.

### Linux and macOS local setup

The local gateway requires Python 3.9 or newer, Bash, and OpenSSH on Linux or macOS. No local GPU, ROCm, or Slurm installation is needed. Configure `ssh epnh` (including any jump host and credentials), then run `bash scripts/init-local.sh`. Add `export PATH="$HOME/.local/bin:$PATH"` to your shell startup file (`~/.zshrc` for the usual macOS shell, or `~/.bashrc` for Bash). Install the `code` command in PATH if using `code-epn`, and the Codex CLI if using `codex-epn`. The launchers resolve symlinks with Python and do not require GNU `readlink` or Linux `/proc`. Native macOS execution has not been tested here.

### Interrupting the provider

Press Ctrl+C once to stop `epn-agent`. Further Ctrl+C and SIGTERM signals are ignored while cleanup completes; there is no fixed five-second cutoff that could interrupt remote cancellation. The controller SSH subprocess runs in a separate session, so repeated terminal interrupts do not kill it. Shutdown waits for an in-flight submission to return its job ID before cancellation. Provider-restart scripts also wait for cleanup instead of forcing a kill after five seconds. A forced SIGKILL, power loss, or unavailable SSH connection can still prevent cancellation; use `llm-epn server-status` and `llm-epn server-cancel` afterward if necessary.

### Multiple users: one shared model installation

Each user first creates their own working `Host` alias in `~/.ssh/config`, using their own EPN username, SSH key, and jump-host settings. The alias can be named anything, for example `my-epn`. Run `./scripts/init-local.sh`: it asks for that alias **before** connecting or selecting a model, checks access, and saves it locally. SSH uses the alias's `User` setting; setup clears any previous explicit username in the framework config.

For noninteractive setup, use:

```bash
./scripts/init-local.sh --ssh-alias my-epn --model qwen3.8-27b-q4km --mtp on
```

All users run against `/scratch/csonnabe/cern-fellowship/misc/lamacpp-llm`, including its existing `models/`, GPU builds, and `scripts/epn/` helpers. They do **not** download models, copy the remote project, rebuild llama.cpp, or install remote helpers separately. Setup disables per-run builds. The shared installation's owner maintains models, builds, and helper scripts. An alternative shared installation can be selected with `--remote-workdir /absolute/path`.

Each distinct remote Unix account gets its own Slurm allocations and `$HOME/.cache/llm-epn` state/logs. Users sharing the same remote account also share that state. Other users need read/traverse access to the shared model directories and read/execute access to the runtime; setup reports access failures before saving. Model selection lists the presets already installed there. No permissions are broadened by local setup.

## Running multiple agents, extra allocations, and multi-node

### Multiple agents on the same allocation

Several local agents can share one remote `llama-server` allocation. Start a second local provider on a different local port:

```bash
EPN_PORT=8766 epn-agent          # second local provider on 127.0.0.1:8766
```

Point that agent's Codex session at `http://127.0.0.1:8766/v1`. Both agents then share the same GPU allocation, so their requests queue on the model rather than running at independent full speed.

### A second Slurm allocation for the same model

To give a second agent its own allocation (true parallelism, at the cost of another node), point it at a second remote server port. When the running job does not already serve that port, a fresh allocation is submitted on it:

```bash
EPN_PORT=8766 EPN_SLURM_JOB=8081 epn-agent   # or: llm-epn serve --port 8766 --slurm-job 8081
```

`--slurm-job` is the *remote* port of the `llama-server`; the Slurm job ID is derived and managed internally. Omit it to reuse the default allocation.

### Multi-node allocations (`--nodes`)

`--nodes N` (or `EPN_NODES=N`, or `[slurm].nodes = N`) requests N full nodes in one allocation:

```bash
EPN_NODES=2 epn-agent            # or: llm-epn serve --nodes 2
```

This is the hook for scaling a single model across more GPUs/VRAM. It only actually increases VRAM for one model when the **remote wrapper** spans the model across nodes (for example llama.cpp built with the RPC/multi-node backend) **and** the Slurm cluster is IB-aware (`SlurmdParameters` with the IB network plugin). With the current single-node wrapper, N > 1 yields N scheduled nodes but the server still uses one of them. InfiniBand is used by the remote inference runtime, not by the EPN launcher.

### EPN agent efficiency and permissions

The `[codex]` section in `config/epn.toml` controls the generated EPN profile and
initial model prompt. After changing it, run `./bin/llm-epn install-codex-config
--no-activate` and start a fresh `codex --profile epn` session. Restart the local
`epn-agent` gateway after gateway/source or `[llamacpp]` changes.

The checked-in EPN profile selects `danger-full-access` with approval policy
`never`: commands can access local files and the network, including SSH, without
Codex approval prompts. This applies to the EPN profile; global Codex permissions
are preserved. To restore sandboxed operation, set `sandbox_mode =
"workspace-write"` and `approval_policy = "on-request"`, then reinstall the profile.
OS permissions and remote account permissions still apply.

The working context budget is 65,536 tokens with compaction at 45,000 tokens,
leaving room for the configured 16,384-token generation ceiling. The remote
server's 262,000-token context capacity is unchanged. `llamacpp.max_tokens` is the
completion request limit sent to the remote llama.cpp server; it does not require
editing remote scripts. A server-side cap can still impose a smaller limit.
`model_reasoning_effort` is a Codex profile setting, not a Qwen thinking switch.

Individual tool results kept in model history are limited to 2,000 tokens.
Reasoning display is hidden and the prompt requests concise command output.
Codex's interactive “Ran/Explored” activity remains visible; these settings do not
provide a switch to hide every tool activity row. Errors are retained for diagnosis.
The bridge emits real custom-tool events for `apply_patch`, preserves tool history
without nested JSON escaping, and reports malformed tool calls as failures instead
of printing executable-looking markup as a final answer.

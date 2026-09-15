# Codex Integration

Install launcher links, Codex config, and model catalog with:

```bash
./scripts/init-local.sh
```

Run the same command on any Remote-SSH development host where the VS Code Codex extension should use EPN. Codex reads config on the host where its extension/backend process runs, so each remote host needs its own `~/.codex` setup.

The generated profile uses:

```toml
[model_providers.epn]
name = "Christian Sonnabend"
base_url = "http://127.0.0.1:8765/v1"
wire_api = "responses"

model_provider = "epn"
model = "qwen3-coder-next-f16-1m"
model_reasoning_effort = "high"
model_catalog_json = "/home/chris/.codex/model-catalogs/epn.json"
```

For global model pickers, `~/.codex/config.toml` can point `model_catalog_json` at
`~/.codex/model-catalogs/combined-with-epn.json`, which contains the normal Codex model catalog plus
`qwen3-coder-next-f16-1m`.

Run:

```bash
epn-agent
./bin/codex-epn
./bin/code-epn /path/to/project
codex --no-alt-screen --profile epn
codex exec --profile epn "Explain this repository"
```

Notes:

- `epn-agent` is the one-command startup path for already-open local or Remote-SSH VS Code windows. It starts the local provider and warms the remote Slurm server immediately with a one-token chat completion, which forces llama.cpp to load the model into VRAM before the first Codex turn.
- The generated provider display name comes from `[codex].provider_display_name` in `config/epn.toml`, currently `Christian Sonnabend`. `[codex].account_email` documents the intended ChatGPT/Codex account, currently `sonnabendch@gmail.com`, but actual account sign-in is managed by the ChatGPT app. Set `LLM_EPN_CODEX_PROVIDER_NAME` before running `./scripts/init-local.sh` if you want a temporary label override.
- `./bin/codex-epn` starts the local provider for the Codex session and stops it when Codex exits. With `gateway.cancel_on_exit = true`, that also cancels the Slurm job used by that provider.
- `./bin/code-epn` starts the local provider when needed, warms the remote Slurm server, and opens VS Code with `--wait`, so the provider it started is stopped when that VS Code window closes.
- The launcher starts a fresh local provider by default so `/exit` and Ctrl+C can cleanly stop the provider and cancel the remote Slurm job. Set `LLM_EPN_RESTART_PROVIDER=0` to reuse an existing local provider.
- Direct `codex --profile epn` still works, but only if `llm-epn serve` is already running.
- Keep the `epn` profile selected when using the EPN model. The catalog controls visibility; the profile controls provider routing to `http://127.0.0.1:8765/v1`.
- The proxy implements plain text Responses streaming events and translates Qwen's `<tool_call>` markup into Responses `function_call` items so Codex can execute tools instead of displaying the markup as text. It handles both inline calls like `<tool_call> function=exec ...` and nested calls like `<tool_call><function=read_file>...`.
- When a model answer contains tool calls, the proxy preserves any visible text around those calls. If the answer contains only tool calls, it emits a short progress message such as `Calling exec (2 calls).` before the hidden `function_call` items.
- Codex tool access is granted by the Codex client, not by the llama.cpp model. The proxy now includes the Responses `tools` list in Qwen's prompt so it sees the real tool names. If Qwen still emits common file-tool aliases such as `read_file`, `list_files`, or `search_files` when only `exec` is available, the proxy rewrites those calls to `exec` shell commands before returning them to Codex.
- Official OpenAI documentation describes `codex exec` for automation and `config.toml` model providers/profiles; this repo uses those surfaces rather than wrapping Codex internals.

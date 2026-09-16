# Codex Integration

Install launcher links, Codex config, and model catalog with:

```bash
./scripts/init-local.sh
```

Setup queries the remote managed model presets and prompts for an installed model before installing the AWAY profile. Use `--model NAME` for noninteractive selection or `--skip-model-selection` for offline setup. To change the model later, run `./bin/llm-away select-model`; it updates the project model settings and AWAY profile/catalog, then asks you to restart the AWAY provider and start a new AWAY Codex session. `./bin/llm-away list-models --json` returns the discovered preset names, served aliases, paths, and file sizes without changing settings.

Run the same command on any Remote-SSH development host where the VS Code Codex extension should use AWAY. Codex reads config on the host where its extension/backend process runs, so each remote host needs its own `~/.codex` setup.

Initialization installs the AWAY provider/profile while preserving global settings, including the selected model/provider and any catalog override. Repeated runs preserve those settings too. Use the AWAY profile to select AWAY for a session. To deliberately replace the global defaults, run `./bin/llm-away install-codex-config --activate`. The compatibility flag `--no-activate` explicitly requests the default behavior.

The generated provider in `config.toml` uses:

```toml
[model_providers.away]
name = "Christian Sonnabend"
base_url = "http://127.0.0.1:8765/v1"
wire_api = "responses"
```

The generated `away.config.toml` profile uses:

```toml
model_provider = "away"
model = "away"
model_reasoning_effort = "high"
model_catalog_json = "/home/chris/.codex/model-catalogs/away.json"
```

The installer also writes `~/.codex/model-catalogs/combined-with-away.json`, preserving
existing non-AWAY entries and updating the stable AWAY entry. This is an export
artifact, not a routing table. It does not fetch or replace the normal Codex
catalog. Explicit `--activate` selects the AWAY-only catalog and provider globally.

The AWAY catalog advertises the configured `llamacpp.context_size`. Input and
output share the server context window. Character truncation is disabled by default.
After changing these settings, reinstall the catalog with
`./bin/llm-away install-codex-config`, restart the gateway and remote server allocation,
and start a fresh Codex session. Codex sizes the skills-list budget from its model
catalog, so changing only the remote model preset does not update that budget.

Run:

```bash
away-agent
./bin/codex-away
./bin/code-away /path/to/project
codex --no-alt-screen --profile away
codex exec --profile away "Explain this repository"
```

Notes:

- `away-agent` is the one-command startup path for already-open local or Remote-SSH VS Code windows. It starts the local provider and warms the remote Slurm server immediately with a one-token chat completion, which forces llama.cpp to load the model into VRAM before the first Codex turn.
- The generated provider display name comes from `[codex].provider_display_name` in `config/away.toml`, currently `Christian Sonnabend`. `[codex].account_email` documents the intended ChatGPT/Codex account, currently `sonnabendch@gmail.com`, but actual account sign-in is managed by the ChatGPT app. Set `LLM_REMOTE_CODEX_PROVIDER_NAME` before running `./scripts/init-local.sh` if you want a temporary label override.
- `./bin/codex-away` starts the local provider for the Codex session and stops it when Codex exits. With `gateway.cancel_on_exit = true`, that also cancels the Slurm job used by that provider.
- `./bin/code-away` starts the local provider when needed, warms the remote Slurm server, and opens VS Code with `--wait`, so the provider it started is stopped when that VS Code window closes.
- The launcher starts a fresh local provider by default so `/exit` and Ctrl+C can cleanly stop the provider and cancel the remote Slurm job. Set `LLM_REMOTE_RESTART_PROVIDER=0` to reuse an existing local provider.
- Direct `codex --profile away` still works, but only if `llm-away serve` is already running.
- Keep the `away` profile selected when using the AWAY model. The catalog controls visibility; the profile controls provider routing to `http://127.0.0.1:8765/v1`.
- The proxy implements plain text Responses streaming events and translates Qwen's `<tool_call>` markup into Responses `function_call` items so Codex can execute tools instead of displaying the markup as text. It handles both inline calls like `<tool_call> function=exec ...` and nested calls like `<tool_call><function=read_file>...`.
- When a model answer contains tool calls, the proxy preserves any visible text around those calls. If the answer contains only tool calls, it emits a short progress message such as `Calling exec (2 calls).` before the hidden `function_call` items.
- Codex tool access is granted by the Codex client, not by the llama.cpp model. The proxy now includes the Responses `tools` list in Qwen's prompt so it sees the real tool names. If Qwen still emits common file-tool aliases such as `read_file`, `list_files`, or `search_files` when only `exec` is available, the proxy rewrites those calls to `exec` shell commands before returning them to Codex.
- Official OpenAI documentation describes `codex exec` for automation and `config.toml` model providers/profiles; this repo uses those surfaces rather than wrapping Codex internals.

## Native model picker scope

The generated entry has slug `away` and display name `AWAY`, independent of the
remote preset. It routes through the existing AWAY profile and follows the
gateway's configured model. The default OpenAI provider and catalog remain
unchanged. Run `codex --profile away` for AWAY and normal `codex` for your default.

The installed app/VS Code model picker does not attach a different provider to
each model entry. A merged catalog alone cannot safely switch between OpenAI
and AWAY. Setup therefore does not install a global merged picker or promise
simultaneous native dropdown access. The AWAY entry is scoped to AWAY-configured
sessions; GUI availability depends on that client's provider/profile support.

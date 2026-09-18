## Reuse a loaded session

`run --session 3` now offers **Keep loaded model / Load a different model**.
Keeping it offers **Reopen agent / Keep running in background**; `--serve`
preselects background mode and returns to the shell once the model is ready.
Reusing skips model discovery and MTP selection and does not reload the weights.
Reopening uses Codex's saved-conversation picker (choose the previous conversation);
it restores conversation history, not the original terminal screen/process.
Future launches remember their working directory for that picker. Older launches
have no saved directory: invoke `run` from the original project folder.
Closing the terminal preserves the model; exiting the agent offers keep-model,
unload-model, or release-allocation choices. A currently attached client still
holds an exclusive lease: close/detach it before reconnecting from another window.

# Local client

Run from this directory; see the [project guide](../README.md) for architecture and configuration.

```sh
./scripts/install-resource-tools.sh    # Once: install command links
res-alloc                            # Choose host and reserve resources
res-alloc --restart                  # Reconfigure a host
res-mon --list
run --session 2                      # Choose model and start the agent
run --session 2 --model glm-5.3-flash-q4 --mtp on
res-background --session 2           # Load/keep the model, then return to the shell
run --session 2 --rag .              # Optional local project retrieval
res-mon --logs 2                     # Ctrl+C only closes the log viewer
res-mon --kill 2 --release
```

Use the session ID returned by `res-alloc`. Exiting the agent lets you keep the
allocation (unloading the model) or release it. Independent sessions can use
different hosts/models. No separate init script is needed.

`res-background --session 2` is the non-interactive equivalent of choosing
**Keep running in background**: it loads or reuses the model, keeps the
allocation alive, and returns to the shell. Reattach later with
`run --session 2` and choose **Reopen agent**.

`res-mon` opens a full-screen monitor in the same terminal. Arrow keys select an
allocation; **1/F1** exits, **2/F2** releases it (confirm with Enter; stops an attached
agent first), and **3/F3** opens its live allocation/provider logs. Esc returns from
logs without stopping anything; End resumes log following. PgUp/PgDn scroll logs
or the selected allocation's details, including scheduler options. The display
resizes with the terminal and reads local daemon snapshots rather than polling SSH.
`res-mon --list`, `--logs ID`, and `--kill ID --release` remain available.

One allocation ID supports one model/agent at a time. A second `run` on that ID is
rejected even if GPU memory is available. Create another allocation for concurrent
agents; sharing one allocation between multiple models is not implemented.

Requires Python 3.10+, Bash, OpenSSH and Codex CLI. Add `~/.local/bin` to PATH.
RAG installs its own optional dependencies and runs embeddings locally on CPU.

## Settings

- `config/model.toml`: model-independent defaults and agent settings.
- `config/model.hosts.json`: private saved host profiles; ignored by Git.
- `run/resources/ID/`: allocation records, logs and generated catalogs.
- `run/rag/`: cached retrieval indexes, embeddings and isolated dependencies.

Keep session records to preserve IDs and recovery information. Logs for released
allocations can be deleted. Do not remove live session files or model/container data.

Use `res-alloc --restart` for connection, remote path, container and scheduler changes.
Supported modes: SSH/local connections; Slurm, Kubernetes or direct execution.
Slurm/Kubernetes options are requested for each allocation.

## Models and agent behavior

`./bin/llm-away list-models` lists installed remote presets. `run` discovers models
and selects MTP; GLM Flash uses embedded heads. Per-process model catalogs avoid
changing global Codex settings. Reasoning, compaction and tool-output limits come
from `config/model.toml` and refresh on each invocation.

`/compact` summarizes chat history; `/new` starts a fresh chat without releasing
resources. Rejected tool payloads are saved under the session's `tool-errors/` folder.

See [initialization](docs/initialization.md), [Codex integration](docs/codex.md),
[remote documentation](../remote/README.md), and [GLM setup](../remote/docs/glm-5.3-flash.md).
# Use a session as a helper for GPT / another primary model

Cleanup: `res-clean` previews identifiable local leftovers, lets you select item
numbers (or `all`), and asks for confirmation. `res-clean --preview` only lists;
`res-clean --session 3` limits cleanup to one session. Active or uncertain sessions
are preserved. Candidates include inactive-session logs, tool-error dumps,
generated catalogs, tracked temporary folders, stale managed MCP registrations,
and recorded orphan tunnel processes (PID and start time must match). Shared
framework SSH masters are offered only when no allocation processes are alive.
Models, environments, RAG caches, session identity records and unrelated SSH
connections are preserved. Release unwanted live allocations using `res-mon`
first. Managed MCP helpers exit after their registration is removed.
Future `run` launches give Codex a session-owned `TMPDIR`; arbitrary old `/tmp`
files and remote folders cannot be reliably attributed and are not deleted.
Tools that ignore `TMPDIR` also remain outside automatic cleanup. No files or
processes are removed merely by installing this command.

Quick setup (run the installer once to add `add-serve` to your PATH):
```sh
/Users/jarvis/alice/misc/LLM-AWAY/local/scripts/install-resource-tools.sh
run --session 2 --serve
# In another terminal, from the project you want searched:
add-serve --session 2
# Or explicitly select one or more folders:
add-serve --session 2 --rag /path/to/project
```
`add-serve` appends a managed `session_helper_2` entry to Codex's config
(`$CODEX_HOME/config.toml` or `~/.codex/config.toml`), preserving other settings.
Reload the MCP connection in Codex after adding it. Each session gets a separate
entry. Repeating the command replaces its managed entry. A background watcher
removes the entry when the provider exits; choosing to unload or release also removes it. Keeping the model in the
background preserves its MCP registration. Registered MCP processes exit within about two
seconds of cleanup, even during a request. Killing the allocation has the same
effect. This also works for allocations whose daemon predates this feature.
Codex may retain a disconnected tool in its current conversation until MCP is
reloaded; removing a config entry cannot force a live client to refresh its tool
list. Restarted models require `add-serve` again. Automatic cleanup applies only
to entries created by this command, not manually configured MCP entries below.

`session-tool` is a read-only MCP server with `summarize_project` (local RAG →
session model → short cited summary) and `summarize_text` (summarize supplied
content). The primary model sees the summary, not all retrieved source excerpts.
It can reduce primary-model input tokens; savings and summary accuracy depend
on the task. This is focused retrieval, not an exhaustive repository analysis.

1. Keep an existing `run --session 2` open, or load a dedicated helper without Codex:
   ```sh
   run --session 2 --serve
   ```
   Model selection and MTP work as usual. `--serve` returns to the shell after
   loading; use `res-mon` to stop the model or release resources. MCP
   disconnection does not stop or release the allocation.
2. Add this block to `~/.codex/config.toml`, replacing the folder and session:
   ```toml
   [mcp_servers.session_helper]
   command = "/Users/jarvis/alice/misc/LLM-AWAY/local/bin/session-tool"
   args = ["--session", "2", "--rag", "/absolute/path/to/your/project"]
   startup_timeout_sec = 120
   tool_timeout_sec = 600
   ```
   Repeat `--rag` for more local folders. Dependencies reuse the isolated RAG
   environment, installed on first use. Restart the MCP connection in Codex.
   Codex app and CLI share MCP configuration ([official documentation](https://developers.openai.com/codex/mcp)).
3. Tell your primary model: “Use session_helper to summarize relevant code before
   broad file reads. Verify cited files before editing.” This enables delegation;
   it does not force every query through the helper.

Only the explicitly selected folders are searched, with existing RAG ignore and
secret-file filtering. Retrieved excerpts are sent to the allocation's model;
summaries are returned to the primary model. Ignore filtering is not a guarantee
that arbitrary source text contains no secrets. Configure narrow project roots.
Summary output defaults to 4,000 characters (maximum 8,000); helper generation is
capped at 2,048 tokens, including any model reasoning. Empty results fail clearly.
The helper reuses the running model and tunnel: it never starts, replaces, or
releases a model. Concurrent helper calls serialize; sharing with an active agent
can queue requests and disrupt prompt-cache reuse, so a dedicated session is best.
Do not attach this helper to its own remote-model Codex session (recursive delegation).
No GPU or end-to-end tests were run for this addition. To check locally, ask for
a short project summary, verify its file citations, and confirm the allocation
remains running after disconnecting MCP.

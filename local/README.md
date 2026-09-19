## Reuse a loaded session

`run --session 3` now offers **Keep loaded model / Load a different model**.
Keeping it offers **Reopen agent / Keep running in background**; `--serve`
preselects background mode and returns to the shell once the model is ready.
Reusing skips model discovery and MTP selection and does not reload the weights.
Reopening uses the selected CLI's saved-conversation picker (choose the previous conversation);
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
agent first), and **3/F3** opens its live allocation/provider logs. Left/right selects
sessions; up/down scrolls logs and reply previews. Esc returns from
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

## Codex or Claude Code

`run --session N` detects `codex` and `claude` on PATH. With both installed,
it asks which CLI to use; with only one, it selects it without asking.
With neither installed, it reports an error. `--serve` needs neither CLI.
Use `run --session N --cli claude` or `--cli codex` to select explicitly.
Non-interactive launches with both installed require an explicit selection.
Selection precedence is `--cli`, `LLM_AWAY_CLI`, then `[agent].cli` in
`config/model.toml` (default `"auto"`). CLI options go after `--`, for example
`run --session 2 --cli claude -- --print "Explain this project"`.

Claude uses the same remote model and provider lifecycle, including streaming,
native tool calls, `--rag`, and its `--resume` conversation picker. Conversations
belong to their CLI; switching CLI does not transfer history. Claude gets
`[claude].instructions` appended to its system prompt, falling back to
`[codex].instructions` when blank. Claude retains its own permission settings.
No global Claude configuration is written; endpoint, model, and local placeholder
authentication are set only for the child process.

For the standalone provider workflow, use `local/bin/claude-away` from the
repository root (or `claude-away` after installing command links), alongside
`codex-away`. These dedicated launchers select the named CLI explicitly.

The remote llama.cpp build must support its native `/v1/messages` endpoint
(and `/v1/messages/count_tokens` for token counting). The local provider forwards
these requests, including streaming and upstream errors, to the existing SSH
tunnel. Older builds that lack these endpoints must be updated.
See [llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
and [Claude gateway configuration](https://code.claude.com/docs/en/llm-gateway).

## Shared models and llama.cpp installations

Run `res-alloc --restart` to configure or update a host. Setup asks, in order:

1. The full path to an existing `LLM-AWAY/remote` installation (your own or shared).
2. The models directory, containing preset folders with `model.env` and GGUF files.
3. The llama.cpp installation directory, containing `builds/<backend>/bin/llama-server`
   or a standard `build/bin/llama-server` checkout.
4. Your own writable state directory, accessible from compute nodes.

The following existing directories are offered as defaults when visible on that
host. Previously saved answers take precedence. All answers remain editable.

| Cluster | Models directory | llama.cpp installation directory |
| --- | --- | --- |
| EPN (my SSH alias: `epnh`) | `/scratch/csonnabe/LLM-AWAY/remote/models` | `/scratch/csonnabe/LLM-AWAY/remote` |
| Hydra (my SSH alias: `hydra`) | `/lustre/alice/users/csonnab/LLM-AWAY/remote/models` | `/lustre/alice/users/csonnab/LLM-AWAY/remote` |

Hydra container paths are
`/lustre/alice/users/csonnab/LLM-AWAY/remote/containers/llama-server-cuda.sif`
for NVIDIA and `.../containers/llama-server-rocm.sif` for AMD. Select the matching
image and GPU backend. Setup retains an existing container choice; otherwise it
offers the shared installation's container path. EPN normally uses native builds.

The installation answer is the parent of `builds/`, not the executable itself.
GPU-specific native builds are selected automatically. Runtime containers that
provide `/app/llama-server` use that binary; development containers can use the
shared `builds/<backend>-container` directories.

Paths are saved per host as `llamacpp.models_dir` and
`llamacpp.installation_dir` in `local/config/model.hosts.json`. Blank TOML values
retain the original checkout-relative behavior. No symlinks or modifications to
shared model/build directories are needed. Update both local and remote code to
use these settings. Users need read/traverse access to shared files and execute
access to binaries on login and compute nodes. Kubernetes users must also mount
these paths into their pods.

No remote clone or copy is required. Use the existing shared installation path
from the table as the `LLM-AWAY/remote` answer. Only the local client is needed
on your computer. Shared code, models and builds require read/traverse access;
executables also require execute access.

Setup asks for a separate state directory owned by the connecting user. It
suggests `/scratch/<remote-user>/.cache/llm-away` on EPN or
`/lustre/alice/users/<remote-user>/.cache/llm-away` on Hydra when the user directory
exists, falling back to the remote home cache. Choose a path visible on compute
nodes; Hydra login-only home paths are unsuitable. State and logs go under
`<state directory>/resources/`, leaving the shared installation unchanged.
Kubernetes mounts this state path separately, so it must be shared across the
login host and eligible worker nodes.

Run `res-alloc --restart` to save this layout, then create a new allocation.
Existing sessions keep their original state locations for compatibility. Shared
installations need the updated `remote/bin/resource-control` script. Do not
run download/build commands against another user's shared directories.

### Different SSH alias names

`epnh` and `hydra` are examples from the owner's SSH config, not required names.
A user can select their own configured alias (for example `my-epn`) or run
`res-alloc --host my-epn --restart`. SSH uses that user's `HostName`, `User`, and
`ProxyJump` settings. Setup detects shared default paths on the connected host,
independently of the alias, and stores the resulting profile under the selected
alias. Check the offered scheduler, partition, backend and GPU settings; these
remain configurable. Renaming an SSH alias creates a separate setup profile.

`res-clean --preview` now lists remote state/cache cleanup for released sessions.
Use `res-clean --session N` and select the remote cleanup item (or `all`). The
remote controller checks ownership, release status and worker shutdown before
removing only that session's state directory. Active or uncertain sessions are
preserved. Shared code/models/builds and the cache root remain. Local session
records are retained until remote cleanup succeeds, allowing retries if SSH is
offline. Update `remote/bin/resource-control` before using remote cleanup.

## Interactive background terminals

`run --session N` and `res-background --session N` now use a persistent tmux
terminal for the selected Codex/Claude agent. `run` attaches immediately;
`res-background` starts detached and returns to your shell.

- **F4 in res-mon:** attach to the existing agent, including while it is working.
- **Ctrl+B, then D:** detach to your shell without stopping the agent.
- **Mouse wheel / Ctrl+B then `[`:** scroll terminal history; press `q` to leave copy mode.
- **Space in res-mon:** send a prompt to that same terminal/conversation.
- Closing an attached terminal also detaches. Exiting the CLI itself ends the agent.

Space pastes into the CLI's current input. If the CLI is asking a question or is
busy, it follows that CLI's normal input/queue behavior; use F4 to inspect it.
The monitor shows captured terminal output. Existing headless tasks cannot be
converted into a terminal mid-flight; let those complete, then reopen with run.

`--agent-location local` remains the default. Local agents survive closing a
terminal but need the computer awake. Remote agents continue on the compute
host while the laptop sleeps, within the allocation's time limit. Both modes
retain shell, file and network access. `--agent-workdir` selects the project.

Install tmux on the agent host (`brew install tmux` on macOS). This Mac also
supports the repository-local binary at `local/run/tools/bin/tmux`. Remote
attachment uses SSH plus `srun --overlap --pty` for Slurm, `kubectl exec -it` for
Kubernetes, or direct SSH. The selected CLI must be installed on that host.
Use a new allocation after updating the remote workers. Releasing/unloading
stops its agent terminal; detaching does not release the allocation.

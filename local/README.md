## Reuse a loaded session

`run --session N` attaches to its existing tmux agent, or starts one.
Detach with **Ctrl+B, then D**; the agent and loaded model keep running.
`run --session N --detach` starts/reuses the agent and returns to the shell.
`run --session N --helper --rag /path/to/project` registers the loaded model
as a read-only helper for your primary Codex model, loading it if necessary.

# Local client

Run from this directory; see the [project guide](../README.md) for architecture and configuration.

```sh
./scripts/install-resource-tools.sh    # Once: install command links
res-alloc                            # Choose host and reserve resources
res-alloc --restart                  # Reconfigure a host
res-mon --list
run --session 2                      # Choose model and start the agent
run --session 2 --model glm-5.3-flash-q4 --mtp on
run --session 2 --detach             # Load and open the agent in background tmux
run --session 2 --rag .              # Optional local project retrieval
res-mon --logs 2                     # Ctrl+C only closes the log viewer
res-mon --kill 2 --release
```

Use the session ID returned by `res-alloc`. Exiting the agent lets you keep the
allocation (unloading the model) or release it. Independent sessions can use
different hosts/models. No separate init script is needed.

`run --session 2` starts model loading inside a retained tmux terminal, then
opens the CLI automatically when ready. Press **Ctrl+B, D** to detach during
loading; `run --session 2` or **4/F4** in `res-mon` reattaches.
`run --session 2 --detach` starts the same process and returns immediately.
Remote allocations use their custom model without asking about a native CLI.

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

`res-alloc` asks for location (`local` / `ssh`), then `Native CLI` / `Custom model`.
Native CLI asks for Claude or Codex and creates a numbered session using its
existing configuration and login, with zero GPUs and no model server. Both local
and SSH launches run in a retained local tmux terminal. Ctrl+B then D detaches;
`res-mon` shows live output/state, F4 (or `run --session ID`) reattaches or reopens,
and F2 releases the session. Exited sessions remain listed until released.
SSH launches require the chosen CLI on the SSH host; tmux is needed locally.
Custom model continues the allocation flow.
For example: `res-alloc --connection local --mode native --cli codex`.

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

Register a loaded model (or load one) as a helper for the main Codex model:

```sh
run --session 2 --helper --rag /path/to/project
```
Omit `--rag` to search the current directory; repeat it for multiple folders.
This returns to the shell without launching another agent CLI. It registers
`session_helper_2` in Codex's configuration. Restart the Codex MCP connection
(or app) and ask the main model to use that helper before broad file reads.
An already running agent is retained. Unloading/releasing the model removes the
registration; after loading it again, repeat `run --session 2 --helper`.
The former `add-serve` and `res-background` commands have been removed; rerun
`./scripts/install-resource-tools.sh` to remove their installed links.
Use `run --session 2 --detach` to start a tmux agent without attaching.

`session-tool` is a read-only MCP server with `summarize_project` (local RAG →
session model → short cited summary) and `summarize_text` (summarize supplied
content). The primary model sees the summary, not all retrieved source excerpts.
It can reduce primary-model input tokens; savings and summary accuracy depend
on the task. This is focused retrieval, not an exhaustive repository analysis.
Helper requests and responses are logged by default to
`run/resources/ID/helper.log`; use `run --session 2 --helper --no-log-helper`
to disable that. The log contains questions, retrieved excerpts, and model
responses.

After restarting Codex, tell the primary model: “Use session_helper_2 to
summarize relevant code before broad file reads. Verify cited files before editing.”
This enables delegation; it does not force every query through the helper.
Codex app and CLI share MCP configuration. No manual TOML edits are needed.

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
With neither installed, it reports an error. `--helper` needs neither CLI.
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

`run --session N` and `run --session N --detach` now use a persistent tmux
terminal for the selected Codex/Claude agent. `run` attaches immediately;
`run --detach` starts detached and returns to your shell.

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
terminal but need the computer awake. When launching a local agent without a
loaded model, `run` offers "Native CLI (own model/account)": this starts your
installed `codex` or `claude` CLI unchanged (ChatGPT/Claude sign-in, no AWAY
model, no GPU load). "Session model (AWAY inference)" keeps the previous
behavior. Remote agents continue on the compute host while the laptop sleeps,
within the allocation's time limit. Both modes retain shell, file and network
access. `--agent-workdir` selects the project.

Install tmux on the agent host (`brew install tmux` on macOS). This Mac also
supports the repository-local binary at `local/run/tools/bin/tmux`. Remote
attachment uses SSH plus `srun --overlap --pty` for Slurm, `kubectl exec -it` for
Kubernetes, or direct SSH. The selected CLI must be installed on that host.
Use a new allocation after updating the remote workers. Releasing/unloading
stops its agent terminal; detaching does not release the allocation.

`res-mon` shows **IS MASTER** (connected helper session IDs) and **IS HELPER**
(session IDs using this helper), sorted and comma-separated; `—` means none.
Both fields appear in session details and `--list`, plus table columns in wide
terminals. These track live MCP connections between numbered sessions, not just
global helper registrations. Restart existing helper MCP connections once to
enable tracking; connections from clients outside `res-mon` have no session ID.

**PS** in `res-mon` is the persistent allocation runner PID (native sessions: the
CLI terminal runner PID). Killing that PID, including with `kill -9`, triggers
an independent watchdog to stop the agent/provider and release its allocation.
Cleanup failures remain visible and retry while the watchdog is running; SSH must
be reachable for remote release. Existing live runners gain watchdogs when viewed
in `res-mon`. Detaching tmux does not terminate the runner. Normal native CLI exit
keeps its session available for reopening.

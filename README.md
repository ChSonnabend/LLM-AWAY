# LLM-AWAY

Optional local code/document retrieval:

```sh
run --session 2 --rag .
run --session 2 --rag ./src --rag ./docs
```

RAG is off unless requested. Paths are local to the machine running `run`.
First use installs isolated dependencies and downloads a small CPU embedding model
(`BAAI/bge-small-en-v1.5`). SQLite keyword search and semantic embeddings share a
persistent index under ignored `local/run/rag/`; no database service or extra GPU
allocation is needed. Multiple sessions can reuse the index.

The agent receives a `search_project` tool returning bounded excerpts with paths
and line numbers. Changed/deleted files are refreshed on launch and each search;
unchanged files reuse embeddings. Python functions/classes and Markdown headings
provide chunk boundaries; other text uses bounded overlapping chunks.

Git ignore rules, `.ragignore` patterns, common generated directories, symlinks,
large/binary files and common secret filenames/content are excluded. Secret detection
is heuristic: select only appropriate folders and add sensitive paths to `.ragignore`.
Embeddings/indexes stay local; retrieved excerpts go to the selected inference host.
Indexing supports code, Markdown and text (not PDF). Limits: 20,000 files, 50,000
chunks; select narrower folders for larger projects. Use `--rag` before `--` or any
agent prompt. Nothing is injected unless the agent invokes the search tool.

One repository for the local agent gateway and remote llama.cpp runners.

- [local/](local/README.md): host/model selection, local gateway, agent commands and configuration.
- [remote/](remote/README.md): models, containers and direct/Slurm/Kubernetes execution.

```bash
git clone git@github.com:ChSonnabend/LLM-AWAY.git
cd LLM-AWAY/local
./scripts/install-resource-tools.sh
res-alloc
run --session 1
```

When setup asks for the remote folder, select the full path ending in
`LLM-AWAY/remote`, not the repository root. Remote setup is run from that folder.
Models, containers, builds and saved host profiles remain ignored by Git.

## Existing installations

- Mac: `/Users/jarvis/alice/misc/LLM-AWAY/local`
- epnh: `/scratch/csonnabe/LLM-AWAY/remote`
- Hydra: `/lustre/alice/users/csonnab/LLM-AWAY/remote`

The remote installations have been moved as full directories, without compatibility
symlinks. On macOS, `LLM-AWAY-local` remains a compatibility symlink to
`LLM-AWAY/local` for the installed command launchers.
Both original Git histories are preserved through subtree merge commits. The
original GitHub repositories remain available as historical copies; new changes
belong here. Each installation's former Git metadata is preserved inside
`.git/migration-backup-local` or `.git/migration-backup-remote`.

Run `git pull --ff-only` from the repository root to update both components.
No model downloads or GPU jobs are required for migration.

## Multiple independent resource sessions

Install the commands once with `local/scripts/install-resource-tools.sh` and ensure
`~/.local/bin` is on PATH. Use `res-alloc --restart` to repeat host setup.

```bash
res-alloc                         # Choose connection/host/resources; returns an ID
res-alloc --host hydra --gpus 2    # Another independent allocation
res-mon --list                    # IDs, hosts, GPU counts, scheduler jobs, models
run --session 1                   # Choose model, load it, open Codex in this directory
run --session 2                   # In another terminal: independent agent/host
res-mon --logs 1                  # Follow allocation/model/provider telemetry; Ctrl+C detaches
res-mon                          # Arrow-key/number menu to stop or release a session
res-mon --kill 1                  # Ask: stop model only, or also release resources?
res-mon --kill 1 --release        # Explicitly release; also works without a terminal
```

`resource-allocator` / `res-alloc` and `resource-monitor` / `res-mon` are equivalent.
`run -s 1 --model PRESET --mtp off -- "Explain this repository"` skips model/MTP
selection and passes the final arguments to Codex. Agent work happens in the shell's
current directory; model inference happens on the allocated host.

Allocation runs in a detached local background process. It reserves one Slurm node
(with the requested GPU count), holds one Kubernetes Job, or starts a direct-host
worker. It does **not** load a model until `run`. Known host settings are remembered;
use `res-alloc --restart` to ask all setup questions again. Allocations snapshot their
configuration, so editing another host does not change existing sessions.

Each session has separate gateway/tunnel/model ports, remote control state, a log,
and per-process Codex configuration. Concurrent agents do not edit the global Codex
provider. Only one `run` client may own a session at a time. On agent exit, choose
whether to unload the model and keep the allocation, or free resources and end the
background process. Stopping from `res-mon` also terminates that foreground agent.
A retained allocation still consumes resources and remains subject to scheduler
walltime. A later `run` can select a different model on the same allocation.

Direct mode cannot reserve hardware against unrelated users/processes. The allocator
rejects GPU overlap with its own unreleased sessions on the same host alias. Choose
explicit device IDs. Slurm/Kubernetes provide actual scheduler reservations.
Kubernetes requires a writable project volume shared with the submitting machine
(hostPath or PVC), plus bash in the image. CPU-only direct mode is supported.

Session records/logs live in ignored `local/run/resources/ID/`; remote control files
live in `remote/.state/resources/TOKEN/`. `res-mon --kill ID --release` can recover an
allocation whose local background daemon has died. Closing a terminal does not free
a reservation; use the monitor. Release sessions before moving the checkout or
rebooting the Mac. Existing allocations are not automatically resumed after reboot.

Verification: one temporary CPU-only two-session lifecycle check; Slurm/Kubernetes
and real model/GPU inference still require a user smoke check. Try allocating one GPU,
starting a small model, exiting with “keep”, then running the same ID and releasing it.

Hydra: [GLM-5.3-Flash Q4 on two H200s, 750k context](remote/docs/glm-5.3-flash.md).

Configuration is `local/config/model.toml`, with saved hosts in `model.hosts.json`.
Use `res-alloc --restart` to configure a host afresh. Obsolete init scripts and
config compatibility links have been removed.
Every allocation asks for additional Slurm options, or Kubernetes CPU, memory,
node selector, tolerations, priority and time limit. Existing allocations are unchanged.

`run` uses the selected model’s actual identifier and display name, provider `remote_resource`, and a compact,
model-specific catalog enabled by `[codex] custom_metadata = true` in `model.toml`.
Set it to false to restore fallback metadata. Agent settings refresh on each `run`.
The configured compaction threshold is honored, capped at 70% of the active context;
the default project threshold is 180k tokens. Guidance discourages repeated SSH/inventory calls;
it cannot guarantee that a model will never loop. GLM's native tool syntax is accepted
and validated. GLM Flash MTP uses embedded weights with two draft tokens:
`run --session 2 --model glm-5.3-flash-q4 --mtp on`.
Exit and rerun an existing agent to pick up bridge changes; keep its allocation.

Server-backed sessions send native tool schemas and structured tool history to
llama.cpp. Custom patch tools are represented by a required string `input` parameter.
Rejected calls get one corrective retry and never execute. Exact rejected outputs
are saved locally with owner-only permissions in `local/run/resources/ID/tool-errors/`;
the error reports their paths. These files are ignored by Git and may contain code
or command arguments. Tool mistakes remain possible; no prompt guarantees their absence.
Reasoning effort is forwarded to the server; for GLM, `medium` maps to `high` and
`xhigh` to `max`. Output verbosity is guided by the concise instructions.

CLI support: `run --session N` asks between Codex and Claude Code when both are installed, or automatically uses the only available CLI. See the [local CLI guide](local/README.md#codex-or-claude-code) for configuration and explicit selection.

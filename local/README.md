# Local client

Run from this directory; see the [project guide](../README.md) for architecture and configuration.

```sh
./scripts/install-resource-tools.sh    # Once: install command links
res-alloc                            # Choose host and reserve resources
res-alloc --restart                  # Reconfigure a host
res-mon --list
run --session 2                      # Choose model and start the agent
run --session 2 --model glm-5.3-flash-q4 --mtp on
run --session 2 --rag .              # Optional local project retrieval
res-mon --logs 2                     # Ctrl+C only closes the log viewer
res-mon --kill 2 --release
```

Use the session ID returned by `res-alloc`. Exiting the agent lets you keep the
allocation (unloading the model) or release it. Independent sessions can use
different hosts/models. No separate init script is needed.

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

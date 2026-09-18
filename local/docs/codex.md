# Codex integration

Use `res-alloc` to reserve resources, then `run --session ID` to launch Codex.
`run` supplies an isolated provider URL and a model-specific catalog for that
process, without replacing global Codex settings. No separate init script is needed.

Agent settings live in `config/model.toml` under `[codex]`: reasoning effort,
custom metadata, instructions, tool-output budget and automatic compaction threshold.
They refresh on every `run`. The model preset can override the available context.

Use `run --session ID --rag FOLDER` for optional local retrieval. The index remains
local; relevant excerpts are sent to the selected inference server when requested.

Native tool schemas/history are forwarded to llama.cpp. Invalid calls receive one
corrective retry and are rejected before execution if still invalid. Rejected
payloads are saved locally under `run/resources/ID/tool-errors/`.

On exit, keep or release the allocation. Use `/compact` to summarize the conversation
or `/new` for a fresh conversation within the same CLI session.

Advanced legacy clients can still call `./bin/llm-away install-codex-config`
explicitly. This is not required by the resource-session workflow.

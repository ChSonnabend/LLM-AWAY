# Model API keys and OpenCode

Managed resource sessions generate a random 256-bit API key on the remote host when a model is next loaded. The key is stored in `resources/<allocation-token>/api-key` with owner-only permissions. llama.cpp receives the file path, never the secret in its command line. Native and container launches both support this.

Clients obtain the credential over their configured SSH connection. The local provider, Codex, Claude, OpenCode, background prompts and session helpers forward it automatically. A second machine attaching to the same allocation uses the same remote key; no copying or typing keys is required. The client caches its key in its session directory with owner-only permissions. Model inference and model-list requests require a bearer token; health checks remain public. The dashboard's session details show whether model API authentication is enabled.

Keys are not included in discovery, session descriptors, telemetry or browser responses. They restrict model API access; SSH and the remote Unix account still authorize allocation management. Clients sharing the same Unix account are mutually trusted. Keys do not add encryption: retain SSH tunnels and do not publish plain HTTP endpoints on untrusted networks. The web dashboard remains loopback-only and uses its existing access model; these keys protect model APIs, not dashboard login.

## Update existing installations

1. Sync the updated repository to both local clients and the remote framework installations, including `remote/bin/resource-*` and `remote/scripts/lib/container-env.sh`.
2. Restart local dashboards to load the new client code.
3. Existing loaded models continue unchanged. Authentication takes effect at the next model load. Old clients must be updated before attaching to protected models.
4. For remote agent/background-prompt support, create a fresh allocation after updating so its long-running prompt worker loads the new code. Existing local-agent sessions can continue using their retained allocations.

## OpenCode

Each chat header has an **Interface** selector for Codex, Claude and OpenCode. Switching restarts only that session’s agent, retains model and RAG settings, and checks availability before stopping the existing agent. Conversation histories remain in the original interface; they are not converted between applications.

Select **Attach → Machine & model loading → Agent CLI → opencode**, configure optional RAG, then choose **Attach to loaded model**. Its full-screen terminal appears in Chats and is also available through `res-mon` / F2. No model reload is needed solely to change the agent interface.

From the terminal: `run --session ID --cli opencode`. If an existing agent is already retained, use the web Attach action to change its CLI. Inside OpenCode, `/sessions` selects a saved conversation; the framework does not silently resume another conversation.

Install OpenCode on the machine where the agent runs. On macOS use `brew install opencode` (already installed on this Mac). On Linux follow the [official installation instructions](https://opencode.ai/docs/). Remote agents also require OpenCode on the compute node's PATH and a fresh allocation after installation so availability is detected.

LLM-AWAY supplies a process-local OpenCode configuration with the selected model, context limit, API credential, and `project_search` MCP connection. Only the AWAY provider is enabled, the small model uses the same allocation, and conversation sharing is disabled. Personal OpenCode configuration files are not rewritten. RAG remains local to the selected client or uses its configured remote bridge.

The implementation follows OpenCode's [provider configuration](https://opencode.ai/docs/providers/), [MCP configuration](https://opencode.ai/docs/mcp-servers/), and llama.cpp's [API-key-file support](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

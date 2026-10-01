# LLM-AWAY Inline Helper

Inline ghost-text code suggestions using a loaded model from your local
[LLM-AWAY](https://github.com/ChSonnabend/LLM-AWAY) installation. Accept with Tab;
dismiss with Escape. Requires desktop VS Code 1.85+ and a loaded, locally attached
LLM-AWAY resource session. After updating from 0.1.1 or earlier, reload the model
provider through LLM-AWAY so it exposes the new guarded inline endpoint, and run
**Developer: Reload Window** in VS Code. Old providers are detected on connection.
No separate inference service or subscription is needed.

## Install

From the LLM-AWAY repository:

```sh
./local/scripts/install.sh
```

The standard framework installer prepares Python and commands and automatically
builds/installs the extension when local VS Code and Node.js 20+ / npm are available.
It reports skipped or failed optional extension setup. Use `--with-vscode` to
require it, `--without-vscode` to skip it, or `--vscode-only` to install only the
extension. Auto mode skips SSH/VS Code Remote shells. `--code-command` selects
another CLI. The old `install-vscode-inline.sh` entry point remains an
extension-only wrapper.

This builds a VSIX and installs it using the selected `code` command. The
extension itself has no npm runtime dependencies. If VS Code requests it, reload
the window. You can also use **Extensions → … → Install from VSIX** with the
file generated in `local/run/artifacts/`. Direct `npm run package` builds in
`vscode-inline/`. The installer reads the package version from the manifest.

This is a local extension, not a Marketplace listing. Publishing requires a
Marketplace publisher account, an available publisher ID, and a separate release.
The manifest's `llm-away` publisher is a local package identifier, not a claim of
Marketplace ownership. Review licensing before public distribution.

## Connect

1. Allocate and load a model using LLM-AWAY as usual. A session needs a running
   local provider; an allocation alone or an account-backed native CLI cannot
   provide inline completions. An existing loaded session can be shared.
2. In the VS Code window you want to use, open the Command Palette and run
   **LLM-AWAY: Select Inline Helper Session**. The default repository path is
   `~/LLM-AWAY`; on first use you can select another folder. You can also change
   **LLM-AWAY Inline Helper: Repository Path** in user settings.
3. Choose a session and start typing. The status bar shows its number.

Alternatively, use the dashboard's **Tools → VSCode inline helper → Connect in
VS Code**. Install the extension first. The URI selects a session in the
extension's configured local repository; it does not transfer a remote dashboard's
configuration. With several windows open, use the command inside the intended
window to avoid URI routing ambiguity. VS Code Insiders users can use the command
instead of the stable `vscode://` link.

**LLM-AWAY: Suggest Inline Now** requests a completion manually.
**Toggle Automatic Suggestions** pauses/resumes automatic requests.
**Disconnect Inline Helper** stops this window's suggestions. It does not unload
models or release allocations. After reloading VS Code or restarting/replacing a
model, select the session again. Use LLM-AWAY to load, reconnect, stop, or release.

## Behavior and limits

- The extension runs on the desktop side, including Remote SSH windows, so it can
  reach the desktop's LLM-AWAY gateway. Browser-only VS Code is not supported.
- Only trusted workspaces are supported. On activation/connection and file
  switches, the extension captures the active editor's full buffer, including
  unsaved changes. Changed snapshots refresh every 30 seconds and immediately
  on save. `File Context Refresh Seconds` controls the interval.
- Each completion includes that stable snapshot followed by up to 8,000 current
  characters before the cursor and 2,000 after it. The current excerpts override
  potentially older snapshot text. Requests are stateless: sending a file once
  would not make it available to later requests. `cache_prompt` is enabled so
  llama.cpp can reuse the unchanged prompt prefix when available; sharing a model
  with other requests can reduce cache reuse. The first request after a snapshot
  refresh may take longer to process.
- Full-file context defaults to a 200,000-character maximum, also conservatively
  limited by the configured model context budget after reserving space for cursor
  text, instructions and output. Oversized files use cursor excerpts only, with a
  diagnostic explaining the omission. `File Context Max Chars` adjusts this limit;
  0 disables full-file context. Token estimates use JSON/UTF-8 byte counts rather
  than a model-specific tokenizer.
- Snapshots refresh locally without background inference or UI activity. Only
  the active file is cached; other files are not scanned and no RAG or agent tools
  run. Full snapshots and cursor excerpts go to the selected model host. Existing
  LLM-AWAY model query logging may retain this source text locally.
- Credentials are read from the existing owner-only session key file and kept in
  memory. They are not stored in VS Code settings, output logs, or connect links.
- Automatic requests wait 200 ms after typing by default. Output is limited to
  256 tokens and the client waits up to 15 seconds. Adjust these settings for your
  model; reasoning models may need a higher token limit to produce code. GLM
  requests use low reasoning effort for inline suggestions only.
- Each window keeps **one running inference and one newest pending request**.
  Further edits replace the pending request. The 200 ms typing delay still applies;
  cursor states invalidated by typing, moving, or closing the editor are skipped.
- Editing or disconnecting invalidates results and clears pending work without
  aborting an inference already sent. The 15-second display deadline hides late
  results but does not free the running slot. Only the newest pending request runs
  once the current one finishes. This prevents cancelled requests building up at
  the remote model.
- The updated gateway admits only one inline inference at a time across windows
  using that local provider. It drains the model response even if an editor closes.
  Another window receives a busy message instead of queuing more remote work; retry
  after the running request finishes. Independent gateways on other machines and
  ordinary agent requests are outside this inline limit.
- If an upstream network failure leaves inference status uncertain, the gateway
  blocks further inline requests until its provider is restarted. Its existing
  inference timeout remains the hard limit; the extension's display deadline does
  not cancel upstream work. The model's CUDA stability is a separate concern.
- Requests share the model's inference slot with any attached agents. A busy
  model can time out; a dedicated fast coding model works best. Auto mode uses native fill-in-the-middle (FIM) for Qwen Coder models and chat
  completion otherwise. `Completion Mode` can override this choice. For supported
  source languages, FIM receives formatting guidance and the snapshot as reference
  comments before the live source; chat receives structured messages. Unknown
  languages in forced FIM mode receive only cursor excerpts; choose chat for metadata.
- Other inline providers can compete for ghost text. Disable their automatic
  suggestions for this workspace if needed. VS Code's `editor.inlineSuggest.enabled`
  must be enabled.

- Language is taken from VS Code's language mode. Requests include the editor's
  spaces/tabs preference, tab size, current indentation and line endings.
- `Prefer Efficient Code` defaults to enabled. It asks for concise continuations,
  appropriate algorithms, existing library operations and fewer unnecessary copies
  or large temporary arrays. For Python with NumPy already imported, it favors
  suitable array operations over nested Python loops. It does not ban loops or
  guarantee runtime performance. Disable the preference in workspace settings
  if you prefer neutral guidance.
- Exact multiline echoes at the cursor are removed; substantial duplicated blocks
  (at least three nonblank lines and 80 characters) are suppressed. This conservative
  heuristic can also hide intentional repetition. Python string/comment contents
  are excluded. Chat responses that repeat the current indentation are normalized;
  native FIM whitespace is preserved.

## Development

```sh
cd vscode-inline
npm ci
npm test
npm run package
```

Run **LLM-AWAY: Show Inline Helper Diagnostics** to see request/response counts,
cancellations, token-limit exhaustion and autocomplete conflicts. Diagnostics
contain lengths and status only, never editor text. The manual **Suggest Inline
Now** command dismisses the regular autocomplete popup before requesting ghost
text. Entire-line and code-only fenced model responses are normalized to the
missing fragment. Inline Markdown backticks around Python code are removed;
quotes and backticks inside Python strings/comments are preserved. Errors also appear in the status bar tooltip.
The extension never includes request bodies or credentials in its error messages.

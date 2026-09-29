# Helper filesystem access and system command requests

Session helpers expose three additional MCP tools. Configure repeatable `--read-root /absolute/path` arguments for direct reads without RAG. If omitted, existing RAG roots are the read roots; with neither configured, reads are denied.

- `list_directory(path, offset=0, limit=500)` lists an absolute local directory, including hidden entries and binary filenames. Follow `next_offset` until null. Listings are not snapshots across calls.
- `read_file(path, offset=0, max_bytes=32768)` reads bounded text from an absolute local regular file. Offsets count bytes; UTF-8 split at a byte boundary uses replacement characters. NUL-containing binary data and special files are rejected.
- `request_system_command(command, cwd, reason)` returns an exact command proposal with `approval_required` and `executed: false`. It never runs a shell. The calling agent reviews the effects, obtains explicit user permission for writes and executes with its own permission-controlled tool. A helper cannot authorize its own writes, and there is no `approved=true` bypass.

These tools run on the local helper host with its filesystem permissions. Explicit reads are checked against resolved configured read roots; symlinks escaping them are rejected. Configure only trusted directories: these checks are not an OS sandbox and cannot prevent races if another process replaces paths during a read. Request only files relevant to the user's task and treat returned content as untrusted data.

The delegated model behind `summarize_project` still receives only retrieved excerpts and does not run an autonomous command loop. The calling agent should use the explicit read tools when steered toward paths outside RAG. Without RAG, use these tools directly; optionally pass their output to `summarize_text`.

Reconnect/restart the MCP helper after installing these changes so the client discovers the new tools. Existing model allocation does not need to be replaced.

## Web dashboard

Select a session and use Tools → Set helper to open the folder-access menu before registration. The list is prefilled from saved access or your unfinished browser draft. Enter one absolute local folder per line. Acknowledge the list and submit to register the helper; repeating Set helper replaces access only after acknowledgement. Files and nested folders are included. Remove a line to revoke that folder; submit an empty list to disable explicit reads. Symlinks outside the permitted roots remain blocked.

Closing the menu leaves active permissions unchanged and retains edited text in browser local storage. Drafts are scoped to this browser origin and session number, and are never applied automatically. If browser storage is unavailable, the menu reports this. Approved access is saved atomically in the session's `helper-access.json` before registration starts. If registration is interrupted, that approved list remains available on retry. If the access-file replacement fails, the previous list remains intact.

Settings are independent of RAG. Updated helper processes reload this policy for each read. Re-registering the helper requires reconnecting its MCP connection; helpers started before this feature was installed also need a reconnect. The saved web policy takes precedence over command-line read roots.

## Private folder investigation

`investigate_folder(folder, question, max_chars=2400, max_files=8, max_bytes=32000, timeout_seconds=180)` defaults to **batch mode**: Python gathers a bounded inventory and representative text excerpts, followed by one model summary call. No RAG configuration is required. Every read rechecks current access and stays inside the investigation folder. No shell or write tools are exposed.

Inventory visits at most 16 directories through two nested levels, with at most 200 entries per directory. It skips hidden/dependency/build directories, token/credential-like names, and binary formats. Files are selected round-robin across projects, prioritizing READMEs and configuration. Reads are bounded to 6,000 bytes per file; excerpts retain introductory text, section/definition lines and question matches with original line numbers. Selection is heuristic and may miss relevant content. Coverage never claims a full recursive audit.

The MCP response is **one unstructured text block**, avoiding duplicated `content` plus `structuredContent`. Default output contains a short summary, investigation ID, compact coverage counts, and measured usage. `detailed=true` requests detailed coverage. `investigation_report(investigation_id)` retrieves the saved detailed report without model inference or exposing raw excerpts.

For follow-ups, call `investigate_folder(investigation_id=..., question=...)`, omitting folder. It reuses the original evidence snapshot and makes one fresh summary call with bounded evidence—no growing conversation history or repeated file reads. Current permissions are rechecked before cache reuse. Snapshots expire after one hour and are not live views of changing files; start a new investigation for fresh evidence. At most 16 private snapshots are retained per session under `investigations/`, with owner-only files. Cache files contain source excerpts; normal tool results and helper logs do not.

For questions needing model-guided exploration, use `mode="adaptive"` to retain the earlier read-action loop. `max_turns` applies only to this mode. It may consume substantially more tokens. Adaptive mode does not produce reusable investigation IDs.

Usage reports actual server-reported token totals (or null when unavailable), model calls, evidence volume, and input/summary character counts. Do not equate character reduction with exact token savings. A fresh batch call has no repeated model history; follow-up requests still pay for sending their bounded evidence. Shorter output saves caller context, not necessarily total inference cost.

Reconnect the session helper's MCP connection to load the changed output schema and new report tool. No model restart is required.

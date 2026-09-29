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

## Deep, resumable document investigation

Use `investigate_folder(folder="/absolute/folder", mode="deep", question="...")` for a recursive document scan. There is no depth limit. Each request is bounded by `max_model_calls=10`, `timeout_seconds=180`, and `max_entries=20000` (configurable up to 100,000 discovered files). Quick-mode `max_files`, `max_bytes`, and `max_turns` do not apply to deep mode.

When the response says `in_progress`, call the same tool with `mode="deep"` and its `investigation_id`. Omit folder and question to resume the original request. Repeat until `complete`; no autonomous scheduled continuation is created. Each invocation reports its actual model usage and cumulative usage. Deep mode can use substantially more tokens than a quick overview.

Deep mode inventories first, then processes all eligible files. Long documents are split into bounded character chunks, maintaining a compact cumulative document summary. Document summaries are merged through a reduction tree before the final overview. Intermediate work is checkpointed atomically after directory scans, extraction, and completed model calls. An interrupted in-flight call may need repeating; completed chunks are reused. Changing the question requires a new investigation.

Resume checks size, modification time and file identity for known files. Changed files are reprocessed; unchanged extraction/chunk summaries are reused. `refresh=true` rescans the directory tree for additions and deletions. `retry_failed=true` retries extraction failures after missing dependencies or damaged documents have been addressed. Cached work remains subject to current read permissions on each resume and each read.

### Formats and limitations

- Text/source files, including LaTeX (`.tex`, `.bib`, `.sty`, `.cls`) are read as text, not executed or compiled.
- Jupyter notebooks include code and markdown cells; outputs and embedded attachments are excluded.
- PDF text uses Poppler `pdftotext`, with page markers. Figures/layout are not interpreted.
- Modern Office (`.docx`, `.pptx`, `.xlsx`) and OpenDocument (`.odt`, `.odp`, `.ods`) text is extracted from XML. Slides include notes; spreadsheet text and stored values are extracted without executing formulas. Formatting, charts and embedded media are excluded.
- `enable_ocr=true` enables image OCR and OCR for PDF pages with no extracted text. Requires `tesseract` plus `pdftoppm` for PDFs. At most 20 image-only PDF pages are OCR'd per document. Missing tools, remaining image-only pages and OCR failures are reported; they are never counted as fully processed. OCR may be inaccurate and does not interpret figures.
- Keynote, legacy binary Office formats, ROOT data, videos, archives, and other unsupported formats are explicitly skipped. No speech transcription, binary data analysis, or notebook execution occurs.

Extractors run in a separate process with CPU, memory, output-file and elapsed-time limits. Sources above 64 MiB fail extraction; expanded Office ZIP data is capped at 128 MiB. Extracted text beyond 2,000,000 characters is marked partial. Credentials, hidden/dependency/build directories and symlinks are excluded. The exact excluded subtrees are saved; their internal file counts are unknown. Filesystem calls remain subject to host filesystem responsiveness.

`investigation_report(investigation_id)` returns the per-file manifest (processed, partial, pending, skipped, failed), extraction notes, character progress and summaries. A compact response reports counts and gaps. `complete` means the available work and overall summary are finished; it does **not** mean every file was readable or visually understood. Check `all_discovered_files_fully_processed` and exclusions. `processed` means all extracted text reached the chunk-summary pipeline; summaries themselves are lossy.

Private extracted text and checkpoints live under the session's `deep-investigations/<id>/` directory, using owner-only files. Unlike quick snapshots, these do not expire after one hour, so long scans can resume later. Source files are never modified. Deep snapshots currently require manual removal when no longer needed; no automatic deletion or retention promise is made.

### macOS and system dependencies

PDF/OCR executables are declared in `local/requirements/documents-system.json`, separately from pip requirements. Python wrappers alone do not install the Tesseract engine. On the helper host, run from the checkout:

```sh
local/bin/setup-documents --check
local/bin/setup-documents --install
```

The installer uses `brew install poppler tesseract` on macOS (Homebrew must already be installed), or `apt-get install poppler-utils tesseract-ocr tesseract-ocr-eng` on apt-based Linux, using sudo when needed. It verifies executable availability and English OCR language data. It does not install packages during ordinary helper startup or reads. Other Linux distributions can install the corresponding system packages manually and use `--check`.

`local/scripts/init --documents` installs the same system requirements while preparing the project Python environment; `--check-documents` checks without installing. The default `init` does not change system packages.

The runtime checks PATH and, on macOS, the standard Apple Silicon Homebrew (`/opt/homebrew/bin`), Intel Homebrew (`/usr/local/bin`), and MacPorts (`/opt/local/bin`) locations. This also covers helpers launched from desktop applications without a shell PATH. macOS uses CPU, output-file-size, source-size and elapsed-time limits; the Linux RLIMIT_AS address-space cap is intentionally not applied on macOS. Unsupported individual resource limits do not disable the remaining limits. Native macOS execution still needs validation on an actual Mac; platform-specific discovery and limit handling are covered by mocked tests.

OCR uses Tesseract's default English language. Additional language packs and language selection are not automatically enabled by this installer. Set `enable_ocr=true` on deep investigations to request OCR; package installation alone does not enable it.

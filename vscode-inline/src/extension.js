'use strict';
const vscode = require('vscode');
const client = require('./client');
const { LatestQueue } = require('./latest-queue');
const { FileContext, contextByteBudget } = require('./file-context');

function activate(context) {
  const queue = new LatestQueue();
  const fileContext = new FileContext();
  let omittedContextKey;
  let connected;
  let active;
  let activeDocument;
  let revision = 0;
  let connectionRevision = 0;
  let lastError = '';
  const output = vscode.window.createOutputChannel('LLM-AWAY Inline Helper');
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 20);
  const config = () => vscode.workspace.getConfiguration('llmAwayInline');
  const root = () => config().get('repositoryPath', '~/LLM-AWAY');
  function cancel() { revision++; active?.abort(); queue.clearPending(); }
  function updateStatus(error) {
    status.text = connected ? `$(sparkle) AWAY ${connected.session.id}${config().get('automatic') ? '' : ' (manual)'}` : '$(plug) AWAY';
    status.tooltip = error || (connected ? `${connected.session.model} · Click to switch session. Use LLM-AWAY commands to disconnect or pause.` : 'Select a loaded LLM-AWAY session for inline suggestions');
    status.command = 'llmAwayInline.connect';
    status.show();
  }
  function trace(message) { output.appendLine(`${new Date().toISOString()} ${message}`); }
  function captureFile(document, force = false, maxBytes = contextByteBudget(connected?.session)) {
    if (!document || document.isClosed || !vscode.workspace.isTrusted ||
        !['file', 'untitled', 'vscode-remote'].includes(document.uri.scheme)) return undefined;
    const snapshot = fileContext.capture(document, {
      refreshMs: config().get('fileContextRefreshSeconds', 30) * 1000,
      maxChars: config().get('fileContextMaxChars', 200000),
      maxBytes
    }, force);
    if (snapshot.omitted) {
      const key = `${snapshot.key}:${snapshot.version}:${snapshot.limits}`;
      if (key !== omittedContextKey) trace(`Full-file context exceeds the configured/model budget (${snapshot.totalChars} characters); using cursor excerpts only.`);
      omittedContextKey = key;
    } else omittedContextKey = undefined;
    return snapshot;
  }
  function report(error) {
    const message = error.message || String(error);
    updateStatus(message);
    if (message !== lastError) trace(message);
    lastError = message;
  }
  async function connect(id) {
    if (!vscode.workspace.isTrusted) throw new Error('Trust this workspace before enabling the inline helper.');
    cancel();
    const ticket = ++connectionRevision;
    let repository = root();
    let session;
    if (id !== undefined) session = await client.readSession(repository, id);
    else {
      let sessions;
      try { sessions = await client.listSessions(repository); }
      catch (error) {
        if (error.code !== 'ENOENT') throw error;
        const folders = await vscode.window.showOpenDialog({ canSelectFolders: true, canSelectFiles: false, canSelectMany: false, openLabel: 'Select LLM-AWAY repository' });
        if (!folders?.length) return;
        repository = folders[0].fsPath;
        await config().update('repositoryPath', repository, vscode.ConfigurationTarget.Global);
        sessions = await client.listSessions(repository);
      }
      if (!sessions.length) throw new Error('No loaded sessions found. Load a model in LLM-AWAY, then select it here.');
      const choice = await vscode.window.showQuickPick(sessions.map(s => ({ label: `Session ${s.id}: ${s.model}`, description: s.host, session: s })), { placeHolder: 'Choose the model for this VS Code window' });
      if (!choice) return;
      session = choice.session;
    }
    const inlineStatus = await client.request(session, '/v1/inline/status', undefined, undefined, 4000);
    if (!inlineStatus.supported || inlineStatus.fault) throw new Error('Reload the model provider in LLM-AWAY before reconnecting the inline helper.');
    const models = await client.request(session, '/v1/models', undefined, undefined, 4000);
    if (!models.data?.some(model => model.id === session.model)) throw new Error('Provider model changed. Refresh the session list.');
    const current = await client.readSession(repository, session.id);
    if (current.identity !== session.identity) throw new Error('Session changed while connecting. Select it again.');
    // A later connect/disconnect wins over any pending selection.
    if (ticket !== connectionRevision || client.repositoryPath(repository) !== client.repositoryPath(root())) return;
    connected = { repository, session };
    captureFile(vscode.window.activeTextEditor?.document, true);
    trace(`Connected to session ${session.id} (${session.model}).`);
    lastError = '';
    updateStatus();
    vscode.window.showInformationMessage(`LLM-AWAY inline helper connected to session ${session.id} (${session.model}).`);
  }
  async function commandConnect(id) {
    try { await connect(id); } catch (error) { report(error); vscode.window.showErrorMessage(error.message); }
  }
  const provider = {
    async provideInlineCompletionItems(document, position, inlineContext, token) {
      const explicit = inlineContext.triggerKind === vscode.InlineCompletionTriggerKind.Invoke;
      if (!connected || !vscode.workspace.isTrusted || token.isCancellationRequested ||
          (!explicit && !config().get('automatic')) || !['file', 'untitled', 'vscode-remote'].includes(document.uri.scheme)) return [];
      cancel();
      const ticket = revision;
      const snapshot = connected;
      const version = document.version;
      const controller = new AbortController();
      active = controller;
      activeDocument = document;
      let sent = false;
      const cancellation = token.onCancellationRequested(() => controller.abort());
      const valid = () => !controller.signal.aborted && !token.isCancellationRequested && ticket === revision &&
        connected === snapshot && document.version === version && !document.isClosed;
      try {
        if (!explicit) await new Promise(resolve => {
          const done = () => { clearTimeout(timer); controller.signal.removeEventListener('abort', done); resolve(); };
          const timer = setTimeout(done, config().get('debounceMs', 200));
          controller.signal.addEventListener('abort', done, { once: true });
        });
        if (!valid()) return [];
        const session = await client.readSession(snapshot.repository, snapshot.session.id);
        if (session.identity !== snapshot.session.identity) {
          connected = undefined;
          throw new Error('The model session changed or restarted. Select an inline helper session again.');
        }
        if (!valid()) return [];
        const offset = document.offsetAt(position);
        const prefix = document.getText(new vscode.Range(document.positionAt(Math.max(0, offset - 8000)), position));
        const suffix = document.getText(new vscode.Range(position, document.positionAt(offset + 2000)));
        trace(`Request ${ticket}: ${queue.active ? 'replaced pending request; waiting for current inference' : 'ready'}.`);
        const response = await queue.submit(async () => {
          // The generation may have changed while this cursor state waited.
          const current = await client.readSession(snapshot.repository, session.id);
          if (!valid() || current.identity !== session.identity) return undefined;
          const fullFile = captureFile(document, false, contextByteBudget(session, prefix, suffix, config().get('maxTokens', 256)));
          sent = true;
          trace(`Request ${ticket}: ${explicit ? 'manual' : 'automatic'}, ${document.languageId}, prefix=${prefix.length}, suffix=${suffix.length}, fileContext=${fullFile?.content?.length || 0}, snapshotVersion=${fullFile?.version ?? 'none'}, autocomplete=${Boolean(inlineContext.selectedCompletionInfo)}.`);
          updateStatus('Waiting for an inline suggestion…');
          // Keep this HTTP request open through editor cancellation and the UI
          // deadline. The gateway has its own finite upstream timeout and guard.
          return client.request(session, '/v1/inline/completions',
            client.completionPayload(session.model, prefix, suffix, document.languageId, config().get('maxTokens', 256), fullFile, config().get('completionMode', 'auto')),
            undefined, 0);
        }, valid, config().get('timeoutMs', 15000), () => {
          trace(`Request ${ticket}: display deadline reached; retaining the inference slot until completion.`);
          if (valid()) updateStatus('Model is still processing; only the newest pending suggestion is retained.');
        });
        if (!response) return [];
        const message = response.choices?.[0]?.message || {};
        const finish = response.choices?.[0]?.finish_reason || 'unknown';
        trace(`Response ${ticket}: content=${message.content?.length || 0}, reasoning=${message.reasoning_content?.length || 0}, finish=${finish}.`);
        if (!valid()) { trace(`Request ${ticket}: discarded because the editor changed.`); return []; }
        const latest = await client.readSession(snapshot.repository, session.id);
        if (!valid() || latest.identity !== session.identity) return [];
        let text = client.completionText(response, suffix, prefix, document.languageId);
        if (!text.trim()) {
          const reason = `No insertable code returned (finish=${finish}, reasoning=${message.reasoning_content?.length || 0} characters).`;
          trace(`Request ${ticket}: ${reason}`); updateStatus(reason);
          if (explicit) vscode.window.showWarningMessage(`LLM-AWAY: ${reason} Try increasing Max Tokens if the response reached its limit.`);
          return [];
        }
        let range = new vscode.Range(position, position);
        const selected = inlineContext.selectedCompletionInfo;
        if (selected) {
          // VS Code only renders a preview alongside the popup when it extends
          // the selected item and uses exactly that item's replacement range.
          if (selected.range.start.line !== position.line || !selected.range.contains(position)) return [];
          const candidate = document.getText(new vscode.Range(selected.range.start, position)) + text;
          if (!candidate.startsWith(selected.text)) {
            trace(`Request ${ticket}: hidden by a conflicting autocomplete item; use Suggest Inline Now.`);
            updateStatus('Autocomplete popup conflicts with this suggestion. Use LLM-AWAY: Suggest Inline Now.');
            return [];
          }
          text = candidate; range = selected.range;
        }
        lastError = ''; updateStatus();
        trace(`Request ${ticket}: offered ${text.length} characters to VS Code.`);
        return [new vscode.InlineCompletionItem(text, range)];
      } catch (error) {
        if (controller.signal.aborted) { if (sent) trace(`Request ${ticket}: cancelled by the editor.`); }
        else if (ticket === revision) { report(error); if (explicit) vscode.window.showWarningMessage(`LLM-AWAY: ${error.message}`); }
        return [];
      } finally {
        cancellation.dispose();
        if (active === controller) { active = undefined; activeDocument = undefined; }
      }
    }
  };
  context.subscriptions.push(output, status,
    vscode.languages.registerInlineCompletionItemProvider([{ scheme: 'file' }, { scheme: 'untitled' }, { scheme: 'vscode-remote' }], provider),
    vscode.commands.registerCommand('llmAwayInline.connect', id => commandConnect(id)),
    vscode.commands.registerCommand('llmAwayInline.disconnect', () => { connectionRevision++; cancel(); connected = undefined; updateStatus(); }),
    vscode.commands.registerCommand('llmAwayInline.toggle', async () => { cancel(); await config().update('automatic', !config().get('automatic'), vscode.ConfigurationTarget.Global); updateStatus(); }),
    vscode.commands.registerCommand('llmAwayInline.suggest', async () => {
      if (!connected) await commandConnect();
      if (connected) {
        await vscode.commands.executeCommand('hideSuggestWidget');
        await vscode.commands.executeCommand('editor.action.inlineSuggest.trigger');
      }
    }),
    vscode.window.registerUriHandler({ async handleUri(uri) {
      if (uri.path !== '/connect') return;
      const id = new URLSearchParams(uri.query).get('session');
      if (!id || !/^[1-9]\d*$/.test(id)) return;
      await commandConnect(id);
    } }),
    vscode.commands.registerCommand('llmAwayInline.showOutput', () => output.show(true)),
    vscode.workspace.onDidChangeTextDocument(event => {
      if (event.document === activeDocument && event.contentChanges.length) cancel();
    }),
    vscode.window.onDidChangeActiveTextEditor(editor => {
      cancel(); captureFile(editor?.document, true);
    }),
    vscode.workspace.onDidSaveTextDocument(document => {
      if (document === vscode.window.activeTextEditor?.document) captureFile(document, true);
    }),
    vscode.workspace.onDidCloseTextDocument(document => {
      if (fileContext.snapshot?.key === document.uri.toString()) fileContext.clear();
    }),
    vscode.workspace.onDidChangeConfiguration(event => {
      if (!event.affectsConfiguration('llmAwayInline')) return;
      cancel();
      if (event.affectsConfiguration('llmAwayInline.repositoryPath')) connected = undefined;
      fileContext.clear();
      captureFile(vscode.window.activeTextEditor?.document, true);
      updateStatus();
    }),
    { dispose: cancel });
  captureFile(vscode.window.activeTextEditor?.document, true);
  // Poll locally for changed active buffers. No model traffic or editor commands.
  const refreshTimer = setInterval(() => {
    if (connected) captureFile(vscode.window.activeTextEditor?.document);
  }, 1000);
  context.subscriptions.push({ dispose: () => { clearInterval(refreshTimer); fileContext.clear(); } });
  updateStatus();
}
module.exports = { activate };

'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const Module = require('node:module');
const client = require('../src/client');

test('connects, suggests, discards edited buffers, pauses and disconnects', async t => {
  const commands = new Map();
  const events = {};
  const settings = { repositoryPath: '/test', automatic: true, debounceMs: 1, maxTokens: 128, timeoutMs: 1000 };
  let provider; let requests = 0; let response;
  const session = { id: '1', model: 'coder', identity: 'generation', port: 1234 };
  const original = { readSession: client.readSession, request: client.request };
  t.after(() => Object.assign(client, original));
  client.readSession = async () => session;
  client.request = async (_s, route, _payload, signal, timeout) => {
    if (route === '/v1/inline/status') return { supported: true };
    if (route === '/v1/models') return { data: [{ id: 'coder' }] };
    assert.equal(route, '/v1/inline/completions');
    assert.equal(signal, undefined); assert.equal(timeout, 0);
    requests++;
    return response || { choices: [{ message: { content: 'return 42;' } }] };
  };
  const disposable = { dispose() {} };
  const vscode = {
    StatusBarAlignment: { Right: 1 }, ConfigurationTarget: { Global: 1 }, EndOfLine: { LF: 1, CRLF: 2 }, InlineCompletionTriggerKind: { Invoke: 0, Automatic: 1 },
    Range: class { constructor(start, end) { this.start = start; this.end = end; } },
    InlineCompletionItem: class { constructor(text, range) { this.insertText = text; this.range = range; } },
    workspace: { isTrusted: true,
      getConfiguration: () => ({ get: (key, fallback) => settings[key] ?? fallback, update: async (key, value) => settings[key] = value }),
      onDidChangeTextDocument: callback => { events.edit = callback; return disposable; },
      onDidSaveTextDocument: () => disposable, onDidCloseTextDocument: () => disposable,
      onDidChangeConfiguration: () => disposable },
    window: { createOutputChannel: () => ({ ...disposable, appendLine() {} }), createStatusBarItem: () => ({ ...disposable, show() {} }),
      showInformationMessage() {}, showErrorMessage(message) { throw new Error(message); },
      onDidChangeActiveTextEditor: () => disposable, registerUriHandler: () => disposable },
    commands: { registerCommand: (name, handler) => { commands.set(name, handler); return disposable; } },
    languages: { registerInlineCompletionItemProvider: (_selector, value) => { provider = value; return disposable; } }
  };
  const load = Module._load;
  Module._load = function(name, ...args) { return name === 'vscode' ? vscode : load.call(this, name, ...args); };
  let extension;
  try { extension = require('../src/extension'); } finally { Module._load = load; }
  const context = { subscriptions: [] }; extension.activate(context);
  t.after(() => context.subscriptions.forEach(item => item.dispose()));
  await commands.get('llmAwayInline.connect')('1');
  const doc = { uri: { scheme: 'file' }, version: 1, languageId: 'javascript', isClosed: false,
    offsetAt: p => p, positionAt: p => p, getText: () => '' };
  const token = { isCancellationRequested: false, onCancellationRequested: () => disposable };
  const suggest = (triggerKind = 0) => provider.provideInlineCompletionItems(doc, 0, { triggerKind }, token);
  assert.equal((await suggest())[0].insertText, 'return 42;');
  let finish;
  response = new Promise(resolve => finish = resolve);
  const pending = suggest();
  await new Promise(resolve => setImmediate(resolve));
  doc.version++; events.edit({ document: doc, contentChanges: [{}] });
  finish({ choices: [{ message: { content: 'stale' } }] });
  assert.deepEqual(await pending, []);
  response = new Promise(resolve => finish = resolve);
  const unaffected = suggest();
  await new Promise(resolve => setImmediate(resolve));
  events.edit({ document: {}, contentChanges: [{}] });
  events.edit({ document: doc, contentChanges: [] });
  finish({ choices: [{ message: { content: 'still valid' } }] });
  assert.equal((await unaffected)[0].insertText, 'still valid');
  response = { choices: [{ message: { content: 'nt(42)' } }] };
  doc.getText = range => range.start === 0 ? 'pri' : '';
  const selectedRange = { start: 0, end: 3, contains: () => true };
  const popup = await provider.provideInlineCompletionItems(doc, 3,
    { triggerKind: 0, selectedCompletionInfo: { range: selectedRange, text: 'print' } }, token);
  assert.equal(popup[0].insertText, 'print(42)');
  assert.equal(popup[0].range, selectedRange);
  doc.getText = () => '';
  response = undefined;
  await commands.get('llmAwayInline.toggle')();
  const before = requests;
  assert.deepEqual(await suggest(1), []); assert.equal(requests, before);
  assert.equal((await suggest(0))[0].insertText, 'return 42;');
  await commands.get('llmAwayInline.disconnect')();
  assert.deepEqual(await suggest(), []);
});

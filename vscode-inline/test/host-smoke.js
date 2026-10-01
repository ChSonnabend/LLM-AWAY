'use strict';
// Run with an isolated VS Code --user-data-dir and --extensionTestsPath.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const vscode = require('vscode');

exports.run = async function() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'away-vscode-host-'));
  let received;
  const server = http.createServer((req, res) => {
    if (req.headers.authorization !== 'Bearer smoke-key') { res.writeHead(401); res.end(); return; }
    res.setHeader('Content-Type', 'application/json');
    if (req.url === '/v1/inline/status') { res.end(JSON.stringify({ supported: true })); return; }
    if (req.url === '/v1/models') { res.end(JSON.stringify({ data: [{ id: 'test-coder' }] })); return; }
    let raw = ''; req.on('data', chunk => raw += chunk);
    req.on('end', () => {
      received = JSON.parse(raw);
      const language = JSON.parse(received.messages.at(-1).content).language;
      const content = language === 'python' ? "`(['rm', '-f', str(archive)])`" : '```javascript\nconst answer = foo();\n```';
      res.end(JSON.stringify({ choices: [{ message: { content } }] }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const dir = path.join(root, 'local/run/resources/1'); await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(path.join(dir, 'session.json'), JSON.stringify({ id: 1, token: 'smoke', model: 'test-coder',
      provider_pid: process.pid, provider_identity: 'smoke', allocation: { model_state: 'LOADED', generation: 'one' },
      config: { server: { host: '127.0.0.1', port: server.address().port } } }));
    await fs.writeFile(path.join(dir, 'api-key'), 'smoke-key\n');
    await vscode.workspace.getConfiguration('llmAwayInline').update('repositoryPath', root, vscode.ConfigurationTarget.Global);
    await vscode.workspace.getConfiguration('llmAwayInline').update('automatic', false, vscode.ConfigurationTarget.Global);
    const extension = vscode.extensions.getExtension('llm-away.llm-away-inline');
    assert.ok(extension, 'extension discovered'); await extension.activate();
    await vscode.commands.executeCommand('llmAwayInline.connect', '1');
    const document = await vscode.workspace.openTextDocument({ language: 'javascript', content: 'const answer = fo' });
    const editor = await vscode.window.showTextDocument(document);
    const pos = document.positionAt(document.getText().length);
    editor.selection = new vscode.Selection(pos, pos);
    await new Promise(resolve => setTimeout(resolve, 500));
    await vscode.commands.executeCommand('workbench.action.focusActiveEditorGroup');
    assert.equal(vscode.window.activeTextEditor.document.uri.toString(), document.uri.toString());
    const completions = vscode.languages.registerCompletionItemProvider('javascript', {
      provideCompletionItems() { return [new vscode.CompletionItem('foo')]; }
    });
    await vscode.commands.executeCommand('editor.action.triggerSuggest');
    await new Promise(resolve => setTimeout(resolve, 300));
    await vscode.commands.executeCommand('llmAwayInline.suggest');
    completions.dispose();
    const deadline = Date.now() + 8000;
    while (!received && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 100));
    assert.ok(received, 'real VS Code inline provider sent request');
    assert.equal(received.max_tokens, 256);
    assert.equal(received.cache_prompt, true);
    const formatting = JSON.parse(received.messages.at(-1).content).formatting;
    assert.equal(formatting.language, 'JavaScript');
    assert.equal(formatting.tab_size, editor.options.tabSize);
    assert.equal(formatting.indent_unit, editor.options.insertSpaces ? ' '.repeat(editor.options.tabSize) : '\t');
    assert.equal(JSON.parse(received.messages[1].content).file_snapshot.content, 'const answer = fo');
    assert.equal(vscode.workspace.getConfiguration('llmAwayInline').get('debounceMs'), 200);
    assert.equal(JSON.parse(received.messages.at(-1).content).prefix, 'const answer = fo');
    await new Promise(resolve => setTimeout(resolve, 700));
    await vscode.commands.executeCommand('editor.action.inlineSuggest.commit');
    assert.equal(document.getText(), 'const answer = foo();', 'completion accepted into real editor');
    const python = await vscode.workspace.openTextDocument({ language: 'python', content: '        subprocess.call' });
    const pyEditor = await vscode.window.showTextDocument(python);
    const pyPos = python.positionAt(python.getText().length);
    pyEditor.selection = new vscode.Selection(pyPos, pyPos);
    received = undefined;
    await new Promise(resolve => setTimeout(resolve, 300));
    await vscode.commands.executeCommand('llmAwayInline.suggest');
    const pyDeadline = Date.now() + 5000;
    while (!received && Date.now() < pyDeadline) await new Promise(resolve => setTimeout(resolve, 50));
    assert.ok(received, 'Python fragment requested');
    await new Promise(resolve => setTimeout(resolve, 500));
    await vscode.commands.executeCommand('editor.action.inlineSuggest.commit');
    assert.equal(python.getText(), "        subprocess.call(['rm', '-f', str(archive)])", 'no Markdown backticks inserted');
    await vscode.commands.executeCommand('llmAwayInline.disconnect');
    console.log('LLM-AWAY extension host smoke test passed');
  } finally {
    server.closeAllConnections(); server.close();
    await fs.rm(root, { recursive: true, force: true });
  }
};

'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const client = require('../src/client');

async function fixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'away-inline-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const directory = client.sessionDirectory(root, 1);
  await fs.mkdir(directory, { recursive: true });
  const data = { id: 1, token: 'allocation', model: 'coder', provider_pid: 123, provider_identity: 'born',
    config: { server: { host: '127.0.0.1', port: 12345 } }, allocation: { generation: 'one', model_state: 'LOADED' } };
  const save = async () => fs.writeFile(path.join(directory, 'session.json'), JSON.stringify(data));
  await save();
  await fs.writeFile(path.join(directory, 'api-key'), 'secret\n');
  return { root, data, save };
}
test('discovers loaded sessions and pins provider generation', async t => {
  const f = await fixture(t);
  const [session] = await client.listSessions(f.root);
  assert.equal(session.key, 'secret');
  f.data.allocation.generation = 'two'; await f.save();
  assert.notEqual((await client.readSession(f.root, 1)).identity, session.identity);
  f.data.provider_exit = 0; await f.save();
  assert.deepEqual(await client.listSessions(f.root), []);
});
test('rejects native, ended, mismatched and non-loopback sessions', async t => {
  const f = await fixture(t);
  for (const change of [ { native: true }, { allocation_cleaned: true }, { model: '' }, { attachment_generation: 'old' } ]) {
    Object.assign(f.data, change); await f.save();
    await assert.rejects(client.readSession(f.root, 1));
    for (const key of Object.keys(change)) delete f.data[key];
    f.data.model = 'coder';
  }
  f.data.config.server.host = 'example.com'; await f.save();
  await assert.rejects(client.readSession(f.root, 1), /loopback/);
  assert.throws(() => client.sessionDirectory(f.root, '../1'), /Invalid/);
});
test('bounds context and never substitutes reasoning or tool calls for code', () => {
  const payload = client.completionPayload('coder', 'p'.repeat(12000), 's'.repeat(5000), 'python', 64);
  const data = JSON.parse(payload.messages[1].content);
  assert.equal(data.prefix.length, 8000); assert.equal(data.suffix.length, 2000);
  assert.equal(client.completionPayload('glm-5.3-flash-q4', '', '', 'python', 128).chat_template_kwargs.reasoning_effort, 'low');
  assert.equal(payload.max_tokens, 64); assert.equal(payload.tools, undefined);
  const response = message => ({ choices: [{ message }] });
  assert.equal(client.completionText(response({ content: 'return x;\n}' }), '\n}'), 'return x;');
  assert.equal(client.completionText(response({ reasoning_content: 'private thoughts' }), ''), '');
  assert.equal(client.completionText(response({ content: '```python\nx\n```' }), ''), 'x');
  assert.equal(client.completionText(response({ content: 'x', tool_calls: [{}] }), ''), '');
  assert.equal(client.completionText(response({ content: '  x' }), ''), '  x');
});
test('authenticates requests; supports cancellation, deadlines and HTTP errors', async t => {
  const server = http.createServer((req, res) => {
    if (req.url === '/hang') return;
    assert.equal(req.headers.authorization, 'Bearer secret');
    if (req.url === '/error') { res.writeHead(401); res.end('sensitive upstream error'); return; }
    let raw = ''; req.on('data', chunk => raw += chunk);
    req.on('end', () => { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({ received: JSON.parse(raw) })); });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => { server.closeAllConnections(); server.close(); });
  const session = { port: server.address().port, key: 'secret' };
  assert.deepEqual(await client.request(session, '/', { text: 'hello' }), { received: { text: 'hello' } });
  await assert.rejects(client.request(session, '/error', {}), /^Error: Provider HTTP 401;/);
  await assert.rejects(client.request(session, '/hang', {}, undefined, 30), /timed out/);
  const controller = new AbortController();
  const pending = client.request(session, '/hang', {}, controller.signal);
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
});

test('extracts missing fragment when model repeats a partial line', () => {
  const response = content => ({ choices: [{ message: { content } }] });
  assert.equal(client.completionText(response('    print("hello")'), '', 'def f():\n    print("hel'), 'lo")');
  assert.equal(client.completionText(response('```python\nprint("hello")\n```'), '', '    print("hel'), 'lo")');
  assert.equal(client.completionText(response('explanation\n```python\nx\n```'), ''), '');
});

test('strips Markdown backticks around Python fragments without changing literals', () => {
  const response = content => ({ choices: [{ message: { content } }] });
  const fragment = "(['rm', '-f', str(archive)])";
  assert.equal(client.completionText(response('`' + fragment + '`'), '', '        subprocess.call', 'python'), fragment);
  assert.equal(client.completionText(response('``' + fragment + '``'), '', '        subprocess.call', 'python'), fragment);
  assert.equal(client.completionText(response("'hello'"), '', 'value = ', 'python'), "'hello'");
  assert.equal(client.completionText(response('"a ` b"'), '', 'value = ', 'python'), '"a ` b"');
  assert.equal(client.completionText(response('`hello`'), '', 'value = "', 'python'), '`hello`');
  assert.equal(client.completionText(response('`hello`'), '', 'value = """', 'python'), '`hello`');
  assert.equal(client.completionText(response('`hello`'), '', '# explain ', 'python'), '`hello`');
  assert.equal(client.completionText(response('`hello`'), '', 'const value = ', 'javascript'), '`hello`');
  assert.equal(client.completionText(response('`hello`'), '', 'echo ', 'shellscript'), '`hello`');
  assert.equal(client.completionText(response('call`(42)` extra'), '', '', 'python'), '');
  assert.equal(client.completionText(response('"&#x6C;"'), '', 'value = ', 'python'), '"&#x6C;"');
});

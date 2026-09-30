'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { FileContext, contextByteBudget } = require('../src/file-context');
const { completionPayload } = require('../src/client');

test('captures on opening, refreshes after 30 seconds or a save, and switches files', () => {
  let now = 0; let reads = 0; let text = 'initial whole file';
  const doc = { uri: { toString: () => 'file:///a.py' }, version: 1, languageId: 'python', getText() { reads++; return text; } };
  const cache = new FileContext(() => now);
  assert.equal(cache.capture(doc).content, text);
  text = 'unsaved edit'; doc.version++; now = 10000;
  assert.equal(cache.capture(doc).content, 'initial whole file'); assert.equal(reads, 1);
  now = 30000;
  assert.equal(cache.capture(doc).content, 'unsaved edit'); assert.equal(reads, 2);
  now = 60000; cache.capture(doc); assert.equal(reads, 2); // unchanged files aren't re-read
  text = 'saved edit'; doc.version++;
  assert.equal(cache.capture(doc, {}, true).content, text);
  const other = { ...doc, uri: { toString: () => 'file:///b.py' }, getText: () => 'other file' };
  assert.equal(cache.capture(other).content, 'other file');
});
test('enforces char and byte budgets without re-reading for each cursor change', () => {
  let reads = 0;
  const doc = { uri: { toString: () => 'file:///a' }, version: 1, getText() { reads++; return '😀'.repeat(10); } };
  const cache = new FileContext();
  assert.equal(cache.capture(doc, { maxBytes: 30 }).omitted, true);
  assert.equal(cache.capture(doc, { maxBytes: 100 }).omitted, false);
  assert.equal(reads, 1);
  assert.equal(cache.capture(doc, { maxChars: 5 }).content, undefined);
  assert.equal(cache.capture(doc, { maxChars: 0 }).omitted, true);
  assert.equal(contextByteBudget({ contextTokens: 100 }), 0);
  assert.ok(contextByteBudget({ contextTokens: 32000 }, '😀'.repeat(500)) < contextByteBudget({ contextTokens: 32000 }, 'a'));
});
test('stable file snapshot precedes authoritative live cursor excerpts', () => {
  const snapshot = { content: 'def elsewhere(): return 42\n', language: 'python', version: 7 };
  const first = completionPayload('coder', 'new edit', 'suffix', 'python', 256, snapshot);
  const next = completionPayload('coder', 'new edit more', 'suffix', 'python', 256, snapshot);
  assert.equal(first.cache_prompt, true);
  assert.deepEqual(first.messages.slice(0, 2), next.messages.slice(0, 2));
  assert.equal(JSON.parse(first.messages[1].content).file_snapshot.content, snapshot.content);
  assert.equal(JSON.parse(next.messages.at(-1).content).prefix, 'new edit more');
  assert.equal(completionPayload('coder', '', '', 'python', 256, { omitted: true }).messages.length, 2);
});

'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { LatestQueue } = require('../src/latest-queue');
const tick = () => new Promise(resolve => setImmediate(resolve));

test('a burst runs only the current request and the newest pending request', async () => {
  const queue = new LatestQueue(); const ran = []; let release;
  const current = queue.submit(async () => { ran.push('current'); await new Promise(resolve => release = resolve); return 'old'; });
  const pending = [];
  for (let i = 0; i < 100; i++) pending.push(queue.submit(async () => { ran.push(i); return i; }));
  assert.deepEqual(ran, ['current']);
  assert.deepEqual(await Promise.all(pending.slice(0, -1)), Array(99).fill(undefined));
  release(); await current;
  assert.equal(await pending.at(-1), 99);
  assert.deepEqual(ran, ['current', 99]);
});
test('a display timeout does not free the inference slot', async () => {
  const queue = new LatestQueue(); let release; let secondRan = false; let expired = false;
  const first = queue.submit(() => new Promise(resolve => release = resolve), () => true, 10, () => expired = true);
  const second = queue.submit(async () => { secondRan = true; return 'new'; });
  assert.equal(await first, undefined); assert.equal(expired, true);
  assert.equal(secondRan, false); assert.ok(queue.active);
  release('late'); assert.equal(await second, 'new');
});
test('invalidated or disconnected pending work never runs; current work drains', async () => {
  const queue = new LatestQueue(); let release; let valid = true;
  const first = queue.submit(() => new Promise(resolve => release = resolve));
  const cancelled = queue.submit(() => assert.fail('stale request ran'), () => valid);
  valid = false; release('done'); await first;
  assert.equal(await cancelled, undefined);
  let releaseNext;
  const running = queue.submit(() => new Promise(resolve => releaseNext = resolve));
  const disconnected = queue.submit(() => assert.fail('disconnected request ran'));
  queue.clearPending(); assert.equal(await disconnected, undefined); assert.ok(queue.active);
  releaseNext(); await running; await tick(); assert.equal(queue.active, undefined);
});
test('failed work releases local capacity and leaves errors observable', async () => {
  const queue = new LatestQueue();
  await assert.rejects(queue.submit(async () => { throw new Error('upstream failed'); }), /upstream failed/);
  assert.equal(await queue.submit(async () => 'next'), 'next');
});

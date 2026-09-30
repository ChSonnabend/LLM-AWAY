'use strict';

// One executing job plus one replaceable pending job. Cancelling the editor's
// wait never frees the executing slot: only completion of work does that.
class LatestQueue {
  constructor() { this.active = undefined; this.pending = undefined; }
  submit(work, valid = () => true, timeoutMs = 0, onTimeout = () => {}) {
    this.clearPending();
    return new Promise((resolve, reject) => {
      const entry = { work, valid, timeoutMs, onTimeout, resolve, reject };
      this.pending = entry;
      this.pump();
    });
  }
  clearPending() {
    if (this.pending) { this.pending.resolve(undefined); this.pending = undefined; }
  }
  async pump() {
    if (this.active || !this.pending) return;
    const entry = this.pending;
    this.pending = undefined;
    if (!entry.valid()) { entry.resolve(undefined); this.pump(); return; }
    this.active = entry;
    let timer;
    if (entry.timeoutMs > 0) timer = setTimeout(() => {
      entry.resolve(undefined);
      entry.onTimeout();
    }, entry.timeoutMs);
    try { entry.resolve(await entry.work()); }
    catch (error) { entry.reject(error); }
    finally {
      clearTimeout(timer);
      this.active = undefined;
      this.pump();
    }
  }
}
module.exports = { LatestQueue };

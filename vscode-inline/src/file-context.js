'use strict';

// One active-file snapshot. Refreshing this cache never calls the model.
class FileContext {
  constructor(now = Date.now) { this.now = now; this.snapshot = undefined; }
  clear() { this.snapshot = undefined; }
  capture(document, { refreshMs = 30000, maxChars = 200000, maxBytes = 200000 } = {}, force = false) {
    const key = document.uri.toString();
    const limits = String(maxChars);
    const old = this.snapshot;
    if (!old || old.key !== key || old.limits !== limits ||
        (old.version !== document.version && (force || this.now() - old.capturedAt >= refreshMs))) {
      const text = document.getText();
      this.snapshot = { key, limits, version: document.version, language: document.languageId,
        capturedAt: this.now(), totalChars: text.length,
        bytes: Buffer.byteLength(JSON.stringify(text), 'utf8'),
        content: text.length <= maxChars && maxChars > 0 ? text : undefined };
    }
    const fits = this.snapshot.content !== undefined && this.snapshot.bytes <= maxBytes;
    return { ...this.snapshot, content: fits ? this.snapshot.content : undefined, omitted: !fits };

  }
}
function contextByteBudget(session, prefix = '', suffix = '', maxTokens = 512) {
  // Count JSON escaping/UTF-8 for the live excerpts, plus generous instruction
  // and output reserves. This byte-based estimate is deliberately conservative.
  const tokens = Number(session?.contextTokens) || 32768;
  const reserved = Buffer.byteLength(JSON.stringify({ prefix, suffix }), 'utf8') + 4096 + maxTokens;
  return Math.max(0, Math.floor(tokens) - reserved);
}
module.exports = { FileContext, contextByteBudget };

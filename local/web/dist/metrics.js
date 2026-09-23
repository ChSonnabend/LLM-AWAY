const cards = document.getElementById('metric-cards');
const updated = document.getElementById('metrics-updated');
const history = new Map();
let selectedSession = '';
const pageParameters = new URLSearchParams(window.location.search);
let requestedSession = pageParameters.get('session') || '';
if (pageParameters.get('embed') === '1') document.body.classList.add('embedded');

// Matplotlib/ColorBrewer-compatible anchors, sampled away from both endpoints.
const YL_OR_RD = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026'];
const GN_BU = ['#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe', '#0868ac', '#084081'];

function csvFields(line) {
  const fields = [];
  let value = '', quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') { value += '"'; index += 1; }
      else quoted = !quoted;
    } else if (char === ',' && !quoted) { fields.push(value.trim()); value = ''; }
    else value += char;
  }
  fields.push(value.trim());
  return fields;
}

function samplesFrom(row) {
  const lines = row.gpu_lines || [];
  if (!lines.length || !lines[0].toLowerCase().startsWith('index')) return [];
  return lines.slice(1).map(line => {
    const values = csvFields(line);
    if (values.length < 6) return null;
    return {index: values[0], name: values[2], used: parseFloat(values[3]) || 0, total: parseFloat(values[4]) || 0, utilization: parseFloat(values[5]) || 0};
  }).filter(Boolean);
}

function record(row) {
  const timestamp = Number(row.gpu_timestamp) || Date.now() / 1000;
  const session = String(row.id);
  if (!history.has(session)) history.set(session, new Map());
  for (const gpu of samplesFrom(row)) {
    const key = String(gpu.index);
    if (!history.get(session).has(key)) history.get(session).set(key, []);
    const points = history.get(session).get(key);
    if (!points.length || points[points.length - 1].time !== timestamp) points.push({...gpu, time: timestamp});
    if (points.length > 360) points.splice(0, points.length - 360);
  }
}

function interpolateColor(palette, position) {
  const scaled = Math.max(0, Math.min(1, position)) * (palette.length - 1);
  const lower = Math.floor(scaled), upper = Math.min(palette.length - 1, lower + 1), mix = scaled - lower;
  const channels = color => [1, 3, 5].map(offset => parseInt(color.slice(offset, offset + 2), 16));
  const a = channels(palette[lower]), b = channels(palette[upper]);
  return `#${a.map((value, index) => Math.round(value + (b[index] - value) * mix).toString(16).padStart(2, '0')).join('')}`;
}

function gpuColors(count) {
  return Array.from({length: count}, (_, index) => {
    const position = (index + 1) / (count + 1);
    return {utilization: interpolateColor(YL_OR_RD, position), vram: interpolateColor(GN_BU, position)};
  });
}

function draw(canvas, series) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
  const left = 42, right = 14, top = 16, bottom = 27, plotWidth = width - left - right, plotHeight = height - top - bottom;
  ctx.clearRect(0, 0, width, height); ctx.font = '10px ui-monospace, monospace'; ctx.fillStyle = '#778398'; ctx.strokeStyle = '#252d3a'; ctx.lineWidth = 1;
  for (let value = 0; value <= 100; value += 25) {
    const y = top + plotHeight * (1 - value / 100); ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke(); ctx.fillText(`${value}%`, 4, y + 3);
  }
  const allPoints = series.flatMap(item => item.points);
  if (!allPoints.length) return;
  const first = Math.min(...allPoints.map(point => point.time));
  const last = Math.max(Math.max(...allPoints.map(point => point.time)), first + 1);
  const plot = (points, color, value) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    points.forEach((point, index) => {
      const x = left + ((point.time - first) / (last - first)) * plotWidth;
      const y = top + (1 - Math.max(0, Math.min(100, value(point))) / 100) * plotHeight;
      if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
  };
  for (const item of series) {
    plot(item.points, item.colors.vram, point => point.total ? point.used / point.total * 100 : 0);
    plot(item.points, item.colors.utilization, point => point.utilization);
  }
  ctx.fillStyle = '#778398'; ctx.fillText(new Date(first * 1000).toLocaleTimeString(), left, height - 8);
  const end = new Date(last * 1000).toLocaleTimeString(); ctx.fillText(end, width - right - ctx.measureText(end).width, height - 8);
}

function render() {
  cards.replaceChildren();
  const gpus = history.get(selectedSession);
  if (!gpus || !gpus.size) {
    const empty = document.createElement('div'); empty.className = 'panel metrics-empty';
    empty.textContent = selectedSession ? 'No GPU telemetry has arrived for this session yet.' : 'No active sessions. GPU history will appear when an allocation starts.';
    cards.append(empty); reportHeight(); return;
  }
  const entries = [...gpus.entries()].sort(([a], [b]) => Number(a) - Number(b));
  const colors = gpuColors(entries.length);
  const series = entries.map(([index, points], position) => ({index, points, colors: colors[position]}));
  const card = document.createElement('article'); card.className = 'panel metric-card combined-metric-card';
  const heading = document.createElement('div'); heading.className = 'metric-heading combined-metric-heading';
  const title = document.createElement('div'); title.innerHTML = `<span class="kicker">SESSION ${selectedSession}</span><h2>All GPUs</h2>`;
  const legend = document.createElement('div'); legend.className = 'gpu-series-legend';
  for (const item of series) {
    const latest = item.points[item.points.length - 1];
    const row = document.createElement('div'); row.className = 'gpu-series-row';
    const label = document.createElement('strong'); label.textContent = `GPU ${item.index}`;
    const name = document.createElement('span'); name.className = 'gpu-series-name'; name.textContent = latest.name || 'GPU';
    const vram = document.createElement('span'); vram.className = 'gpu-series-value'; vram.innerHTML = `<i style="background:${item.colors.vram}"></i>VRAM ${(latest.used / 1024).toFixed(1)} / ${(latest.total / 1024).toFixed(1)} GiB`;
    const utilization = document.createElement('span'); utilization.className = 'gpu-series-value'; utilization.innerHTML = `<i style="background:${item.colors.utilization}"></i>Usage ${latest.utilization.toFixed(0)}%`;
    row.append(label, name, vram, utilization); legend.append(row);
  }
  heading.append(title, legend);
  const canvas = document.createElement('canvas'); canvas.className = 'metric-chart combined-metric-chart';
  card.append(heading, canvas); cards.append(card); draw(canvas, series); reportHeight();
}

function reportHeight() {
  if (document.body.classList.contains('embedded')) window.parent.postMessage({type: 'metrics-height', height: document.documentElement.scrollHeight}, window.location.origin);
}

async function refresh() {
  try {
    const response = await fetch('/api/sessions', {cache: 'no-store'}); const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    data.sessions.forEach(record);
    const preferred = requestedSession || selectedSession;
    const firstWithTelemetry = data.sessions.find(row => samplesFrom(row).length);
    selectedSession = data.sessions.some(row => String(row.id) === preferred) ? preferred : String(firstWithTelemetry?.id || data.sessions[0]?.id || '');
    updated.textContent = `Live · ${new Date(data.time * 1000).toLocaleTimeString()}`; render();
  } catch (error) { updated.textContent = error.message; }
}

window.addEventListener('message', event => {
  if (event.origin !== window.location.origin || event.data?.type !== 'select-session') return;
  requestedSession = String(event.data.session || ''); selectedSession = requestedSession; render();
});
window.addEventListener('resize', render);
refresh(); setInterval(refresh, 5000);

// Each browser tab owns a lease; the server exits after the last tab closes.
const browserLease = crypto.randomUUID();
function pingBrowser() {
  fetch('/api/browser/ping', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id: browserLease})}).catch(() => {});
}
pingBrowser();
setInterval(pingBrowser, 15000);
window.addEventListener('pageshow', pingBrowser);
window.addEventListener('pagehide', () => navigator.sendBeacon('/api/browser/close', JSON.stringify({id: browserLease})));

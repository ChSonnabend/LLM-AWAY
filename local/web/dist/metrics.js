const sessionSelect = document.getElementById('metrics-session');
const cards = document.getElementById('metric-cards');
const updated = document.getElementById('metrics-updated');
const history = new Map();
let selectedSession = '';
let requestedSession = '';
if (new URLSearchParams(window.location.search).get('embed') === '1') document.body.classList.add('embedded');

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

function draw(canvas, points) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
  const left = 42, right = 14, top = 16, bottom = 27, plotWidth = width - left - right, plotHeight = height - top - bottom;
  ctx.clearRect(0, 0, width, height); ctx.font = '10px ui-monospace, monospace'; ctx.fillStyle = '#778398'; ctx.strokeStyle = '#252d3a'; ctx.lineWidth = 1;
  for (let value = 0; value <= 100; value += 25) {
    const y = top + plotHeight * (1 - value / 100); ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke(); ctx.fillText(`${value}%`, 4, y + 3);
  }
  if (!points.length) return;
  const first = points[0].time, last = Math.max(points[points.length - 1].time, first + 1);
  const plot = (color, value) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    points.forEach((point, index) => { const x = left + ((point.time - first) / (last - first)) * plotWidth; const y = top + (1 - Math.max(0, Math.min(100, value(point))) / 100) * plotHeight; if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.stroke();
  };
  plot('#79e6c5', point => point.total ? point.used / point.total * 100 : 0);
  plot('#f4bd67', point => point.utilization);
  ctx.fillStyle = '#778398'; ctx.fillText(new Date(first * 1000).toLocaleTimeString(), left, height - 8); const end = new Date(last * 1000).toLocaleTimeString(); ctx.fillText(end, width - right - ctx.measureText(end).width, height - 8);
}

function render() {
  cards.replaceChildren();
  const gpus = history.get(selectedSession);
  if (!gpus || !gpus.size) { const empty = document.createElement('div'); empty.className = 'panel metrics-empty'; empty.textContent = selectedSession ? 'No GPU telemetry has arrived for this session yet.' : 'No active sessions. GPU history will appear when an allocation starts.'; cards.append(empty); reportHeight(); return; }
  for (const [index, points] of gpus) {
    const latest = points[points.length - 1];
    const card = document.createElement('article'); card.className = 'panel metric-card';
    const heading = document.createElement('div'); heading.className = 'metric-heading';
    const title = document.createElement('div'); title.innerHTML = `<span class="kicker">GPU ${index}</span><h2></h2>`; title.querySelector('h2').textContent = latest.name || 'GPU';
    const values = document.createElement('div'); values.className = 'metric-values'; values.textContent = `${(latest.used / 1024).toFixed(1)} / ${(latest.total / 1024).toFixed(1)} GiB · ${latest.utilization.toFixed(0)}%`;
    heading.append(title, values); const canvas = document.createElement('canvas'); canvas.className = 'metric-chart'; card.append(heading, canvas); cards.append(card); draw(canvas, points);
  }
  reportHeight();
}

function reportHeight() {
  if (document.body.classList.contains('embedded')) window.parent.postMessage({type: 'metrics-height', height: document.documentElement.scrollHeight}, window.location.origin);
}

async function refresh() {
  try {
    const response = await fetch('/api/sessions', {cache: 'no-store'}); const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    data.sessions.forEach(record);
    const previous = selectedSession; sessionSelect.replaceChildren();
    data.sessions.forEach(row => { const option = document.createElement('option'); option.value = row.id; option.textContent = `Session ${row.id} · ${row.model || row.phase || 'no model'}`; sessionSelect.append(option); });
    const preferred = requestedSession || previous;
    selectedSession = data.sessions.some(row => String(row.id) === preferred) ? preferred : String(data.sessions[0]?.id || ''); sessionSelect.value = selectedSession;
    sessionSelect.disabled = !data.sessions.length;
    updated.textContent = `Live · ${new Date(data.time * 1000).toLocaleTimeString()}`; render();
  } catch (error) { updated.textContent = error.message; }
}

sessionSelect.addEventListener('change', () => { selectedSession = sessionSelect.value; render(); });
window.addEventListener('message', event => {
  if (event.origin !== window.location.origin || event.data?.type !== 'select-session') return;
  requestedSession = String(event.data.session || '');
  if ([...sessionSelect.options].some(option => option.value === requestedSession)) {
    selectedSession = requestedSession;
    sessionSelect.value = selectedSession;
    render();
  }
});
window.addEventListener('resize', render);
refresh(); setInterval(refresh, 5000);

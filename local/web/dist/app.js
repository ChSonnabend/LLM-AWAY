const $ = id => document.getElementById(id);

let selected = null;
let logName = 'session';
let terminal = null;
let terminalId = null;
let terminalOffset = 0;
let terminalPoll = null;
const chatTerminals = new Map();
let sessionsBusy = false;
let logBusy = false;
let dialogToken = 0;

async function api(path, options = {}) {
  const response = await fetch(path, {headers: {'Content-Type': 'application/json'}, ...options});
  let body;
  try { body = await response.json(); }
  catch { throw new Error(`Request failed (${response.status})`); }
  if (!response.ok) throw new Error(body.error || 'Request failed');
  return body;
}

function notice(message, error = false) {
  const node = $('notice');
  node.textContent = message;
  node.className = error ? 'error' : '';
  window.setTimeout(() => { if (node.textContent === message) node.textContent = ''; }, 5500);
}

function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}

function openDialog(title, content, kind = '') {
  dialogToken += 1;
  $('modal-title').textContent = title;
  $('modal-body').replaceChildren(content);
  document.querySelector('.dialog').className = `dialog ${kind}`.trim();
  $('modal').hidden = false;
  return dialogToken;
}

function closeDialog() {
  dialogToken += 1;
  const host = $('terminal-host');
  if (host.parentElement !== $('terminal-slot')) {
    $('terminal-slot').append(host);
    host.hidden = true;
  }
  $('modal').hidden = true;
  $('modal-body').replaceChildren();
  document.querySelector('.dialog').className = 'dialog';
}

function menuOption(label, description, onClick, danger = false) {
  const button = node('button', `menu-option${danger ? ' danger' : ''}`);
  const copy = node('span');
  copy.append(node('strong', '', label), node('small', '', description));
  button.append(copy, node('span', 'chevron', '›'));
  button.addEventListener('click', onClick);
  return button;
}

function menu(title, options) {
  const content = node('div', 'menu-options');
  options.forEach(item => content.append(menuOption(item.label, item.description, item.run, item.danger)));
  openDialog(title, content);
}

function operationView(title, message) {
  const view = node('div', 'operation');
  const content = node('div', 'operation-content');
  content.append(node('div', 'spinner'), node('div', 'result-icon', '✓'), node('h3', '', title), node('p', '', message));
  view.append(content);
  return view;
}

async function pollJob(identifier) {
  for (;;) {
    const status = await api(`/api/job?id=${encodeURIComponent(identifier)}`);
    if (status.state !== 'running') return status;
    await new Promise(resolve => window.setTimeout(resolve, 450));
  }
}

async function runOperation(title, description, endpoint, payload, closeOnSuccess = false) {
  const view = operationView(title, description);
  const token = openDialog(title, view);
  try {
    const started = await api(endpoint, {method: 'POST', body: JSON.stringify(payload)});
    const result = await pollJob(started.job);
    if (token !== dialogToken) {
      notice(result.message, result.state === 'error');
      await loadSessions();
      return;
    }
    view.classList.add(result.state === 'done' ? 'success' : 'error');
    view.querySelector('.result-icon').textContent = result.state === 'done' ? '✓' : '!';
    view.querySelector('h3').textContent = result.state === 'done' ? 'Completed' : 'Could not complete';
    view.querySelector('p').textContent = result.message;
    await loadSessions();
    if (result.state === 'done' && closeOnSuccess) {
      window.setTimeout(() => { if (token === dialogToken) closeDialog(); }, 500);
    }
  } catch (error) {
    view.classList.add('error');
    view.querySelector('.result-icon').textContent = '!';
    view.querySelector('h3').textContent = 'Could not complete';
    view.querySelector('p').textContent = error.message;
  }
}

function renderDetails(row) {
  const list = $('detail');
  list.replaceChildren();
  for (const line of row.details || []) {
    const split = line.indexOf(':');
    const item = node('div');
    item.append(node('dt', '', split < 0 ? 'Info' : line.slice(0, split)), node('dd', '', split < 0 ? line : line.slice(split + 1).trim()));
    list.append(item);
  }
}

function renderChats(rows) {
  $('chat-count').textContent = rows.length;
  const cards = $('chat-cards');
  const active = new Set(rows.map(row => String(row.id)));
  for (const card of [...cards.children]) if (!active.has(card.dataset.session)) card.remove();
  for (const row of rows) {
    const key = String(row.id);
    const ready = Boolean(row.terminal) && ['LOADED', 'READY', 'RUNNING'].includes(String(row.model_state || '').toUpperCase());
    let card = cards.querySelector(`[data-session="${key}"]`);
    if (!card) {
      card = node('article', 'panel chat-card'); card.dataset.session = key;
      const heading = node('header', 'chat-heading');
      const title = node('div'); title.append(node('span', 'kicker', `SESSION ${row.id}`), node('h2'));
      heading.append(title, node('span', 'chat-status'));
      const screen = node('div', 'chat-terminal'); screen.dataset.session = key;
      card.append(heading, screen); cards.append(card);
    }
    card.querySelector('h2').textContent = row.model || 'No model selected';
    const status = card.querySelector('.chat-status'); status.className = `chat-status ${ready ? 'ready' : ''}`; status.textContent = ready ? 'Terminal' : row.model_state || row.phase || 'Unavailable';
    const screen = card.querySelector('.chat-terminal');
    if (!ready && !chatTerminals.has(key)) screen.textContent = 'Load the model and attach its agent to open this terminal.';
    if (ready) ensureChatTerminal(row, screen);
  }
}

async function ensureChatTerminal(row, host) {
  const key = String(row.id);
  if (chatTerminals.has(key) || !window.Terminal) return;
  host.replaceChildren();
  const xterm = new Terminal({cursorBlink: true, convertEol: true, fontSize: 13, fontFamily: 'SFMono-Regular, Menlo, monospace', scrollback: 10000, theme: {background: '#080a0e', foreground: '#dce4ef', cursor: '#79e6c5', selectionBackground: '#244c43'}});
  xterm.open(host); xterm.focus();
  const item = {terminal: xterm, id: null, offset: 0, poll: null}; chatTerminals.set(key, item);
  try {
    const created = await api('/api/terminal', {method: 'POST', body: '{}'}); item.id = created.id;
    xterm.onData(data => api('/api/terminal/input', {method: 'POST', body: JSON.stringify({id: item.id, data})}).catch(error => notice(error.message, true)));
    const resize = () => { const cols = Math.max(50, Math.floor(host.clientWidth / 8.1)); const rows = 28; xterm.resize(cols, rows); api('/api/terminal/resize', {method: 'POST', body: JSON.stringify({id: item.id, cols, rows})}).catch(() => {}); };
    item.resize = resize;
    resize();
    await api('/api/terminal/input', {method: 'POST', body: JSON.stringify({id: item.id, data: `run --session ${Number(row.id)} --resume\n`})});
    item.poll = window.setInterval(async () => { try { const data = await api(`/api/terminal/output?id=${encodeURIComponent(item.id)}&offset=${item.offset}`); item.offset = data.offset; if (data.data) { xterm.write(data.data); xterm.scrollToBottom(); } } catch {} }, 120);
  } catch (error) { chatTerminals.delete(key); xterm.write(`\r\nCould not open terminal: ${error.message}\r\n`); }
}

function connectReadyChats() {
  for (const item of chatTerminals.values()) item.resize?.();
  for (const card of $('chat-cards').children) {
    if (card.querySelector('.chat-status')?.classList.contains('ready')) ensureChatTerminal({id: card.dataset.session}, card.querySelector('.chat-terminal'));
  }
}

function selectSession(row, refreshLog = true) {
  selected = row;
  document.querySelectorAll('#sessions tr').forEach(item => item.classList.toggle('selected', Number(item.dataset.id) === Number(row.id)));
  $('selected-name').textContent = `Session ${row.id}`;
  renderDetails(row);
  document.querySelectorAll('.session-action').forEach(button => button.disabled = false);
  $('gpu-overview').contentWindow?.postMessage({type: 'select-session', session: String(row.id)}, window.location.origin);
  if (refreshLog && $('modal').hidden && $('terminal-host').hidden) loadLog();
}

$('gpu-overview').addEventListener('load', () => {
  if (selected) $('gpu-overview').contentWindow?.postMessage({type: 'select-session', session: String(selected.id)}, window.location.origin);
});
window.addEventListener('message', event => {
  if (event.origin !== window.location.origin || event.data?.type !== 'metrics-height') return;
  $('gpu-overview').style.height = `${Math.max(360, Math.min(1600, Number(event.data.height) || 410))}px`;
});

function openSession(row) {
  selectSession(row, false);
  const state = String(row.model_state || '').toUpperCase();
  const loaded = Boolean(row.model) && ['LOADED', 'READY', 'RUNNING'].includes(state);
  if (loaded) {
    terminalWindow(`Session ${row.id}`, `run --session ${row.id} --resume\n`);
  } else {
    modelDialog();
  }
}

async function loadSessions() {
  if (sessionsBusy) return;
  sessionsBusy = true;
  try {
    const data = await api('/api/sessions');
    $('updated').textContent = `Live · ${new Date(data.time * 1000).toLocaleTimeString()}`;
    $('allocation-count').textContent = data.sessions.length;
    renderChats(data.sessions);
    const body = $('sessions');
    body.replaceChildren();
    if (!data.sessions.length) {
      const row = node('tr');
      const cell = node('td', 'empty', 'No active allocations');
      cell.colSpan = 6;
      row.append(cell);
      body.append(row);
      selected = null;
      $('selected-name').textContent = 'None';
      document.querySelectorAll('.session-action').forEach(button => button.disabled = true);
      return;
    }
    for (const row of data.sessions) {
      const tr = node('tr');
      tr.dataset.id = row.id;
      const values = [row.id, row.model_state || row.phase || '—', row.model || 'No model', row.node || row.host || '—', row.gpus || 0, row.job_id || '—'];
      values.forEach(value => tr.append(node('td', '', String(value))));
      tr.title = row.model && ['LOADED', 'READY', 'RUNNING'].includes(String(row.model_state || '').toUpperCase())
        ? 'Double-click to attach or choose a conversation'
        : 'Double-click to load a model';
      tr.addEventListener('click', () => selectSession(row));
      tr.addEventListener('dblclick', () => openSession(row));
      body.append(tr);
    }
    const current = data.sessions.find(row => Number(row.id) === Number(selected?.id)) || data.sessions[0];
    selectSession(current, false);
  } catch (error) {
    $('updated').textContent = 'Disconnected';
    notice(error.message, true);
  } finally { sessionsBusy = false; }
}

async function loadLog() {
  if (!selected || logBusy || $('log-output').hidden || !$('chats-view').hidden) return;
  const sessionId = selected.id;
  const requestedLog = logName;
  logBusy = true;
  try {
    const data = await api(`/api/log?session=${sessionId}&name=${requestedLog}`);
    if (Number(selected?.id) !== Number(sessionId) || logName !== requestedLog) return;
    const label = requestedLog === 'session' ? 'Telemetry' : requestedLog[0].toUpperCase() + requestedLog.slice(1);
    $('output-title').textContent = `${label} log`;
    $('log-path').textContent = data.path;
    $('log-output').textContent = data.content || 'No output yet.';
    $('log-output').scrollTop = $('log-output').scrollHeight;
  } catch (error) { notice(error.message, true); }
  finally { logBusy = false; }
}

function actionDialog(button) {
  const label = button.textContent.trim();
  const danger = button.dataset.danger === 'true';
  menu(label, [{
    label: `${danger ? 'Confirm' : 'Run'} ${label}`,
    description: button.dataset.description || 'Run this resource monitor action.',
    danger,
    run: () => runOperation(label, 'The operation is running. This window will update when it finishes.', '/api/actions', {action: button.dataset.action, session: selected?.id, confirmed: true})
  }]);
}

function field(label, control, help = '', full = false) {
  const wrapper = node('div', `field${full ? ' full' : ''}`);
  wrapper.append(node('label', '', label), control);
  if (help) wrapper.append(node('small', '', help));
  return wrapper;
}

function selectControl(options, value = '') {
  const control = document.createElement('select');
  options.forEach(option => {
    const item = document.createElement('option');
    item.value = typeof option === 'string' ? option : option.value;
    item.textContent = typeof option === 'string' ? option : option.label;
    control.append(item);
  });
  control.value = value;
  return control;
}

function inputControl(value = '', type = 'text') {
  const control = document.createElement('input');
  control.type = type;
  control.value = value;
  return control;
}

async function allocationDialog() {
  const loading = operationView('Allocate resources', 'Loading saved hosts and allocation defaults…');
  openDialog('Allocate resources', loading);
  try {
    const options = await api('/api/options');
    const form = node('form', 'form');
    const grid = node('div', 'form-grid');
    const connection = selectControl(['local', 'ssh'], options.default_connection);
    const type = selectControl([{value: 'custom', label: 'Custom model'}, {value: 'native', label: 'Native CLI'}], 'custom');
    const host = selectControl(options.hosts.length ? options.hosts : [{value: '', label: 'No saved SSH hosts'}], options.default_host);
    const gpus = inputControl(String(options.default_gpus), 'number'); gpus.min = '0'; gpus.step = '1';
    const slurm = inputControl(options.slurm_options || '');
    const cli = selectControl(['codex', 'claude'], options.available_clis.includes('codex') ? 'codex' : 'claude');
    const hostField = field('SSH host', host, options.hosts.length ? 'Saved host profile' : 'Configure an SSH host first.');
    const gpuField = field('GPUs', gpus, `Backend: ${options.backend}`);
    const slurmField = field('Additional scheduler options', slurm, 'Shell-style Slurm options for this allocation.', true);
    const cliField = field('Native CLI', cli, 'Uses its own account and model.');
    grid.append(field('Location', connection), field('Session type', type), hostField, gpuField, slurmField, cliField);
    const actions = node('div', 'form-actions');
    const configure = node('button', '', 'Configure hosts…'); configure.type = 'button';
    configure.addEventListener('click', () => terminalWindow('Host configuration', 'res-alloc --restart\n'));
    const submit = node('button', 'primary', 'Allocate'); submit.type = 'submit';
    actions.append(configure, submit); form.append(grid, actions);
    function update() {
      const remote = connection.value === 'ssh';
      const native = type.value === 'native';
      hostField.hidden = !remote;
      gpuField.hidden = native;
      slurmField.hidden = native;
      cliField.hidden = !native;
      submit.disabled = remote && !host.value;
    }
    connection.addEventListener('change', update); type.addEventListener('change', update); host.addEventListener('change', update); update();
    form.addEventListener('submit', event => {
      event.preventDefault();
      const settings = {connection: connection.value, native: type.value === 'native', host: host.value, gpus: gpus.value, slurm_options: slurm.value, cli: cli.value};
      runOperation('Allocate resources', 'Creating the session and starting its resource monitor…', '/api/allocate', {settings}, true);
    });
    openDialog('Allocate resources', form);
  } catch (error) {
    loading.classList.add('error');
    loading.querySelector('.result-icon').textContent = '!';
    loading.querySelector('h3').textContent = 'Could not load allocation options';
    loading.querySelector('p').textContent = error.message;
  }
}

async function modelDialog() {
  const sessionId = selected.id;
  const loading = operationView(`Session ${sessionId}`, 'Discovering installed models…');
  openDialog(`Session ${sessionId}`, loading);
  try {
    const data = await api(`/api/models?session=${sessionId}`);
    if (data.native) {
      menu(`Session ${sessionId}`, [{label: 'Attach native agent', description: 'Open the retained native CLI terminal.', run: () => terminalWindow(`Session ${sessionId}`, `run --session ${sessionId} --resume\n`)}]);
      return;
    }
    const form = node('form', 'form');
    const grid = node('div', 'form-grid');
    const modelOptions = data.models.map(model => ({value: model.name, label: `${model.alias || model.name} · ${(model.size_bytes / 1073741824).toFixed(1)} GiB`}));
    const model = selectControl(modelOptions, data.models.find(item => item.alias === data.current || item.name === data.current)?.name || data.models[0]?.name || '');
    const mtp = selectControl(['auto', 'on', 'off'], 'auto');
    const location = selectControl(['local', 'remote'], 'local');
    const cli = selectControl(['auto', 'codex', 'claude'], 'auto');
    const workdir = inputControl(''); workdir.placeholder = 'Default project directory';
    const server = inputControl(data.server_options_by_model?.[model.value] || '');
    grid.append(field('Model', model, '', true), field('MTP', mtp), field('Agent location', location), field('Agent CLI', cli), field('Agent work directory', workdir), field('Model server options', server, 'Batch, context and backend arguments.', true));
    model.addEventListener('change', () => { server.value = data.server_options_by_model?.[model.value] || ''; });
    const actions = node('div', 'form-actions');
    const attach = node('button', '', 'Attach existing terminal'); attach.type = 'button'; attach.addEventListener('click', () => terminalWindow(`Session ${sessionId}`, `run --session ${sessionId} --resume\n`));
    const submit = node('button', 'primary', 'Load and start'); submit.type = 'submit';
    actions.append(attach, submit); form.append(grid, actions);
    form.addEventListener('submit', event => {
      event.preventDefault();
      runOperation(`Start session ${sessionId}`, 'Loading the model and preparing the retained agent terminal…', '/api/load', {session: sessionId, settings: {model: model.value, mtp: mtp.value, agent_location: location.value, cli: cli.value, agent_workdir: workdir.value, server_options: server.value}}, true);
    });
    openDialog(`Attach session ${sessionId}`, form);
  } catch (error) {
    loading.classList.add('error');
    loading.querySelector('.result-icon').textContent = '!';
    loading.querySelector('h3').textContent = 'Could not discover models';
    loading.querySelector('p').textContent = error.message;
  }
}

function releaseDialog() {
  const sessionId = selected.id;
  menu(`Release session ${sessionId}`, [
    {label: 'Unload model', description: 'Stop the model and keep the resource allocation.', run: () => runOperation('Unload model', 'Stopping the model while retaining the allocation…', '/api/actions', {action: 'unload', session: sessionId, confirmed: true})},
    {label: 'Load a different model', description: 'Unload as needed and open browser model selection.', run: modelDialog},
    {label: 'Release allocation', description: 'Stop the model, end the agent and return all resources.', danger: true, run: () => runOperation('Release allocation', 'Stopping session processes and releasing resources…', '/api/actions', {action: 'release', session: sessionId, confirmed: true}, true)}
  ]);
}

async function ensureTerminal() {
  if (terminalId) return;
  if (!window.Terminal) throw new Error('The terminal renderer could not load. Check the network connection and reload the page.');
  const created = await api('/api/terminal', {method: 'POST', body: '{}'});
  terminalId = created.id;
  terminalOffset = 0;
  terminal = new Terminal({cursorBlink: true, convertEol: true, fontSize: 13, fontFamily: 'SFMono-Regular, Menlo, monospace', theme: {background: '#080a0e', foreground: '#dce4ef', cursor: '#79e6c5', selectionBackground: '#244c43'}});
  terminal.open($('terminal-host'));
  terminal.onData(data => api('/api/terminal/input', {method: 'POST', body: JSON.stringify({id: terminalId, data})}).catch(error => notice(error.message, true)));
  resizeTerminal();
  terminalPoll = window.setInterval(readTerminal, 120);
}

function resizeTerminal() {
  if (!terminal || $('terminal-host').hidden) return;
  const host = $('terminal-host');
  const cols = Math.max(50, Math.floor((host.clientWidth - 20) / 8));
  const rows = Math.max(12, Math.floor((host.clientHeight - 20) / 17));
  terminal.resize(cols, rows);
  api('/api/terminal/resize', {method: 'POST', body: JSON.stringify({id: terminalId, cols, rows})}).catch(() => {});
}

async function readTerminal() {
  if (!terminalId) return;
  try {
    const data = await api(`/api/terminal/output?id=${encodeURIComponent(terminalId)}&offset=${terminalOffset}`);
    terminalOffset = data.offset;
    if (data.data) terminal.write(data.data);
  } catch (error) { notice(error.message, true); }
}

async function terminalWindow(title, command = '') {
  const shell = node('div', 'terminal-shell');
  const host = $('terminal-host');
  shell.append(host);
  host.hidden = false;
  openDialog(title, shell, 'terminal-dialog');
  try {
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    await ensureTerminal();
    resizeTerminal();
    terminal.focus();
    if (command) await api('/api/terminal/input', {method: 'POST', body: JSON.stringify({id: terminalId, data: command})});
  } catch (error) {
    shell.replaceChildren(operationView('Terminal unavailable', error.message));
  }
}

function showPanelTerminal() {
  closeDialog();
  $('terminal-slot').append($('terminal-host'));
  $('terminal-host').hidden = false;
  $('log-path').hidden = true;
  $('log-output').hidden = true;
  $('output-title').textContent = 'Integrated terminal';
  document.querySelectorAll('.tab').forEach(tab => tab.classList.toggle('active', tab.dataset.view === 'terminal'));
  ensureTerminal().then(() => { resizeTerminal(); terminal.focus(); }).catch(error => notice(error.message, true));
}

function showLog(name, tab) {
  logName = name;
  $('terminal-host').hidden = true;
  $('log-path').hidden = false;
  $('log-output').hidden = false;
  document.querySelectorAll('.tab').forEach(item => item.classList.toggle('active', item === tab));
  $('output-title').textContent = `${tab.dataset.label || name} log`;
  loadLog();
}

document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', () => actionDialog(button)));
document.querySelectorAll('[data-page]').forEach(button => button.addEventListener('click', () => {
  const monitoring = button.dataset.page === 'monitoring';
  document.querySelectorAll('.view-tab').forEach(tab => tab.classList.toggle('active', tab === button));
  document.querySelectorAll('.monitoring-view').forEach(item => item.hidden = !monitoring);
  $('chats-view').hidden = monitoring;
  if (!monitoring) connectReadyChats();
}));
document.querySelectorAll('[data-log]').forEach(button => button.addEventListener('click', () => showLog(button.dataset.log, button)));
document.querySelector('[data-view="terminal"]').addEventListener('click', showPanelTerminal);
$('allocate').addEventListener('click', allocationDialog);
$('attach-menu').addEventListener('click', modelDialog);
$('release-menu').addEventListener('click', releaseDialog);
$('open-terminal').addEventListener('click', () => terminalWindow('Integrated terminal'));
$('modal-close').addEventListener('click', closeDialog);
$('modal').addEventListener('click', event => { if (event.target === $('modal')) closeDialog(); });
window.addEventListener('keydown', event => { if (event.key === 'Escape' && !$('modal').hidden) closeDialog(); });
window.addEventListener('resize', resizeTerminal);
window.addEventListener('beforeunload', () => {
  if (terminalId) fetch(`/api/terminal?id=${encodeURIComponent(terminalId)}`, {method: 'DELETE', keepalive: true});
  for (const item of chatTerminals.values()) if (item.id) fetch(`/api/terminal?id=${encodeURIComponent(item.id)}`, {method: 'DELETE', keepalive: true});
});
window.addEventListener('unhandledrejection', event => notice(event.reason?.message || String(event.reason), true));
window.addEventListener('error', event => notice(event.message || 'Unexpected browser error', true));

loadSessions();
window.setInterval(loadSessions, 4000);
window.setInterval(loadLog, 1000);

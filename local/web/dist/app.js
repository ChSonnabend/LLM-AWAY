const $ = id => document.getElementById(id);

let selected = null;
let logName = 'session';
let terminal = null;
let terminalId = null;
let terminalOffset = 0;
let terminalPoll = null;
const chatTerminals = new Map();
const pendingChats = new Set();
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
    if (result.state === 'done' && endpoint === '/api/load') {
      pendingChats.add(String(payload.session));
      notice(result.message, false);
      if (token === dialogToken) closeDialog();
      await loadSessions();
    } else if (result.state === 'done' && closeOnSuccess) {
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
    const nativeReady = Boolean(row.phase && String(row.phase).toUpperCase() === 'RUNNING');
    const ready = Boolean(row.attached && row.terminal) && ((row.native && nativeReady) || ['LOADED', 'READY', 'RUNNING'].includes(String(row.model_state || '').toUpperCase()));
    let card = cards.querySelector(`[data-session="${key}"]`);
    if (!card) {
      card = node('article', 'panel chat-card'); card.dataset.session = key;
      const heading = node('header', 'chat-heading');
      const title = node('div'); title.append(node('span', 'kicker', `SESSION ${row.id}`), node('h2'));
      heading.append(title, node('span', 'chat-status'));
      const screen = node('div', 'chat-terminal'); screen.dataset.session = key;
      card.append(heading, screen); cards.append(card); makeResizable(card,screen,`chat-${key}`);
    }
    card.querySelector('h2').textContent = row.model || (nativeReady ? `Native ${row.native_cli || 'CLI'}` : 'No model selected');
    const status = card.querySelector('.chat-status'); status.className = `chat-status ${ready ? 'ready' : ''}`; status.textContent = ready ? `Terminal · ${row.time_left || '∞'}` : `${row.model_state || row.phase || 'Unavailable'} · ${row.time_left || '∞'}`;
    const screen = card.querySelector('.chat-terminal');
    if (!ready && chatTerminals.has(key)) closeChatTerminal(key, screen);
    else if (!ready) screen.textContent = 'Load the model and attach its agent to open this terminal.';
    if (pendingChats.has(key) && ready) {
      pendingChats.delete(key);
      document.querySelector('[data-page="chats"]')?.click();
    } else if (pendingChats.has(key) && ['ERROR','EXITED','FAILED','STOPPED'].includes(String(row.model_state).toUpperCase())) {
      pendingChats.delete(key); notice(`Session ${row.id} failed to load. See its model log.`, true);
    }
    if (ready && !$('chats-view').hidden) ensureChatTerminal(row, screen);
  }
}

function closeChatTerminal(key, host) {
  const item = chatTerminals.get(key);
  if (!item) return;
  chatTerminals.delete(key);
  if (item.poll) window.clearInterval(item.poll);
  item.observer?.disconnect();
  if (item.id) fetch(`/api/terminal?id=${encodeURIComponent(item.id)}`, {method: 'DELETE'}).catch(() => {});
  item.terminal.dispose();
  host.textContent = 'Load the model and attach its agent to open this terminal.';
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
    const resize = () => {
      if (!host.clientWidth) return;
      const cols = Math.max(50, Math.floor(host.clientWidth / 8.1)); const rows = Math.max(6,Math.floor((host.clientHeight-20)/17));
      if (xterm.cols === cols && xterm.rows === rows) return;
      xterm.resize(cols, rows); api('/api/terminal/resize', {method: 'POST', body: JSON.stringify({id: item.id, cols, rows})}).catch(() => {});
    };
    item.resize = resize;
    item.observer = new ResizeObserver(() => window.requestAnimationFrame(resize)); item.observer.observe(host);
    window.requestAnimationFrame(resize);
    await api('/api/terminal/input', {method: 'POST', body: JSON.stringify({id: item.id, data: `run --session ${Number(row.id)} --resume\n`})});
    item.poll = window.setInterval(async () => { try { const data = await api(`/api/terminal/output?id=${encodeURIComponent(item.id)}&offset=${item.offset}`); item.offset = data.offset; if (data.data) { xterm.write(data.data); xterm.scrollToBottom(); } } catch {} }, 120);
  } catch (error) { chatTerminals.delete(key); xterm.write(`\r\nCould not open terminal: ${error.message}\r\n`); }
}

function connectReadyChats() {
  window.requestAnimationFrame(() => {
    for (const item of chatTerminals.values()) item.resize?.();
    for (const card of $('chat-cards').children) {
      if (card.querySelector('.chat-status')?.classList.contains('ready')) ensureChatTerminal({id: card.dataset.session}, card.querySelector('.chat-terminal'));
    }
  });
}

function selectSession(row, refreshLog = true) {
  selected = row;
  document.querySelectorAll('#sessions tr').forEach(item => item.classList.toggle('selected', Number(item.dataset.id) === Number(row.id)));
  $('selected-name').textContent = `Session ${row.id}`;
  renderDetails(row);
  document.querySelectorAll('.session-action').forEach(button => button.disabled = false);
  $('metrics-link').href = `/metrics.html?session=${encodeURIComponent(row.id)}`;
  $('gpu-overview').contentWindow?.postMessage({type: 'select-session', session: String(row.id)}, window.location.origin);
  if (refreshLog && $('modal').hidden && $('terminal-host').hidden) loadLog();
}

$('gpu-overview').addEventListener('load', () => {
  if (selected) $('gpu-overview').contentWindow?.postMessage({type: 'select-session', session: String(selected.id)}, window.location.origin);
});
window.addEventListener('message', event => {
  if (event.origin !== window.location.origin || event.data?.type !== 'metrics-height') return;
  if ($('gpu-overview').dataset.userSized) return;
  $('gpu-overview').style.height = `${Math.max(360, Math.min(1600, Number(event.data.height) || 410))}px`;
});

function openSession(row) {
  selectSession(row, false);
  const state = String(row.model_state || '').toUpperCase();
  if (row.native) { terminalWindow(`Session ${row.id}`, `run --session ${row.id} --resume\n`); return; }
  const loaded = Boolean(row.model) && ['LOADED', 'READY', 'RUNNING'].includes(state);
  if (loaded && row.attached) {
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
      cell.colSpan = 8;
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
      const values = [row.id, row.model_state || row.phase || '—', row.model || (row.native ? `Native ${row.native_cli || 'CLI'}` : 'No model'), row.node || row.host || '—', row.gpus || 0, row.job_id || '—', row.time_left || '∞', row.attached ? 'Yes' : 'No'];
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
    const label = requestedLog === 'model-queries' ? 'Model queries' : requestedLog === 'session' ? 'Telemetry' : requestedLog[0].toUpperCase() + requestedLog.slice(1);
    $('output-title').textContent = `${label} log`;
    $('log-path').textContent = data.path;
    const output = $('log-output');
    const follow = output.scrollHeight - output.scrollTop - output.clientHeight < 24;
    const position = output.scrollTop;
    output.textContent = data.content || 'No output yet.';
    output.scrollTop = follow ? output.scrollHeight : position;
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

function toolsDialog() {
  const sessionId = selected?.id;
  const run = (label, description, action, danger = false) => ({
    label, description, danger,
    run: () => runOperation(label, `${description}…`, '/api/actions', {action, session: sessionId, confirmed: true})
  });
  const options = [
    run('Discover remote jobs', 'Find existing allocations on configured hosts', 'discover-remote'),
    run('Refresh res-mon', 'Remove ended allocations while preserving uncertain sessions', 'refresh-monitor'),
    {label:'Cleanup', description:'Review every cleanup path before confirming.', danger:true, run:cleanupDialog},
  ];
  if (sessionId != null) options.push(
    run('Refresh session', 'Restart the selected agent as a fresh conversation', 'refresh-session'),
    run('Set helper', 'Register the selected session as a helper', 'set-helper'),
    run('Restart', 'Retry the selected failed or ended allocation', 'restart'),
    run('Reconnect', 'Repair the selected model SSH tunnel', 'reconnect'),
  );
  options.push({label: 'Terminal', description: 'Open the integrated command terminal.', run: () => terminalWindow('Integrated terminal')});
  menu('Tools', options);
}

async function cleanupDialog() {
  const content=node('div','form','Collecting cleanup paths…');
  const token=openDialog('Review cleanup',content);
  try {
    const plan=await api('/api/cleanup-preview');
    if(token!==dialogToken)return;
    content.replaceChildren(node('p','',`${plan.paths.length} paths. Directories and their contents are listed below. Remote paths are prefixed with their host.`));
    const list=node('pre','cleanup-paths',plan.paths.join('\n') || 'No safely identifiable leftovers.');
    list.tabIndex=0;content.append(list);
    content.append(node('p','',plan.actions.join('\n')));
    const actions=node('div','form-actions');
    const cancel=node('button','','Cancel');cancel.onclick=closeDialog;
    const confirm=node('button','danger','Acknowledge and clean listed files');
    confirm.disabled=!plan.paths.length;
    confirm.onclick=()=>runOperation('Cleanup','Cleaning the reviewed paths…','/api/actions',
      {action:'cleanup',confirmed:true,preview_token:plan.token});
    actions.append(cancel,confirm);content.append(actions);
  } catch(error) { if(token===dialogToken)content.textContent=`Unable to preview cleanup: ${error.message}. Nothing was deleted.`; }
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
  const loading = operationView(`Session ${sessionId}`, 'Loading session options…');
  openDialog(`Session ${sessionId}`, loading);
  try {
    const data = await api(`/api/models?session=${sessionId}`);
    if (data.native) {
      const form = node('form', 'form');
      const grid = node('div', 'form-grid');
      const cli = selectControl(['auto', 'codex', 'claude'], data.agent_cli || 'auto');
      const workdir = inputControl(data.agent_workdir || ''); workdir.placeholder = 'Default project directory';
      const rag = inputControl(data.rag || ''); rag.placeholder = '/path/to/src:/path/to/README.md';
      const ragThreads = inputControl(String(data.rag_threads || 2), 'number'); ragThreads.min = '1'; ragThreads.step = '1';
      const ragMemory = inputControl(String(data.rag_memory_gb || 0), 'number'); ragMemory.min = '0'; ragMemory.step = '0.5';
      const ragGpu = selectControl(['no', 'yes'], data.rag_gpu ? 'yes' : 'no');
      grid.append(field('Native CLI', cli, 'Uses its own account and model.'), field('Agent work directory', workdir), field('RAG', rag, 'Colon-separated source paths.', true), field('RAG CPU cores', ragThreads, 'Embedding threads on this host.'), field('RAG memory (GiB)', ragMemory, 'Hard memory limit; 0 means unlimited.'), field('RAG GPU', ragGpu, 'Uses an available local GPU for RAG.'));
      const submit = node('button', 'primary', 'Attach / Restart agent'); submit.type = 'submit';
      const actions = node('div', 'form-actions'); actions.append(submit); form.append(grid, actions);
      form.addEventListener('submit', event => {
        event.preventDefault();
        const settings = {cli: cli.value, agent_workdir: workdir.value, rag: rag.value, rag_threads: ragThreads.value, rag_memory_gb: ragMemory.value, rag_gpu: ragGpu.value, resume: true};
        runOperation(`Session ${sessionId}`, 'Starting the native agent terminal…', '/api/native', {session: sessionId, settings}, true);
      });
      openDialog(`Session ${sessionId}`, form);
      return;
    }
    const form = node('form', 'form');
    const grid = node('div', 'form-grid');
    const machineSection = node('details', 'attach-section');
    machineSection.open = !data.can_attach;
    machineSection.append(node('summary', '', 'Machine & model loading'), grid);
    const ragSection = node('details', 'attach-section'); ragSection.open = true;
    const ragGrid = node('div', 'form-grid');
    ragSection.append(node('summary', '', 'RAG'), ragGrid);
    const modelOptions = data.models.map(model => ({value: model.name, label: `${model.alias || model.name} · ${(model.size_bytes / 1073741824).toFixed(1)} GiB`}));
    if (data.discovery_warning) notice(data.discovery_warning, true);
    const model = selectControl(modelOptions, data.models.find(item => item.alias === data.current || item.name === data.current)?.name || data.models[0]?.name || '');
    const mtp = selectControl(['auto', 'on', 'off'], 'auto');
    const location = selectControl(['local', 'remote'], data.agent_location || 'local');
    const cli = selectControl(['auto', 'codex', 'claude'], data.agent_cli || 'auto');
    const workdir = inputControl(data.agent_workdir || ''); workdir.placeholder = 'Default project directory';
    const rag = inputControl(data.rag || ''); rag.placeholder = '/path/to/src:/path/to/README.md';
    const ragThreads = inputControl(String(data.rag_threads || 2), 'number'); ragThreads.min = '1'; ragThreads.step = '1';
    const ragMemory = inputControl(String(data.rag_memory_gb || 0), 'number'); ragMemory.min = '0'; ragMemory.step = '0.5';
    const ragGpu = selectControl(['no', 'yes'], data.rag_gpu ? 'yes' : 'no');
    const ragCompute = selectControl(['local', 'remote'], data.rag_compute || data.agent_location || 'local');
    const ragPathsLocation = selectControl([{value: 'local', label: 'Local paths'}, {value: 'remote', label: 'Remote paths'}, {value: 'shared', label: 'Shared: local = remote'}], data.rag_paths_location || data.agent_location || 'local');
    const remoteRag = selectControl([{value:'',label:'This machine’s RAG settings'}, ...Object.entries(data.remote_rags || {}).map(([id,config]) => ({value:id,label:config.paths.join(', ')}))], '');
    const remoteRagField = field('RAG configuration',remoteRag,'Local RAG belongs to this terminal. Remote RAG sources and indexes can be reused by either machine.',true);
    ragGrid.append(remoteRagField);
    remoteRag.addEventListener('change', () => {
      const saved = data.remote_rags?.[remoteRag.value];
      if (saved) {
        rag.value = saved.paths.join(':'); ragCompute.value = 'remote'; ragPathsLocation.value = 'remote';
        ragThreads.value = saved.threads || 2; ragMemory.value = saved.memory_gb || 0; ragGpu.value = saved.gpu ? 'yes' : 'no';
      } else applyModelDefaults(true);
      for (const control of [rag,ragCompute,ragPathsLocation,ragThreads,ragMemory,ragGpu]) control.disabled = Boolean(saved);
      updateRagPaths();
    });
    const server = inputControl(data.server_options_by_model?.[model.value] || '');
    const build = selectControl([{value:'native',label:'Native llama.cpp build'},{value:'container',label:'Container'}],data.build_mode || 'native');
    const containerPath = inputControl(data.container_path || '');
    containerPath.placeholder = '/remote/path/llama-server.sif';
    const containerField = field('Container path',containerPath,'Path on the model host.',true);
    const updateBuild = () => { containerField.hidden = build.value !== 'container'; containerPath.required = build.value === 'container'; };
    build.addEventListener('change',updateBuild); updateBuild();
    grid.append(field('llama.cpp runtime',build,'Changing the runtime requires loading the model again.'),containerField);

    grid.append(field('Model', model, '', true), field('MTP', mtp), field('Agent location', location), field('Agent CLI', cli), field('Agent work directory', workdir), field('Model server options', server, 'Batch, context and backend arguments.', true));
    ragGrid.append(field('RAG', rag, 'Colon-separated source paths.', true), field('RAG compute', ragCompute, 'Where the embedding model and vector index run.'), field('RAG paths live on', ragPathsLocation, 'Different hosts are synchronized to a per-session snapshot.', true), field('RAG CPU cores', ragThreads, 'Embedding threads on the RAG compute host.'), field('RAG memory (GiB)', ragMemory, 'Hard limit on the RAG compute host; 0 means unlimited.'), field('RAG GPU', ragGpu, 'Uses an available GPU on the RAG compute host.'));

    function updateRagLocation() {
      const remote = location.value === 'remote';
      workdir.placeholder = remote ? 'Absolute directory on the remote host' : 'Default local project directory';
      if (!data.rag_compute) ragCompute.value = location.value;
      if (!data.rag_paths_location) ragPathsLocation.value = location.value;
      updateRagPaths();
    }
    function updateRagPaths() {
      const source = ragPathsLocation.value;
      rag.placeholder = source === 'local' ? '/Users/me/project/src:/Users/me/project/README.md' : source === 'remote' ? '/remote/project/src:/remote/project/docs' : '/shared/project/src:/shared/project/docs';
      rag.closest('.field').querySelector('small').textContent = source === ragCompute.value || source === 'shared'
        ? 'Paths are read directly; no snapshot copy is needed.'
        : `Paths are synchronized once from ${source} storage to ${ragCompute.value} RAG compute.`;
    }
    location.addEventListener('change', updateRagLocation);ragCompute.addEventListener('change',updateRagPaths);ragPathsLocation.addEventListener('change',updateRagPaths);updateRagLocation();
    function applyModelDefaults(initial = false) {
      const saved = data.preferences?.[model.value] || {};
      const current = data.models.find(item => item.name === model.value);
      const sameModel = current && [current.name,current.alias].includes(data.current);
      rag.value = saved.rag ?? (initial && sameModel ? data.rag || '' : '');
      ragCompute.value = saved.rag_compute || (initial && sameModel ? data.rag_compute : '') || location.value;
      ragPathsLocation.value = saved.rag_paths_location || (initial && sameModel ? data.rag_paths_location : '') || location.value;
      build.value = saved.build_mode || data.build_mode || 'native';
      server.value = data.server_options_by_model?.[model.value] || '';
      updateRagPaths(); updateBuild();
    }
    model.addEventListener('change', () => applyModelDefaults()); applyModelDefaults(true);
    form.append(machineSection, ragSection);
    let replaceLoaded = null;
    if (data.loaded || data.remote_owner) {
      const warning = node('label', 'replace-warning');
      replaceLoaded = document.createElement('input'); replaceLoaded.type = 'checkbox';
      warning.append(replaceLoaded, node('span', '', data.remote_owner ? `This session is allocated on ${data.remote_owner}. Acknowledge to release its model and start the new model and terminal on this machine.` : 'A model is already loaded. Stop it and replace the model and flags for all attached machines. To change only this terminal’s RAG, use Attach to loaded model.'));
      machineSection.append(warning);
    }
    const actions = node('div', 'form-actions');
    const attach = node('button', '', 'Attach existing terminal'); attach.type = 'button'; attach.addEventListener('click', () => terminalWindow(`Session ${sessionId}`, `run --session ${sessionId} --resume\n`));
    const submit = node('button', 'primary', 'Load and start'); submit.type = 'submit';
    if (data.can_attach) {
      const reuse = node('button', 'primary', 'Attach to loaded model'); reuse.type = 'button';
      reuse.onclick = () => runOperation(`Attach session ${sessionId}`, 'Connecting to the shared model and preparing this machine’s terminal and RAG…', '/api/load', {session:sessionId,settings:{attach_existing:true,expected_owner:data.expected_owner,expected_generation:data.expected_generation,agent_location:location.value,cli:cli.value,agent_workdir:workdir.value,rag:rag.value,rag_compute:ragCompute.value,rag_paths_location:ragPathsLocation.value,rag_threads:ragThreads.value,rag_memory_gb:ragMemory.value,rag_gpu:ragGpu.value,remote_rag_id:remoteRag.value}}, true);
      actions.append(reuse);
    }
    attach.hidden = !selected?.attached;
    const modelActions = node('div', 'form-actions'); modelActions.append(submit); machineSection.append(modelActions);
    actions.append(attach); form.append(actions);
    form.addEventListener('submit', event => {
      event.preventDefault();
      if (replaceLoaded && !replaceLoaded.checked) { notice('Acknowledge model replacement before loading the new configuration.', true); return; }
      runOperation(`Start session ${sessionId}`, 'Loading the model and preparing the retained agent terminal…', '/api/load', {session: sessionId, settings: {build_mode:build.value, container_path:containerPath.value, model: model.value, mtp: mtp.value, agent_location: location.value, cli: cli.value, agent_workdir: workdir.value, rag: rag.value, rag_compute: ragCompute.value, rag_paths_location: ragPathsLocation.value, rag_threads: ragThreads.value, rag_memory_gb: ragMemory.value, rag_gpu: ragGpu.value, remote_rag_id:remoteRag.value, replace_loaded: Boolean(replaceLoaded?.checked), expected_owner: data.expected_owner, expected_generation:data.expected_generation, server_options: server.value}}, true);
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
    {label: 'Unload model', description: 'Stop the model and keep the resource allocation.', run: () => runOperation('Unload model', 'Stopping the model while retaining the allocation…', '/api/actions', {action: 'unload', session: sessionId, confirmed: true}, true)},
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
$('tools-menu').addEventListener('click', toolsDialog);
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

// Each browser tab owns a lease; the server exits after the last tab closes.
const browserLease = crypto.randomUUID();
function pingBrowser() {
  fetch('/api/browser/ping', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id: browserLease})}).catch(() => {});
}
pingBrowser();
setInterval(pingBrowser, 15000);
window.addEventListener('pageshow', pingBrowser);
window.addEventListener('pagehide', () => navigator.sendBeacon('/api/browser/close', JSON.stringify({id: browserLease})));


// A full-width drag edge keeps resizing usable for both mouse and touch input.
function makeResizable(panel, target, key) {
  const handle = node('div','panel-resizer'); handle.tabIndex=0; handle.setAttribute('role','separator');
  handle.setAttribute('aria-label','Resize '+key); handle.setAttribute('aria-orientation','horizontal');
  const storageKey='llm-away-height-'+key;
  const apply = height => {
    target.style.height = `${Math.max(140,Math.min(1800,height))}px`;
    target.dataset.userSized='1';
    try { localStorage.setItem(storageKey,target.style.height); } catch {}
    resizeTerminal();
  };
  try { const saved=parseFloat(localStorage.getItem(storageKey)); if(Number.isFinite(saved)) apply(saved); } catch {}
  handle.onpointerdown = event => {
    event.preventDefault(); const initial=target.getBoundingClientRect().height; const y=event.clientY;
    handle.setPointerCapture(event.pointerId); document.body.classList.add('resizing-panel');
    handle.onpointermove = move => apply(initial+move.clientY-y);
    handle.onpointerup = handle.onpointercancel = () => { handle.onpointermove=null; document.body.classList.remove('resizing-panel'); };
  };
  handle.onkeydown = event => { if(['ArrowUp','ArrowDown'].includes(event.key)) {event.preventDefault();apply(target.getBoundingClientRect().height+(event.key==='ArrowUp'?-20:20));} };
  panel.append(handle);
}
makeResizable(document.querySelector('.monitor-panel'),$('log-output'),'monitor-log');
makeResizable(document.querySelector('.gpu-overview-panel'),$('gpu-overview'),'gpu-history');
makeResizable($('terminal-slot'),$('terminal-host'),'terminal');

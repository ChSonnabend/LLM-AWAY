/* Allocation-bound training controls. All payloads are checked remotely too. */
let trainingRows = [];
const trainingPending = new Set();
const trainingActive = new Set(['QUEUED','PREPARING','DISTILLING','LOADING','TRAINING','SAVING','EXPORTING','EVALUATING','STOPPING']);
function renderTrainingRows(rows) {
  trainingRows = rows.filter(row => !row.native);
  const select = $('training-session');
  const wanted = select.value || String(trainingRows.find(row => trainingActive.has(row.training?.phase))?.id || selected?.id || '');
  const options = trainingRows.map(row => `${row.id}:${row.host}` ).join('|');
  if (select.dataset.options !== options) {
    select.replaceChildren(...trainingRows.map(row => { const option = node('option','',`Session ${row.id} · ${row.host}`); option.value = row.id; return option; }));
    select.dataset.options = options;
    if (trainingRows.some(row => String(row.id) === wanted)) select.value = wanted;
  }
  const host = trainingRows.find(row => String(row.id) === select.value)?.host || '';
  if (select.dataset.defaultsHost !== host) { select.dataset.defaultsHost = host; restoreTrainingDefaults(true); }
  renderTrainingStatus();
}
function renderTrainingStatus() {
  const row = trainingRows.find(row => String(row.id) === $('training-session').value);
  syncTrainingGraph(row);
  const state = row?.training || {};
  $('training-log-title').textContent = row ? `Training output · Session ${row.id} · ${row.host} · ${state.phase || 'No run'}` : 'Training output';
  const active = trainingActive.has(state.phase);
  const percent = Math.min(100, Math.max(0, Number(state.percent) || 0));
  $('training-summary').textContent = row ? `${state.phase || 'Ready'} · ${state.files_evaluated ?? state.files_trained ?? 0}/${state.files_total || 0} files ${state.files_evaluated != null ? 'evaluated' : 'trained'} (${percent}%)${state.step != null ? ` · step ${state.step}/${state.steps_total} · epoch ${(state.epoch || 0).toFixed(2)}` : ''}` : 'Select an allocation.';
  const fill = $('training-progress-fill'); fill.style.width = `${percent}%`; fill.style.background = `hsl(${percent * 1.2} 65% 43%)`;
  fill.parentElement.setAttribute('aria-valuenow', String(percent));
  const details = [state.files_prepared != null ? `Prepared: ${state.files_prepared}/${state.files_registered || state.files_total} files; skipped: ${state.skipped || 0}` : '', state.teacher_files != null ? `Teacher processed ${state.teacher_files} files` : '', state.run_id ? `Run: ${state.run_id}` : '', state.time ? `Last status update: ${new Date(state.time*1000).toLocaleString()}` : '', state.output ? `Run directory: ${state.output}` : '', state.adapter ? `Saved adapter: ${state.adapter}` : '', state.export_path ? `Export: ${state.export_path}` : '', state.model_preset ? `Chat model preset: ${state.model_preset}` : '', state.evaluation_path ? `Evaluation report: ${state.evaluation_path}` : '', state.metrics ? JSON.stringify(state.metrics) : '', state.error || state.message || '', ...(state.skipped_files || []).map(file => `Skipped: ${file.path} — ${file.reason}`), 'File progress counts all chunks of each source through at least one optimizer update. Later epochs are shown separately.'].filter(Boolean).join('\n');
  const output = $('training-details');
  const selection = window.getSelection();
  if (!output.dataset.selecting && !(selection && !selection.isCollapsed && (output.contains(selection.anchorNode) || output.contains(selection.focusNode))) && output.textContent !== details) output.textContent = details;
  for (const id of ['training-start','training-q4','training-q8','training-evaluate']) $(id).disabled = !row || active || trainingPending.has(String(row.id));
  if (!$('training-adapter').value && state.adapter) $('training-adapter').value=state.adapter;
  $('training-stop').disabled = !active;
  const age = state.log_time ? Math.max(0, Math.floor(Date.now()/1000-state.log_time)) : null;
  $('training-log-age').textContent = age === null ? 'No worker output yet.' : `Last log output ${age}s ago. ${active && age > 120 ? 'No recent log output; check GPU activity below. Silence alone does not prove the process is stuck.' : 'File training progress begins after weight loading and tokenization.'}`;
  updateTrainingText($('training-live-log'), (state.log_tail || 'Waiting for worker output…').replace(/\x1b\[[0-9;?]*[A-Za-z]/g,''));
}
async function trainingOperation(action, quantization) {
  const row = trainingRows.find(row => String(row.id) === $('training-session').value);
  if (!row) return;
  if (['evaluate','export'].includes(action) && !$('training-adapter').value.trim()) { notice('Select a saved adapter directory before evaluation or export.', true); $('training-adapter').focus(); return; }
  if (action === 'evaluate' && !$('training-eval-paths').value.trim()) { notice('Enter the held-out evaluation folders/files and choose their source location.', true); $('training-eval-paths').focus(); return; }
  const confirmed = $('training-confirm').checked;
  if (action !== 'stop' && !confirmed) { notice('Acknowledge that this interrupts all chats on this allocation.', true); return; }
  const settings = {model:$('training-model').value,paths:$('training-paths').value,paths_location:$('training-source').value,
    precision:$('training-precision').value,epochs:$('training-epochs').value,sequence_length:$('training-length').value,
    lora_rank:$('training-rank').value,teacher_session:$('training-teacher').value,python:$('training-python').value,
    adapter:$('training-adapter').value,save_destination:$('training-save-destination').value,export_destination:$('training-export-destination').value,eval_paths:$('training-eval-paths').value,eval_paths_location:$('training-eval-source').value,runtime:$('training-runtime').value,container:$('training-container').value,quantization};
  if (trainingPending.has(String(row.id))) return;
  trainingPending.add(String(row.id));
  renderTrainingStatus();
  try {
  await runOperation('Fine-tuning', action === 'stop' ? 'Requesting a checkpoint and graceful stop…' : 'Preparing the allocation. Existing chats will be paused…', '/api/training', {session:row.id,action,settings,confirmed,run_id:row.training?.run_id}, true, true);
  } finally { trainingPending.delete(String(row.id)); renderTrainingStatus(); }
  await loadSessions();
}
$('training-session').addEventListener('change', renderTrainingStatus);
$('training-start').addEventListener('click', () => trainingOperation('start'));
$('training-stop').addEventListener('click', () => trainingOperation('stop'));
$('training-evaluate').addEventListener('click', () => trainingOperation('evaluate'));
$('training-q4').addEventListener('click', () => trainingOperation('export','q4_k_m'));
$('training-q8').addEventListener('click', () => trainingOperation('export','q8_0'));

let previousTrainingRuntime = 'native';
const trainingPythonByRuntime = {};
function updateTrainingRuntime() {
  const runtime = $('training-runtime').value;
  if (runtime !== previousTrainingRuntime) {
    trainingPythonByRuntime[previousTrainingRuntime] = $('training-python').value;
    $('training-python').value = trainingPythonByRuntime[runtime] || '';
    previousTrainingRuntime = runtime;
  }
  const container = $('training-runtime').value !== 'native';
  $('training-container-field').hidden = !container;
  $('training-container').required = container;
  $('training-python').placeholder = container ? 'python3 (inside the container)' : 'remote/venvs/unsloth/bin/python';
  $('training-python-help').textContent = container ? 'Optional Python executable inside the image, e.g. /opt/venv/bin/python. Applies to training and export.' : 'Install on the remote host with remote/bin/setup-training. Uses the allocation’s GPUs with automatic layer sharding.';
}
function syncTrainingGraph(row) {
  $('training-gpu-panel').hidden = !row;
  if (!row) return;
  $('training-metrics-link').href = `/metrics.html?session=${encodeURIComponent(row.id)}`;
  $('training-gpu-overview').contentWindow?.postMessage({type:'select-session',session:String(row.id)}, window.location.origin);
}
$('training-runtime').addEventListener('change', updateTrainingRuntime);
$('training-gpu-overview').addEventListener('load', () => syncTrainingGraph(trainingRows.find(row => String(row.id) === $('training-session').value)));
makeResizable($('training-gpu-panel'), $('training-gpu-overview'), 'training-gpu-history');
updateTrainingRuntime();

$('training-details').addEventListener('pointerdown', event => { if (event.button === 0) $('training-details').dataset.selecting = 'true'; });
for (const event of ['pointerup','pointercancel','blur']) window.addEventListener(event, () => { delete $('training-details').dataset.selecting; });

function updateTrainingText(output, text) {
  const selection=window.getSelection();
  if(output.dataset.selecting || (selection && !selection.isCollapsed && (output.contains(selection.anchorNode) || output.contains(selection.focusNode))) || output.textContent===text) return;
  const following=output.scrollHeight-output.scrollTop-output.clientHeight<24;
  const position=output.scrollTop;
  output.textContent=text;
  output.scrollTop=following ? output.scrollHeight : position;
}
$('training-live-log').addEventListener('pointerdown', event => { if(event.button===0) $('training-live-log').dataset.selecting='true'; });
for(const event of ['pointerup','pointercancel','blur']) window.addEventListener(event, () => { delete $('training-live-log').dataset.selecting; });
let defaultsRequest=0;
async function restoreTrainingDefaults(lastModel=false) {
  const session=$('training-session').value;
  if(!session) return;
  const request=++defaultsRequest;
  const model=lastModel ? '' : $('training-model').value.trim();
  const snapshot=['model','paths','source','runtime','container','python','adapter','save-destination','export-destination','eval-paths','eval-source'].map(k=>[$('training-'+k),$('training-'+k).value]);
  try {
    const values=await api(`/api/training/defaults?session=${encodeURIComponent(session)}&model=${encodeURIComponent(model)}`);
    if(request!==defaultsRequest || session!==$('training-session').value || snapshot.some(([el,value])=>el.value!==value)) return;
    for(const [key,id] of Object.entries({model:'model',paths:'paths',paths_location:'source',runtime:'runtime',container:'container',python:'python',adapter:'adapter',save_destination:'save-destination',export_destination:'export-destination',eval_paths:'eval-paths',eval_paths_location:'eval-source'})) {
      if(values[key]!==undefined) $('training-'+id).value=values[key];
    }
    // Set runtime before restoring its saved Python path.
    updateTrainingRuntime();
    if(values.python!==undefined) $('training-python').value=values.python;
  } catch(error) { notice('Could not restore training defaults: '+error.message,true); }
}
$('training-model').addEventListener('change',()=>restoreTrainingDefaults());
$('training-session').addEventListener('change',()=>restoreTrainingDefaults(true));

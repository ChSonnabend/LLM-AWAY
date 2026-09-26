/* Allocation-bound training controls. All payloads are checked remotely too. */
let trainingRows = [];
const trainingActive = new Set(['QUEUED','PREPARING','DISTILLING','LOADING','TRAINING','SAVING','EXPORTING','STOPPING']);
function renderTrainingRows(rows) {
  trainingRows = rows.filter(row => !row.native);
  const select = $('training-session');
  const wanted = select.value || String(selected?.id || '');
  const options = trainingRows.map(row => `${row.id}:${row.host}` ).join('|');
  if (select.dataset.options !== options) {
    select.replaceChildren(...trainingRows.map(row => { const option = node('option','',`Session ${row.id} · ${row.host}`); option.value = row.id; return option; }));
    select.dataset.options = options;
    if (trainingRows.some(row => String(row.id) === wanted)) select.value = wanted;
  }
  renderTrainingStatus();
}
function renderTrainingStatus() {
  const row = trainingRows.find(row => String(row.id) === $('training-session').value);
  syncTrainingGraph(row);
  const state = row?.training || {};
  const active = trainingActive.has(state.phase);
  const percent = Math.min(100, Math.max(0, Number(state.percent) || 0));
  $('training-summary').textContent = row ? `${state.phase || 'Ready'} · ${state.files_trained || 0}/${state.files_total || 0} files trained (${percent}%)${state.step != null ? ` · step ${state.step}/${state.steps_total} · epoch ${(state.epoch || 0).toFixed(2)}` : ''}` : 'Select an allocation.';
  const fill = $('training-progress-fill'); fill.style.width = `${percent}%`; fill.style.background = `hsl(${percent * 1.2} 65% 43%)`;
  fill.parentElement.setAttribute('aria-valuenow', String(percent));
  $('training-details').textContent = [state.files_prepared != null ? `Prepared: ${state.files_prepared}/${state.files_registered || state.files_total} files; skipped: ${state.skipped || 0}` : '', state.teacher_files != null ? `Teacher processed ${state.teacher_files} files` : '', state.output ? `Run directory: ${state.output}` : '', state.adapter ? `Saved adapter: ${state.adapter}` : '', state.export_path ? `Export: ${state.export_path}` : '', state.metrics ? JSON.stringify(state.metrics) : '', state.error || state.message || '', 'File progress counts all chunks of each source through at least one optimizer update. Later epochs are shown separately.'].filter(Boolean).join('\n');
  for (const id of ['training-start','training-q4','training-q8']) $(id).disabled = !row || active;
  $('training-stop').disabled = !active;
}
async function trainingOperation(action, quantization) {
  const row = trainingRows.find(row => String(row.id) === $('training-session').value);
  if (!row) return;
  const confirmed = $('training-confirm').checked;
  if (action !== 'stop' && !confirmed) { notice('Acknowledge that this interrupts all chats on this allocation.', true); return; }
  const settings = {model:$('training-model').value,paths:$('training-paths').value,paths_location:$('training-source').value,
    precision:$('training-precision').value,epochs:$('training-epochs').value,sequence_length:$('training-length').value,
    lora_rank:$('training-rank').value,teacher_session:$('training-teacher').value,python:$('training-python').value,
    adapter:$('training-adapter').value,runtime:$('training-runtime').value,container:$('training-container').value,quantization};
  await runOperation('Fine-tuning', action === 'stop' ? 'Requesting a checkpoint and graceful stop…' : 'Preparing the allocation. Existing chats will be paused…', '/api/training', {session:row.id,action,settings,confirmed,run_id:row.training?.run_id}, true);
  await loadSessions();
}
$('training-session').addEventListener('change', renderTrainingStatus);
$('training-start').addEventListener('click', () => trainingOperation('start'));
$('training-stop').addEventListener('click', () => trainingOperation('stop'));
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

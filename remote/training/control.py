"""Small dependency-free controller, invoked under the resource allocation lock."""
import json
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import uuid

ACTIVE={'QUEUED','PREPARING','DISTILLING','LOADING','TRAINING','SAVING','EXPORTING','EVALUATING','STOPPING'}

def read_status(state):
    path=state/'training-status.json'
    try:status=json.loads(path.read_text())
    except (OSError,ValueError):return {}
    output=Path(status.get('output') or state/'training'/str(status.get('run_id','')))
    status['log_path']=str(output/'training.log')
    try:
        log=output/'training.log'
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-12000));tail=stream.read().decode('utf-8',errors='replace')
        status.update(log_tail=tail.replace('\r','\n'),log_time=log.stat().st_mtime)
    except OSError:pass
    try:status['skipped_files']=json.loads((output/'dataset-report.json').read_text()).get('skipped',[])
    except (OSError,ValueError):pass
    return status

def active(state):return read_status(state).get('phase') in ACTIVE

def normalize_settings(settings,root,export=False,evaluate=False):
    """Resolve user paths once; workers and containers may have a different cwd."""
    settings=dict(settings)
    def resolve(raw,repository=False):
        raw=str(raw).strip()
        if not raw:return raw
        path=Path(raw).expanduser()
        candidates=[path] if path.is_absolute() else [root/path,root.parent/path]
        for candidate in candidates:
            if candidate.exists():return str(candidate.resolve())
        if repository and not raw.startswith(('/', '.', '~', 'remote/', 'models/')) and len(raw.split('/'))<=2:
            return raw
        raise ValueError('Remote path does not exist: '+raw+' (relative paths use '+str(root)+')')
    keys=['adapter'] if export or evaluate else ['model']
    if settings.get('runtime','native')!='native':keys.append('container')
    for key in keys:
        if settings.get(key):
            try:settings[key]=resolve(settings[key],repository=key=='model')
            except ValueError:
                if key!='container':raise  # execution() reports runtime/image errors together
    settings['paths']=[] if export else [resolve(raw) for raw in settings.get('paths',[])]
    if export or evaluate:
        settings.pop('model',None)
        adapter_config=Path(settings.get('adapter',''))/'adapter_config.json'
        if adapter_config.is_file():settings['base_model']=json.loads(adapter_config.read_text()).get('base_model_name_or_path','')
    if evaluate:
        # Evaluate the saved representation, not an unrelated training-form default.
        metadata=Path(settings.get('adapter','')).parent/'base-model.json'
        if metadata.is_file():
            precision=json.loads(metadata.read_text()).get('precision')
            if precision in ('bf16','qlora'):settings['precision']=precision
    if export:settings['precision']='qlora'
    destination=str(settings.get('destination','')).strip()
    if destination:
        path=Path(destination).expanduser()
        if not path.is_absolute():path=root/path
        path=path.resolve();parent=path
        while not parent.exists():parent=parent.parent
        if not parent.is_dir() or not os.access(parent,os.W_OK):raise ValueError('Output destination is not writable: '+str(path))
        settings['destination']=str(path)
        settings['destination_mount']=str(parent)
    return settings

def launch_log(state,message):
    path=state/'training-launch.log'
    with path.open('a') as stream:
        stream.write(time.strftime('%Y-%m-%d %H:%M:%S')+' | '+message+'\n')
    return path


def preflight_key(settings,current):
    return hashlib.sha256(json.dumps([settings,current.get('generation'),current.get('job_id')],sort_keys=True).encode()).hexdigest()


def execution(settings,state,root,current=None):
    """Use the same interpreter and mounts for preflight and the allocated worker."""
    mode=settings.get('runtime','native')
    if mode not in ('native','apptainer','singularity'):raise ValueError('Choose native, apptainer or singularity training runtime')
    if mode=='native':
        python=Path(settings.get('python') or root/'venvs/unsloth/bin/python').expanduser()
        if not python.is_file() or not os.access(python,os.X_OK):raise ValueError('Training environment missing. Run remote/bin/setup-training first, or choose a container')
        command=[str(python)]
    else:
        runtime=shutil.which(mode)
        if not runtime:raise ValueError(mode+' is not installed on the remote host')
        image_path=str(settings.get('container','')).strip()
        image=Path(image_path).expanduser()
        if not image_path or not image.exists():raise ValueError('Provide an existing remote container image path')
        image=image.resolve()
        mounts={root.absolute(),root.resolve(),state.absolute(),state.resolve()}
        for raw in list(settings.get('paths') or [])+[settings.get('model',''),settings.get('adapter',''),settings.get('destination_mount',''),settings.get('models_dir',''),settings.get('base_model','')]:
            if not raw:continue
            path=Path(raw).expanduser()
            if path.exists():
                directory=path if path.is_dir() else path.parent
                mounts.update((directory.absolute(),directory.resolve()))
        binds=[]
        for path in sorted(mounts):
            if any(c in str(path) for c in (':',',','\n')):raise ValueError('Container bind paths cannot contain commas, colons or newlines: '+str(path))
            binds+=['--bind',str(path)+':'+str(path)]
        prefix=[runtime,'exec',*binds]
        python=str(settings.get('python') or 'python3')
        command=prefix+['--nv',str(image),python]
    probe_command=command+[str(root/'training/preflight.py'),str(settings.get('model') or settings.get('adapter') or ''),settings.get('precision','qlora')]
    current=current or {}
    if current.get('backend_type')=='slurm_server':
        if not current.get('job_id') or current.get('slurm_state')!='RUNNING':raise ValueError('Wait for the allocation to be RUNNING before training')
        probe_command=['srun','--jobid='+str(current['job_id']),'--overlap','--exact','--ntasks=1','--cpus-per-task=1','--chdir='+str(root),*probe_command]
    receipt=state/'training-preflight.json'
    key=preflight_key(settings,current)
    try:cached=json.loads(receipt.read_text())
    except (OSError,ValueError):cached={}
    if cached.get('key')==key and 0<=time.time()-cached.get('time',0)<300:
        launch_log(state,'Reusing successful preflight for this configuration and model generation.')
        return command
    log_path=launch_log(state,'Preflight command (180s timeout): '+shlex.join(probe_command))
    try:
        with log_path.open('a') as stream:
            probe=subprocess.run(probe_command,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=180)
    except (OSError,subprocess.TimeoutExpired) as exc:
        launch_log(state,'Preflight failed: '+str(exc))
        raise ValueError('Training environment check failed: '+str(exc)+'; log: '+str(log_path)) from exc
    if probe.returncode:
        detail=getattr(probe,'stderr','')
        if not isinstance(detail,str) or not detail:detail=log_path.read_text()[-2000:]
        launch_log(state,'Preflight exited with code '+str(probe.returncode))
        raise ValueError('Training environment check failed: '+detail[-2000:]+'; log: '+str(log_path))
    launch_log(state,'Preflight passed; inference may now be replaced.')
    receipt.write_text(json.dumps({'key':key,'time':time.time()}))
    return command

def dispatch(action,p,state,root,current,write):
    if action in ('training-status','training-stop'):
        return _dispatch(action,p,state,root,current,write)
    launch_log(state,'Received '+action+'; generation '+str(current.get('generation','')))
    try:return _dispatch(action,p,state,root,current,write)
    except Exception as exc:
        launch_log(state,type(exc).__name__+': '+str(exc))
        raise


def _dispatch(action,p,state,root,current,write):
    status=read_status(state)
    if action=='training-status':return status
    if action=='training-stop':
        if p.get('run_id')!=status.get('run_id'):raise ValueError('Training run changed; refresh before stopping')
        if status.get('phase') in ACTIVE:
            (Path(status['output'])/'stop').touch()
        return dict(status,stop_requested=True)
    validate_only=action=='training-check'
    if validate_only:action='training-'+p.get('operation','start')
    if action not in ('training-start','training-export','training-evaluate'):raise ValueError('Unknown training action')
    if p.get('confirmed') is not True:raise ValueError('Confirm that training/export will unload the model for all attached clients')
    if not current.get('active'):raise ValueError('Allocation has ended')
    if current.get('generation','')!=p.get('expected_generation',''):raise ValueError('Model changed; refresh before starting training')
    if active(state):raise ValueError('Training or export already occupies this allocation')
    settings=normalize_settings(p.get('settings') or {},root,export=action=='training-export',evaluate=action=='training-evaluate')
    if action=='training-export':settings['models_dir']=(current.get('session',{}).get('llamacpp',{}).get('models_dir') or str(root/'models'))
    sequence=int(settings.get('sequence_length',2048));epochs=float(settings.get('epochs',1));rank=int(settings.get('lora_rank',16))
    if not 128<=sequence<=131072 or not 0<epochs<=100 or not 1<=rank<=256:raise ValueError('Invalid sequence length, epoch count or LoRA rank')
    precision=settings.get('precision','qlora')
    if precision not in ('qlora','bf16'):raise ValueError('Choose qlora or bf16')
    run_id=uuid.uuid4().hex;output=state/'training'/run_id
    spec=dict(run_id=run_id,output=str(output),status_path=str(state/'training-status.json'),sequence_length=sequence,
              epochs=epochs,lora_rank=rank,precision=precision,gradient_accumulation=int(settings.get('gradient_accumulation',4)),
              learning_rate=float(settings.get('learning_rate',0.0002)),save_steps=int(settings.get('save_steps',50)))
    if not 1<=spec['gradient_accumulation']<=1024 or not 0<spec['learning_rate']<=0.1 or spec['save_steps']<1:raise ValueError('Invalid optimizer settings')
    if action in ('training-export','training-evaluate'):
        source=str(settings.get('adapter',''));adapter=Path(source).expanduser().resolve()
        if not source or not (adapter/'adapter_config.json').is_file():raise ValueError('Provide a saved adapter directory containing adapter_config.json')
        spec.update(action='evaluate' if action=='training-evaluate' else 'export',adapter=str(adapter))
        if action=='training-export':
            quant=settings.get('quantization','q4_k_m')
            if quant not in ('q4_k_m','q8_0'):raise ValueError('Choose q4_k_m or q8_0')
            spec.update(quantization=quant,models_dir=settings['models_dir'])
        else:
            if not settings.get('paths'):raise ValueError('Provide held-out evaluation folders/files')
            spec['paths']=settings['paths']
    else:
        model=str(settings.get('model','')).strip()
        if not model or model.lower().endswith('.gguf'):raise ValueError('Training requires a Hugging Face base checkpoint, not a GGUF')
        paths=settings.get('paths') or []
        if not paths or not all(Path(s).expanduser().exists() for s in paths):raise ValueError('All source paths must exist on the allocation host')
        spec.update(model=model,paths=paths)
        if settings.get('teacher'):
            spec['teacher']=settings['teacher']
    if settings.get('destination'):spec['artifact_dir']=str(Path(settings['destination'])/run_id)
    command=execution(settings,state,root,current)
    if validate_only:
        log=state/'training-launch.log'
        return {'validated':True,'log_path':str(log),'log_tail':log.read_text()[-8000:]}
    if spec.get('artifact_dir'):Path(spec['artifact_dir']).mkdir(parents=True,mode=0o700,exist_ok=False)
    output.mkdir(parents=True,mode=0o700)
    launch_log(output,'Queued '+action+'. Waiting for allocation worker to stop inference. Output log: '+str(output/'training.log'))
    (output/'training-launch.log').rename(output/'training.log')
    if settings.get('teacher'):
        key=output/'teacher-api-key';fd=os.open(str(key),os.O_CREAT|os.O_WRONLY|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:stream.write(str(settings.get('teacher_key','')))
        spec['teacher_key_file']=str(key)
    request=output/'request.json';write(request,json.dumps(spec))
    environment=''
    if settings.get('runtime','native')!='native':
        environment='if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then\n  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"\nfi\n'
    launch_log(state,'Publishing '+action+' worker command: '+shlex.join(command+[str(root/'training/runner.py'),str(request)]))
    script='#!/bin/bash\nset -euo pipefail\numask 077\nexport PYTHONUNBUFFERED=1\n'+environment+'echo "$(date -Is) | Inference stopped; starting training worker" >> '+shlex.quote(str(output/'training.log'))+'\nexec '+' '.join(shlex.quote(v) for v in command+[str(root/'training/runner.py'),str(request)])+' >> '+shlex.quote(str(output/'training.log'))+' 2>&1\n'
    write(state/(run_id+'.sh'),script)
    write(state/(run_id+'.training.json'),json.dumps({'output':str(output)}))
    write(state/'training-status.json',json.dumps(dict(run_id=run_id,output=str(output),phase='QUEUED',files_total=0,files_trained=0,percent=0,time=time.time())))
    # The worker stops the previous inference process before starting this generation.
    write(state/'desired',run_id)
    return read_status(state)

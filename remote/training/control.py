"""Small dependency-free controller, invoked under the resource allocation lock."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import uuid

ACTIVE={'QUEUED','PREPARING','DISTILLING','LOADING','TRAINING','SAVING','EXPORTING','STOPPING'}

def read_status(state):
    path=state/'training-status.json'
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def active(state):return read_status(state).get('phase') in ACTIVE

def execution(settings,state,root):
    """Use the same interpreter and mounts for preflight and the allocated worker."""
    mode=settings.get('runtime','native')
    if mode not in ('native','apptainer','singularity'):raise ValueError('Choose native, apptainer or singularity training runtime')
    probe_code="import importlib.util; assert all(importlib.util.find_spec(n) for n in ('unsloth','torch','transformers','pypdf','markitdown')), 'Install training requirements in the selected environment'"
    if mode=='native':
        python=Path(settings.get('python') or root/'venvs/unsloth/bin/python').expanduser()
        if not python.is_file() or not os.access(python,os.X_OK):raise ValueError('Training environment missing. Run remote/bin/setup-training first, or choose a container')
        command=[str(python)]
        probe_command=command
    else:
        runtime=shutil.which(mode)
        if not runtime:raise ValueError(mode+' is not installed on the remote host')
        image_path=str(settings.get('container','')).strip()
        image=Path(image_path).expanduser()
        if not image_path or not image.exists():raise ValueError('Provide an existing remote container image path')
        image=image.resolve()
        mounts={root.absolute(),root.resolve(),state.absolute(),state.resolve()}
        for raw in list(settings.get('paths') or [])+[settings.get('model',''),settings.get('adapter','')]:
            if not raw:continue
            path=Path(raw).expanduser()
            if path.exists():
                directory=path if path.is_dir() else path.parent
                mounts.update((directory.absolute(),directory.resolve()))
        binds=[]
        for path in sorted(mounts):
            if any(c in str(path) for c in (':',',','\n')):raise ValueError('Container bind paths cannot contain commas, colons or newlines: '+str(path))
            binds+=['--bind',str(path)+':'+str(path)]
        # No GPU probe on the SSH/login host; --nv is applied inside the allocation.
        prefix=[runtime,'exec',*binds]
        python=str(settings.get('python') or 'python3')
        probe_command=prefix+[str(image),python]
        command=prefix+['--nv',str(image),python]
    try:probe=subprocess.run(probe_command+['-c',probe_code],capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.TimeoutExpired) as exc:raise ValueError('Training environment check failed: '+str(exc)) from exc
    if probe.returncode:raise ValueError('Training environment check failed: '+(probe.stderr or probe.stdout)[-2000:])
    return command

def dispatch(action,p,state,root,current,write):
    status=read_status(state)
    if action=='training-status':return status
    if action=='training-stop':
        if p.get('run_id')!=status.get('run_id'):raise ValueError('Training run changed; refresh before stopping')
        if status.get('phase') in ACTIVE:
            (Path(status['output'])/'stop').touch()
        return dict(status,stop_requested=True)
    validate_only=action=='training-check'
    if validate_only:action='training-'+p.get('operation','start')
    if action not in ('training-start','training-export'):raise ValueError('Unknown training action')
    if p.get('confirmed') is not True:raise ValueError('Confirm that training/export will unload the model for all attached clients')
    if not current.get('active'):raise ValueError('Allocation has ended')
    if current.get('generation','')!=p.get('expected_generation',''):raise ValueError('Model changed; refresh before starting training')
    if active(state):raise ValueError('Training or export already occupies this allocation')
    settings=dict(p.get('settings') or {})
    command=execution(settings,state,root)
    sequence=int(settings.get('sequence_length',2048));epochs=float(settings.get('epochs',1));rank=int(settings.get('lora_rank',16))
    if not 128<=sequence<=131072 or not 0<epochs<=100 or not 1<=rank<=256:raise ValueError('Invalid sequence length, epoch count or LoRA rank')
    precision=settings.get('precision','qlora')
    if precision not in ('qlora','bf16'):raise ValueError('Choose qlora or bf16')
    run_id=uuid.uuid4().hex;output=state/'training'/run_id
    spec=dict(run_id=run_id,output=str(output),status_path=str(state/'training-status.json'),sequence_length=sequence,
              epochs=epochs,lora_rank=rank,precision=precision,gradient_accumulation=int(settings.get('gradient_accumulation',4)),
              learning_rate=float(settings.get('learning_rate',0.0002)),save_steps=int(settings.get('save_steps',50)))
    if not 1<=spec['gradient_accumulation']<=1024 or not 0<spec['learning_rate']<=0.1 or spec['save_steps']<1:raise ValueError('Invalid optimizer settings')
    if action=='training-export':
        source=str(settings.get('adapter',''));adapter=Path(source).expanduser().resolve()
        if not source or not (adapter/'adapter_config.json').is_file():raise ValueError('Provide a saved adapter directory containing adapter_config.json')
        quant=settings.get('quantization','q4_k_m')
        if quant not in ('q4_k_m','q8_0'):raise ValueError('Choose q4_k_m or q8_0')
        spec.update(action='export',adapter=str(adapter),quantization=quant)
    else:
        model=str(settings.get('model','')).strip()
        if not model or model.lower().endswith('.gguf'):raise ValueError('Training requires a Hugging Face base checkpoint, not a GGUF')
        paths=settings.get('paths') or []
        if not paths or not all(Path(s).expanduser().exists() for s in paths):raise ValueError('All source paths must exist on the allocation host')
        spec.update(model=model,paths=paths)
        if settings.get('teacher'):
            spec['teacher']=settings['teacher']
    if validate_only:return {'validated':True}
    output.mkdir(parents=True,mode=0o700)
    if settings.get('teacher'):
        key=output/'teacher-api-key';fd=os.open(str(key),os.O_CREAT|os.O_WRONLY|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:stream.write(str(settings.get('teacher_key','')))
        spec['teacher_key_file']=str(key)
    request=output/'request.json';write(request,json.dumps(spec))
    environment=''
    if settings.get('runtime','native')!='native':
        environment='if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then\n  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"\nfi\n'
    script='#!/bin/bash\nset -euo pipefail\numask 077\n'+environment+'exec '+' '.join(shlex.quote(v) for v in command+[str(root/'training/runner.py'),str(request)])+' >> '+shlex.quote(str(output/'training.log'))+' 2>&1\n'
    write(state/(run_id+'.sh'),script)
    write(state/(run_id+'.training.json'),json.dumps({'output':str(output)}))
    write(state/'training-status.json',json.dumps(dict(run_id=run_id,output=str(output),phase='QUEUED',files_total=0,files_trained=0,percent=0,time=time.time())))
    # The worker stops the previous inference process before starting this generation.
    write(state/'desired',run_id)
    return read_status(state)

"""Fine-tuning control through the same SSH allocation authority as inference."""
import json
import os
from pathlib import Path
import shlex
import tempfile
import uuid
import threading

ACTIVE={'QUEUED','PREPARING','DISTILLING','LOADING','TRAINING','SAVING','EXPORTING','EVALUATING','STOPPING'}

def state(number):
    from . import resources as r
    path=r.path_for(int(number));data=json.loads((path/'session.json').read_text())
    if data.get('native'):raise ValueError('Fine-tuning requires a GPU allocation, not a native CLI session')
    return path,data,r.config(data['config'])

def defaults(number,model='',save=None):
    """Machine-local input preferences keyed by SSH host, remote root and model."""
    path,data,cfg=state(number)
    from .resources import ROOT
    store=Path(os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/model.toml'))).with_suffix('.training-defaults.json')
    key=cfg.ssh.destination+'|'+cfg.remote.workdir
    with _defaults_lock:
        try:all_values=json.loads(store.read_text())
        except (OSError,ValueError):all_values={}
        host=all_values.setdefault(key,{'models':{}})
        model=str(model or host.get('last_model','')).strip()
        if save is not None:
            allowed=('paths','paths_location','runtime','container','python','adapter','save_destination','export_destination','eval_paths','eval_paths_location')
            host['models'].setdefault(model,{}).update({k:save[k] for k in allowed if k in save})
            host['last_model']=model
            store.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w',dir=store.parent,delete=False) as stream:
                json.dump(all_values,stream);temporary=stream.name
            os.chmod(temporary,0o600);os.replace(temporary,store)
        return dict(host['models'].get(model,{}),model=model)

_defaults_lock=threading.Lock()

def local_sources(cfg,allocation,token,paths,path):
    """Snapshot sources and rewrite explicit answer paths only in the copied manifests."""
    from . import rag_transfer
    import runpy
    from .resources import ROOT
    parser=runpy.run_path(str(ROOT.parent/'remote/training/dataset.py'))
    roots,files=parser['discover'](paths)
    roots=[root for i,root in enumerate(roots) if root not in roots[:i] and not any(other.is_dir() and other in root.parents for other in roots)]
    paths=[str(root) for root in roots]
    manifests=[p for p in files if p.name==parser['MANIFEST']]
    pairs_by_manifest={m:parser['labels']([m],roots) for m in manifests}
    targets=rag_transfer.snapshot(cfg,allocation,token,paths,'local','remote',path,snapshot_name='training-inputs-'+uuid.uuid4().hex)
    def mapped(source):
        for root,target in zip(roots,targets):
            if source==root:return target
            if root.is_dir() and root in source.parents:return target+'/'+str(source.relative_to(root))
        raise ValueError('Answer file is outside supplied paths')
    import csv,io
    rewritten={}
    for manifest,pairs in pairs_by_manifest.items():
        stream=io.StringIO();writer=csv.writer(stream)
        for question,answer in pairs:
            writer.writerow([question]);stream.write('['+mapped(answer)+']\n')
        rewritten[mapped(manifest)]=stream.getvalue()
    if rewritten:
        with tempfile.TemporaryFile() as stream:
            stream.write(json.dumps(rewritten).encode());stream.seek(0)
            rag_transfer._remote_command(cfg,allocation,['python3','-c',
                'import json,sys; from pathlib import Path; [Path(p).write_text(s) for p,s in json.load(sys.stdin).items()]'],input_stream=stream)
    return targets

def operate(number,action,settings=None,confirmed=False,run_id=None):
    from . import resources as r, shared_sessions
    path,data,cfg=state(number);settings=dict(settings or {});entered_settings=dict(settings)
    if action in ('status','stop'):
        result=r.remote(cfg,data['token'],data['remote_port'],'training-'+action,run_id=run_id)
        return result if action=='status' else 'Stop requested; weights will be saved at the next optimizer step. During preparation no weights have been trained yet.'
    if action not in ('start','export','evaluate'):raise ValueError('Unknown training operation')
    if not confirmed:raise ValueError('Confirm that all chats using this allocation will be interrupted')
    allocation=shared_sessions.current(path)
    if allocation.get('training',{}).get('phase') in ACTIVE:raise ValueError('Training/export is already running')
    if action=='evaluate':
        settings['paths']=settings.get('eval_paths','')
        settings['paths_location']=settings.get('eval_paths_location','remote')
    settings['destination']=settings.get('export_destination' if action=='export' else 'save_destination','')
    if action in ('start','evaluate'):
        paths=[s.strip() for s in str(settings.get('paths','')).split(os.pathsep) if s.strip()]
        if settings.get('paths_location','remote')=='local':paths=local_sources(cfg,allocation,data['token'],paths,path)
        settings['paths']=paths
        teacher_id=settings.pop('teacher_session',None)
        if teacher_id and action=='start':
            _,teacher,teacher_cfg=state(teacher_id)
            if teacher['token']==data['token']:raise ValueError('Teacher needs a separate allocation so it stays loaded during student training')
            if cfg.ssh.destination!=teacher_cfg.ssh.destination:raise ValueError('Teacher and student currently need the same configured SSH host')
            remote=shared_sessions.current(r.path_for(int(teacher_id)))
            if remote.get('model_state') not in ('LOADED','READY','RUNNING'):raise ValueError('Load the teacher model first')
            host=remote.get('host')
            if host in ('127.0.0.1','localhost') and cfg.backend_type!='direct':raise ValueError('Teacher loopback endpoint is not reachable from this allocation')
            settings['teacher']={'base_url':'http://'+host+':'+str(teacher['remote_port']),
                                 'model':remote['session']['model']['name'],'max_tokens':2048}
            settings['teacher_key']=r.remote(teacher_cfg,teacher['token'],teacher['remote_port'],'credential').get('api_key','')
    # Check prerequisites before changing ownership or interrupting any chat.
    r.remote(cfg,data['token'],data['remote_port'],'training-check',operation=action,settings=settings,
             confirmed=True,expected_generation=allocation.get('generation',''))
    owner=allocation.get('owner') or {}
    if owner.get('id') and owner['id']!=shared_sessions.client_identity():
        allocation=shared_sessions.claim(path,owner['id'],allocation.get('generation',''))
    result=r.remote(cfg,data['token'],data['remote_port'],'training-'+action,settings=settings,
                    confirmed=True,expected_generation=allocation.get('generation',''))
    preference_warning=''
    try:defaults(number,entered_settings.get('model',''),save=entered_settings)
    except (OSError,ValueError,TypeError) as exc:preference_warning=' (Could not save defaults: '+str(exc)+')'
    # No local model stop RPC: only detach this client; the remote worker owns the transition.
    from .terminals import engine
    engine()['stop'](path)
    return 'Queued '+action+' in allocation '+str(number)+'. Output: '+result['output']+preference_warning

def menu(number):
    from .monitor_ui import dropdown_win,_form_screen,terminal_operation
    choice=dropdown_win('Fine-tuning — session '+str(number),['Start training','Stop and save','Evaluate adapter','Export Q4','Export Q8','Show progress'],'Show progress')
    if choice is None:return 'Cancelled'
    if choice=='Show progress':
        s=terminal_operation(operate,number,'status')
        return f"{s.get('phase','No training')} | files {s.get('files_trained',0)}/{s.get('files_total',0)} ({s.get('percent',0)}%) | {s.get('error') or s.get('output','')}"
    if choice=='Stop and save':
        s=terminal_operation(operate,number,'status');return terminal_operation(operate,number,'stop',None,False,s.get('run_id'))
    saved=defaults(number)
    runtime_names={'native':'Native Python','apptainer':'Apptainer container','singularity':'Singularity container'}
    runtime=dropdown_win('Training runtime',list(runtime_names.values()),runtime_names.get(saved.get('runtime'),'Native Python'))
    if runtime is None:return 'Cancelled'
    mode={value:key for key,value in runtime_names.items()}[runtime]
    if choice.startswith('Export'):
        keys=['adapter','python','export_destination']
        labels=['Saved adapter directory','Training Python (optional)','Export under remote folder (optional)']
        action='export'
    elif choice=='Evaluate adapter':
        keys=['adapter','eval_paths','eval_paths_location','precision','sequence_length','python','save_destination']
        labels=['Saved adapter directory','Held-out folders/files (colon-separated)','Paths on local or remote machine','Precision: qlora or bf16','Sequence length','Training Python (optional)','Save results under remote folder (optional)']
        action='evaluate'
    else:
        keys=['model','paths','paths_location','precision','epochs','sequence_length','lora_rank','teacher_session','python','save_destination']
        labels=['Base BF16 model directory / HF repository','Source folders/files (colon-separated)','Paths on local or remote machine','Precision: qlora or bf16','Epochs','Sequence length','LoRA rank','Teacher session (optional)','Training Python (optional)','Save adapters under remote folder (optional)']
        action='start'
    if mode!='native':keys.append('container');labels.append('Remote container image path')
    initial={'paths_location':'remote','eval_paths_location':'remote','precision':'qlora','epochs':'1','sequence_length':'2048','lora_rank':'16',**saved}
    fields=[('text',label,None,str(initial.get(key,''))) for key,label in zip(keys,labels)]
    values=_form_screen(choice,fields,[f[3] for f in fields],choice)
    if values is None:return 'Cancelled'
    if dropdown_win('This unloads the model and interrupts all attached chats',['Cancel','Continue'],'Cancel')!='Continue':return 'Cancelled'
    settings=dict(zip(keys,values),runtime=mode)
    if action=='export':settings['quantization']='q4_k_m' if choice=='Export Q4' else 'q8_0'
    return terminal_operation(operate,number,action,settings,True)

"""Fine-tuning control through the same SSH allocation authority as inference."""
import json
import os
from pathlib import Path
import shlex
import tempfile
import uuid

ACTIVE={'QUEUED','PREPARING','DISTILLING','LOADING','TRAINING','SAVING','EXPORTING','STOPPING'}

def state(number):
    from . import resources as r
    path=r.path_for(int(number));data=json.loads((path/'session.json').read_text())
    if data.get('native'):raise ValueError('Fine-tuning requires a GPU allocation, not a native CLI session')
    return path,data,r.config(data['config'])

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
    path,data,cfg=state(number);settings=dict(settings or {})
    if action in ('status','stop'):
        result=r.remote(cfg,data['token'],data['remote_port'],'training-'+action,run_id=run_id)
        return result if action=='status' else 'Stop requested; weights will be saved at the next optimizer step. During preparation no weights have been trained yet.'
    if action not in ('start','export'):raise ValueError('Unknown training operation')
    if not confirmed:raise ValueError('Confirm that all chats using this allocation will be interrupted')
    allocation=shared_sessions.current(path)
    if allocation.get('training',{}).get('phase') in ACTIVE:raise ValueError('Training/export is already running')
    if action=='start':
        paths=[s.strip() for s in str(settings.get('paths','')).split(os.pathsep) if s.strip()]
        if settings.get('paths_location','remote')=='local':paths=local_sources(cfg,allocation,data['token'],paths,path)
        settings['paths']=paths
        teacher_id=settings.pop('teacher_session',None)
        if teacher_id:
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
    # No local model stop RPC: only detach this client; the remote worker owns the transition.
    from .terminals import engine
    engine()['stop'](path)
    return 'Queued '+action+' in allocation '+str(number)+'. Output: '+result['output']

def menu(number):
    from .monitor_ui import dropdown_win,_form_screen,terminal_operation
    choice=dropdown_win('Fine-tuning — session '+str(number),['Start training','Stop and save','Export Q4','Export Q8','Show progress'],'Show progress')
    if choice is None:return 'Cancelled'
    if choice=='Show progress':
        s=terminal_operation(operate,number,'status')
        return f"{s.get('phase','No training')} | files {s.get('files_trained',0)}/{s.get('files_total',0)} ({s.get('percent',0)}%) | {s.get('error') or s.get('output','')}"
    if choice=='Stop and save':
        s=terminal_operation(operate,number,'status');return terminal_operation(operate,number,'stop',None,False,s.get('run_id'))
    runtime=dropdown_win('Training runtime',['Native Python','Apptainer container','Singularity container'],'Native Python')
    if runtime is None:return 'Cancelled'
    mode={'Native Python':'native','Apptainer container':'apptainer','Singularity container':'singularity'}[runtime]
    if choice.startswith('Export'):
        fields=[('text','Saved adapter directory',None,''),('text','Training Python (optional)',None,'')]
    else:
        fields=[('text','Base BF16 model directory / HF repository',None,''),('text','Source folders/files (colon-separated)',None,''),
                ('text','Paths on local or remote machine',None,'remote'),('text','Precision: qlora or bf16',None,'qlora'),
                ('text','Epochs',None,'1'),('text','Sequence length',None,'2048'),('text','Teacher session (optional)',None,''),('text','Training Python (optional)',None,'')]
    if mode!='native':fields.append(('text','Remote container image path',None,''))
    values=_form_screen(choice,fields,[f[3] for f in fields],choice)
    if values is None:return 'Cancelled'
    if dropdown_win('This unloads the model and interrupts all attached chats',['Cancel','Continue'],'Cancel')!='Continue':return 'Cancelled'
    if choice.startswith('Export'):
        settings=dict(adapter=values[0],python=values[1],quantization='q4_k_m' if choice=='Export Q4' else 'q8_0');action='export'
    else:
        settings=dict(zip(['model','paths','paths_location','precision','epochs','sequence_length','teacher_session','python'],values));action='start'
    settings['runtime']=mode
    if mode!='native':settings['container']=values[-1]
    return terminal_operation(operate,number,action,settings,True)

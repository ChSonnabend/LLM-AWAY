"""Numbered native CLI sessions, retained in a local tmux terminal (also for SSH)."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid

from .config import AppConfig
from . import terminals


@contextmanager
def locked(path):
    with (path/'native.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield


def allocate(store, cli, host=None):
    from .resources import write
    # Fail before registering if persistent terminals are unavailable.
    terminals.engine()['tmux']()
    store.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (store/'registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        number=1+max([int(p.name) for p in store.iterdir() if p.name.isdigit()]+[0])
        path=store/str(number);path.mkdir(mode=0o700)
        cfg=AppConfig()
        cfg=replace(cfg,backend_type='direct',slurm=replace(cfg.slurm,gpus=0),
                    llamacpp=replace(cfg.llamacpp,visible_devices=''),
                    ssh=replace(cfg.ssh,connection='local',host='localhost'))
        data=dict(id=number,token=uuid.uuid4().hex,config=asdict(cfg),remote_port=0,
                  native=True,native_cli=cli,host=host or 'local',gpus=0,
                  phase='STARTING',model='',remote_cleanup_done=True)
        selection=dict(location='local',cli=cli,cwd=os.getcwd(),extra_args=[],
                       native=True,native_host=host)
        write(path/'session.json',data)
        write(path/'agent-selection.json',selection)
        (path/'session.log').write_text(f'Native {cli} session on {host or "local"}.\n')
    print(f'Session {number}: native {cli} on {host or "local"}. Ctrl+B then D detaches; res-mon / F2 reattaches.',flush=True)
    run(path)


def run(path, args=None):
    from .resources import write
    with locked(path):
        data=json.loads((path/'session.json').read_text())
        if data['phase']=='RELEASED':raise ValueError('This native session has been released')
        selection=json.loads((path/'agent-selection.json').read_text())
        if args:
            if any(getattr(args,key,None) for key in ('model','helper','rag','mtp')):
                raise ValueError('Native CLI sessions do not use custom models, helpers, or RAG options')
            if (getattr(args,'cli',None) not in (None,'auto',selection['cli']) or
                getattr(args,'agent_location','local')!='local' or
                getattr(args,'agent_workdir',None) not in (None,selection['cwd']) or
                getattr(args,'agent_args',None)):
                raise ValueError('Use the saved native session settings or create another session')
        try:
            terminals.ensure(path,data,selection)
            data.update(phase='RUNNING',error='')
        except Exception as exc:
            data.update(phase='FAILED',error=str(exc))
            write(path/'session.json',data)
            raise
        write(path/'session.json',data)
    if not getattr(args,'detach',False) and sys.stdout.isatty():terminals.attach(path,data,selection)


def update_settings(path, settings):
    """Apply saved native settings (CLI, workdir, RAG) and restart the agent."""
    from .resources import write
    with locked(path):
        data=json.loads((path/'session.json').read_text())
        if data['phase']=='RELEASED':raise ValueError('This native session has been released')
        selection=json.loads((path/'agent-selection.json').read_text())
        cli=settings.get('cli') or selection.get('cli')
        if cli not in (None,'auto',selection['cli']):
            raise ValueError('Native sessions keep the allocated CLI; create another session to change it')
        if cli in (None,'auto'):cli=selection['cli']
        cwd=settings.get('agent_workdir')
        if cwd is not None and str(cwd).strip():
            selection['cwd']=str(Path(str(cwd).strip()).expanduser())
        rag_paths=[item.strip() for item in str(settings.get('rag') or '').replace(';',':').split(':') if item.strip()]
        rag_config=None
        if rag_paths:
            rag_config={'paths':rag_paths,'threads':max(1,int(settings.get('rag_threads') or 2)),
                        'memory_gb':max(0,float(settings.get('rag_memory_gb') or 0)),
                        'gpu':settings.get('rag_gpu') in (True,'yes','on',1),
                        'compute':'local','paths_location':'local'}
        selection['rag_config']=rag_config;selection['rag_command']=None
        instructions=settings.get('instructions')
        if instructions is not None:selection['instructions']=instructions
        write(path/'agent-selection.json',selection)
        terminals.engine()['stop'](path)
        data['phase']='STARTING';data['error']=''
        write(path/'session.json',data)
    run(path)


def status(path, data=None):
    data=dict(data or json.loads((path/'session.json').read_text()))
    terminal=terminals.engine()['capture'](path)
    error=path/'terminal-error.txt'
    data['error']=error.read_text() if error.exists() else data.get('error','')
    if data['phase'] not in ('RELEASED','STOPPED'):
        data['phase']='RUNNING' if terminal.get('status')=='TERMINAL' else ('FAILED' if data['error'] else 'EXITED')
    data['allocation']=dict(active=data['phase']=='RUNNING',host=data['host'],
                            model_state='NATIVE',terminal=terminal,prompt=terminal)
    return data


def control(path, action):
    from .resources import write
    if action=='status':return status(path)
    if action not in ('stop','release'):raise ValueError('Unsupported native session action: '+action)
    with locked(path):
        data=json.loads((path/'session.json').read_text())
        with (path/'session.log').open('a') as log:
            import time
            log.write(json.dumps(dict(event='native_control',time=time.time(),action=action,
                pid=os.getpid(),uid=os.getuid()))+'\n')
        terminals.engine()['stop'](path)
        if data['phase']!='RELEASED':data['phase']='RELEASED' if action=='release' else 'STOPPED'
        write(path/'session.json',data)
    return data

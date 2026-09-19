"""Numbered native CLI sessions, retained in a local tmux terminal (also for SSH)."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
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
    print(f'Session {number}: native {cli} on {host or "local"}. Ctrl+B then D detaches; res-mon / F4 reattaches.',flush=True)
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
    if not getattr(args,'detach',False):terminals.attach(path,data,selection)


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
        terminals.engine()['stop'](path)
        if data['phase']!='RELEASED':data['phase']='RELEASED' if action=='release' else 'STOPPED'
        write(path/'session.json',data)
    return data

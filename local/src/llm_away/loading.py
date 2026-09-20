"""Load a model in the retained terminal before starting its interactive CLI."""
import json
import os
from pathlib import Path
import time
from urllib.request import urlopen


def prepare(path, spec):
    from .resources import rpc, config, write, identity, remote
    from .agents import choose_cli
    from . import terminals
    loading=spec['loading'];cfg=config(loading['config'])
    model=loading['model']
    print('Loading '+model['alias']+'. Ctrl+B then D detaches; F2 reattaches.',flush=True)
    write(path/'attachment.json',dict(client_pid=os.getpid(),client_identity=identity(os.getpid())))
    try:
        if not loading['reuse']:
            rpc(path,'start',model=model,mtp=loading['mtp'],
                server_extra_args=cfg.llamacpp.server_extra_args,client_pid=os.getpid())
        deadline=time.monotonic()+cfg.gateway.startup_timeout_seconds
        previous=None
        while True:
            try:
                with urlopen(f'http://127.0.0.1:{cfg.server.port}/health',timeout=1) as response:
                    if response.status==200:break
            except OSError:pass
            data=rpc(path,'status');allocation=data.get('allocation',{})
            # Only the provider owns startup failure: allocation snapshots can
            # still describe the previous generation immediately after start.
            if data.get('provider_exit') is not None or not allocation.get('active',True):
                raise RuntimeError('Backend stopped; inspect res-mon logs')
            progress=allocation.get('model_state') or data.get('phase') or 'Waiting for model'
            if progress!=previous:print(progress,flush=True);previous=progress
            if time.monotonic()>=deadline:raise TimeoutError('Model startup timed out; inspect res-mon logs')
            time.sleep(1)
        spec['model']=model['alias']
        print('Model ready. Opening CLI.',flush=True)
        if spec.get('target_location')=='remote':
            data=json.loads((path/'session.json').read_text())
            info=remote(cfg,data['token'],data['remote_port'],'agent-info')
            if not info.get('clis'):raise ValueError('Install codex or claude on the compute host')
            spec['cli']=choose_cli(spec['cli'],info['clis'])
            spec['instructions']=cfg.claude.instructions or cfg.codex.instructions if spec['cli']=='claude' else cfg.codex.instructions
            selection=json.loads((path/'agent-selection.json').read_text())
            selection['cli']=spec['cli'];write(path/'agent-selection.json',selection)
            (path/'agent-ready').touch()
            if (path/'resume-requested').exists():spec['resume']=True
            remote_selection={k:v for k,v in spec.items() if k not in ('loading','local','state','target_location')}
            remote_selection['location']='remote'
            terminals.ensure(path,data,remote_selection)
            terminals.attach(path,data,remote_selection)
            return True
        return False
    finally:
        (path/'attachment.json').unlink(missing_ok=True)

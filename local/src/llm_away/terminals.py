"""Local/remote tmux routing for run, Space prompts, and F2."""
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import time


def engine():
    return runpy.run_path(str(Path(__file__).resolve().parents[3]/'remote/bin/resource-terminal'))


def ensure(path,data,selection):
    from .resources import remote,config
    spec=dict(selection,token=data['token'],model=data['model'],resource_session_id=data.get('id'))
    if selection['location']=='local':
        # MCP clients filter inherited environment variables. Pass the caller's
        # identity explicitly to each registered helper through CLI overrides.
        from .serve_registration import registered
        helpers=[]
        for record_path in path.parent.glob('*/serve-registration.json'):
            try:
                record=json.loads(record_path.read_text())
                if registered(record):helpers.append(int(record['session']))
            except (OSError,ValueError,KeyError):continue
        spec['helper_session_ids']=sorted(set(helpers))
        spec.update(local=True,base_url='http://127.0.0.1:'+str(data['config']['server']['port']))
        engine()['start'](path,spec)
    else:
        remote(config(data['config']),data['token'],data['remote_port'],'terminal-start',spec=spec)
        for _ in range(30):
            status=remote(config(data['config']),data['token'],data['remote_port'],'status')
            terminal=status.get('terminal',{})
            if terminal.get('status')=='TERMINAL':return
            if terminal.get('status')=='FAILED':raise ValueError(terminal.get('error','Remote terminal failed'))
            time.sleep(1)
        raise ValueError('Remote terminal not ready; inspect res-mon logs')


def attach(path,data,selection):
    from .resources import config,remote
    if selection['location']=='local':
        e=engine()
        e['configure'](path)
        return subprocess.call(e['tmux'](path)+['attach-session','-t',e['name'](path)])
    cfg=config(data['config']);a=remote(cfg,data['token'],data['remote_port'],'status')
    command=['tmux','-L','llm-away-'+data['token'],'attach-session','-t','away-'+data['token']]
    if cfg.backend_type=='slurm_server':
        command=['srun','--jobid='+str(a['job_id']),'--overlap','--nodes=1','--ntasks=1','--pty',*command]
    elif cfg.backend_type=='kubernetes':
        command=['kubectl',*(['--context',cfg.kubernetes.context] if cfg.kubernetes.context else []),'-n',cfg.kubernetes.namespace,'exec','-it',a['host'],'--',*command]
    if cfg.ssh.connection!='local':command=['ssh','-tt',cfg.ssh.destination,shlex.join(command)]
    return subprocess.call(command)


def send(path,data,text):
    from .resources import remote,config
    selection=json.loads((path/'agent-selection.json').read_text())
    if selection['location']=='local':engine()['send'](path,text)
    else:remote(config(data['config']),data['token'],data['remote_port'],'terminal-send',prompt=text)
    return {'id':data['token'],'status':'TERMINAL'}

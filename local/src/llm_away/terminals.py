"""Local/remote tmux routing for run, Space prompts, and F4."""
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
    spec=dict(selection,token=data['token'],model=data['model'])
    if selection['location']=='local':
        spec.update(local=True,base_url='http://127.0.0.1:'+str(data['config']['server']['port']))
        try:engine()['start'](path,spec)
        except ValueError:selection['foreground']=True
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
        try:return subprocess.call(e['tmux'](path)+['attach-session','-t',e['name'](path)])
        except ValueError:return run_foreground(path,data,selection)
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

"""Local/remote tmux routing for run, Space prompts, and F2."""
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import time


def remote_rag_command(cfg,allocation,rag_config,token):
    """Return a local stdio command whose RAG process runs in the allocation."""
    script=cfg.remote.workdir+'/bin/resource-terminal'
    base=cfg.remote.resource_state_dir or (cfg.remote.workdir+'/.state')
    from .shared_sessions import client_key
    rag_log=base.rstrip('/')+'/resources/'+token+'/clients/'+client_key()+'/rag.log'
    spec=json.dumps({'allocation_kind':cfg.backend_type,'rag_config':rag_config,
                     'rag_log_path':rag_log},separators=(',',':'))
    if cfg.backend_type=='slurm_server':
        job=allocation.get('job_id')
        if not job:raise ValueError('Cannot start RAG before the Slurm allocation is ready')
        command=['srun','--jobid='+str(job),'--overlap','--nodes=1','--ntasks=1',script,'rag-stdio',spec]
    elif cfg.backend_type=='kubernetes':
        pod=allocation.get('host')
        if not pod:raise ValueError('Cannot start RAG before the Kubernetes pod is ready')
        command=['kubectl',*(['--context',cfg.kubernetes.context] if cfg.kubernetes.context else []),
                 '-n',cfg.kubernetes.namespace,'exec','-i',pod,'--',script,'rag-stdio',spec]
    elif cfg.backend_type=='direct':command=[script,'rag-stdio',spec]
    else:raise ValueError('Remote RAG bridge requires a Slurm or Kubernetes allocation')
    if cfg.ssh.connection!='local':
        command=['ssh','-x','-o','BatchMode=yes',
                 '-o','ConnectTimeout='+str(cfg.ssh.connect_timeout_seconds),cfg.ssh.destination,shlex.join(command)]
    return command


def allocation_command(cfg,allocation,command):
    if cfg.backend_type=='slurm_server':
        command=['srun','--jobid='+str(allocation['job_id']),'--overlap','--nodes=1','--ntasks=1',*command]
    elif cfg.backend_type=='kubernetes':
        command=['kubectl',*(['--context',cfg.kubernetes.context] if cfg.kubernetes.context else []),
                 '-n',cfg.kubernetes.namespace,'exec','-i',allocation['host'],'--',*command]
    elif cfg.backend_type=='direct':pass
    else:raise ValueError('RAG bridge requires Slurm or Kubernetes')
    if cfg.ssh.connection!='local':
        command=['ssh','-x','-o','BatchMode=yes',
                 '-o','ConnectTimeout='+str(cfg.ssh.connect_timeout_seconds),cfg.ssh.destination,shlex.join(command)]
    return command


def local_rag_for_remote_agent(path,data,cfg,allocation,rag_command):
    """Keep RAG local and expose its stdio to an agent inside the allocation."""
    state_base=cfg.remote.resource_state_dir or (cfg.remote.workdir+'/.state')
    from .shared_sessions import client_key
    socket_path=state_base.rstrip('/')+'/resources/'+data['token']+'/rag-'+client_key()+'.sock'
    script=cfg.remote.workdir+'/bin/resource-terminal'
    server=allocation_command(cfg,allocation,[script,'rag-socket-server',socket_path])
    log=(Path(path)/'rag-bridge.log').open('a')
    process=subprocess.Popen([sys.executable,'-m','llm_away.rag_bridge',json.dumps(rag_command),json.dumps(server)],
                             stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    log.close()
    record=Path(path)/'rag-bridge.json';record.write_text(json.dumps({'pid':process.pid}));record.chmod(0o600)
    return [script,'rag-socket-client',socket_path]


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
    if selection.get('loading') and not (path/'agent-ready').exists():
        (path/'resume-requested').touch()
    if selection['location']=='local':
        e=engine()
        e['configure'](path)
        return subprocess.call(e['tmux'](path)+['attach-session','-t',e['name'](path)])
    cfg=config(data['config']);a=remote(cfg,data['token'],data['remote_port'],'status')
    from .shared_sessions import client_key
    terminal_id=data['token']+'-'+client_key()
    command=['tmux','-L','llm-away-'+terminal_id,'attach-session','-t','away-'+terminal_id]
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

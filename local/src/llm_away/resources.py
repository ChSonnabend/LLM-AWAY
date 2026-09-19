"""Numbered background allocations with isolated providers and foreground agents."""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import asdict, is_dataclass, replace
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import threading
import uuid
from urllib.request import urlopen

from .config import AppConfig, load_config
from .backends import SlurmServerBackend, KubernetesBackend, BackendError
from .models import discover_models, choose_model, choose_mtp
from .onboarding import configure_local, ask
from .prompt import choose_option
from .agents import choose_cli, claude_launch
from .cli import remote_model_catalog

ROOT=Path(__file__).resolve().parents[2]
STORE=ROOT/'run'/'resources'

def write(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2));temp.chmod(0o600);temp.replace(path)

def config(data):
    base=AppConfig()
    return replace(base,**{key:type(getattr(base,key))(**value) if is_dataclass(getattr(base,key)) else value
                          for key,value in data.items() if key not in ('hosts','saved_hosts')})

def remote(cfg, token, port, action, **extra):
    command=['python3',cfg.remote.workdir+'/bin/resource-control',action]
    if cfg.ssh.connection!='local':
        # Reuse one authenticated connection for repeated allocation polls.
        # Without multiplexing, every 5-second poll performs a new SSH and
        # ProxyJump handshake; transient banner timeouts then appear as errors.
        options=['-x','-o','BatchMode=yes','-o',f'ConnectTimeout={cfg.ssh.connect_timeout_seconds}',
                 '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2']
        control=os.environ.get('LLM_AWAY_SSH_CONTROL')
        if control:
            options += ['-o','ControlMaster=auto','-o',f'ControlPath={control}','-o','ControlPersist=60']
        command=['ssh',*options,cfg.ssh.destination,shlex.join(command)]
    result=subprocess.run(command,input=json.dumps(dict(config=asdict(cfg),token=token,port=port,**extra)),
                          text=True,capture_output=True,timeout=120)
    if result.returncode: raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)

def identity(pid):
    try:return subprocess.check_output(['ps','-p',str(pid),'-o','lstart='],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.SubprocessError):return ''

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]

class AdoptedProcess:
    """Minimal Popen stand-in for a provider started by a previous daemon."""
    def __init__(self,pid):
        self.pid=pid;self.returncode=None
    def poll(self):
        if self.returncode is not None:return self.returncode
        try:os.kill(self.pid,0)
        except ProcessLookupError:self.returncode=0
        except PermissionError:pass
        return self.returncode
    def terminate(self):
        try:os.kill(self.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    def kill(self):
        try:os.kill(self.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    def wait(self,timeout=None):
        deadline=time.time()+(timeout or 0)
        while self.poll() is None:
            if timeout is not None and time.time()>=deadline:
                raise subprocess.TimeoutExpired(self.pid,timeout)
            time.sleep(0.1)
        return self.returncode

def path_for(number):
    if not str(number).isdigit(): raise ValueError('Session ID must be a number')
    path=STORE/str(int(number))
    if not (path/'session.json').exists(): raise ValueError('Unknown session '+str(number))
    return path

def released_ids(store=STORE):
    """Session IDs that are explicitly released and safe to archive/delete."""
    released=[]
    for path in store.glob('[0-9]*/session.json'):
        try:data=json.loads(path.read_text())
        except (OSError,ValueError):continue
        if data.get('phase')=='RELEASED' and not (path.parent/'control.sock').exists():
            released.append(int(path.parent.name))
    return released

def rpc(path, action, **extra):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(130);s.connect(str(path/'control.sock'))
        s.sendall((json.dumps(dict(action=action,**extra))+'\n').encode())
        data=b''
        while not data.endswith(b'\n'):
            part=s.recv(65536)
            if not part: raise RuntimeError('Resource process disconnected')
            data+=part
    result=json.loads(data)
    if 'rpc_error' in result: raise RuntimeError(result['rpc_error'])
    return result

def rpc_alive(path):
    try:
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(1);s.connect(str(path/'control.sock'))
            s.sendall(b'{"action":"status"}\n')
            return bool(s.recv(16))
    except OSError:return False

class ResourceBackend(SlurmServerBackend):
    def __init__(self,cfg,token,port):
        super().__init__(cfg);self.token=token;self.port=port
    def serverctl(self,command,model,extra=None):
        return remote(self.config,self.token,self.port,{'ensure':'start','cancel':'stop'}.get(command,command),**(extra or {}))
    def ensure_ready(self,model):
        with self._ready_lock:
            if self._closing.is_set(): raise BackendError('Model is stopping')
            if self._ready_model==model and self.http_ready(model): return
            state=self.serverctl('ensure',model)
            deadline=time.time()+self.config.gateway.startup_timeout_seconds
            while time.time()<deadline and not self._closing.is_set():
                state=self.serverctl('status',model)
                if not state.get('active'): raise BackendError('Resource allocation ended')
                if state.get('model_state')=='EXITED': raise BackendError('Model exited; see res-mon logs')
                if state.get('host') and state.get('model_state')=='LOADING':
                    self.ensure_tunnel(state['host'])
                    if self.http_ready(model): self._ready_model=model;return
                self._closing.wait(2)
            raise BackendError('Model startup timed out or was canceled')
    def ensure_tunnel(self,host):
        if self.config.backend_type=='kubernetes': KubernetesBackend.ensure_tunnel(self,host)
        else: super().ensure_tunnel(host)
        if self._tunnel and os.environ.get('LLM_SESSION_DIR'):
            write(Path(os.environ['LLM_SESSION_DIR'])/'tunnel.json',
                  dict(pid=self._tunnel.pid,identity=identity(self._tunnel.pid)))
    def close(self):
        self._closing.set()
        if self._tunnel:
            self._tunnel.terminate()
            try:self._tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:self._tunnel.kill();self._tunnel.wait()
            self._tunnel=None


def provider(path):
    from .server import serve
    data=json.loads((path/'session.json').read_text());cfg=config(data['config'])
    settings=Path(os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/model.toml')))
    if settings.exists():cfg=replace(cfg,codex=load_config(settings).codex)
    os.environ['LLM_TOOL_ERROR_DIR']=str(path/'tool-errors')
    os.environ['LLM_SESSION_DIR']=str(path)
    backend=ResourceBackend(cfg,data['token'],data['remote_port'])
    def stop(sig,frame):
        backend.close();raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    try:
        backend.ensure_ready(cfg.model.name)
        print('Model ready: '+cfg.model.name,flush=True)
        serve(cfg,backend,warm=False)
    finally: backend.close()


def daemon(path):
    data=json.loads((path/'session.json').read_text());cfg=config(data['config'])
    token=data['token'];port=data['remote_port'];child=None;offset=0;last=None
    if cfg.ssh.connection!='local':
        os.environ['LLM_AWAY_SSH_CONTROL']=str((path/'ssh.sock').resolve())
    if (path/'control.sock').exists():
        if rpc_alive(path):raise RuntimeError('Daemon already owns this session')
        (path/'control.sock').unlink(missing_ok=True)
    pid=data.get('provider_pid');born=data.get('provider_identity')
    if pid and born and identity(pid)==born:
        # Adopt a live provider after a daemon restart so status and stop remain accurate.
        child=AdoptedProcess(pid)
    sock=socket.socket(socket.AF_UNIX);sock.bind(str(path/'control.sock'));os.chmod(path/'control.sock',0o600)
    sock.listen(4);sock.settimeout(1)
    def state(**kwargs):
        data.update(kwargs);write(path/'session.json',data)
    def stop_model(caller=None):
        nonlocal child,cfg
        from .serve_registration import remove
        from .terminals import engine
        engine()['stop'](path)
        try:remove(path)
        except (OSError,ValueError) as exc:print(f'MCP cleanup: {exc}',file=sys.stderr)
        current=data
        try:current=json.loads((path/'attachment.json').read_text())
        except (OSError,ValueError):pass
        client=current.get('client_pid')
        if client and client!=caller and current.get('client_identity') and identity(client)==current['client_identity']:
            try:os.kill(client,signal.SIGTERM)
            except ProcessLookupError:pass
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        child=None
        remote(cfg,token,port,'stop')
        deadline=time.time()+20
        while time.time()<deadline:
            s=remote(cfg,token,port,'status')
            if not s.get('active') or s.get('model_state') in ('IDLE','STOPPED'):
                state(model='',client_pid=None,client_identity='',provider_pid=None,provider_identity='');return
            time.sleep(1)
        raise RuntimeError('Model stop not acknowledged; inspect res-mon logs')
    state(pid=os.getpid(),phase='RUNNING',error='')
    if child is None:
        try:
            allocation=remote(cfg,token,port,'reserve');state(allocation=allocation,phase=allocation['slurm_state'])
            print('Allocation:',allocation,flush=True)
        except Exception as exc:
            state(phase='ERROR',error=str(exc));print(exc,flush=True)
    else:
        print('Adopted live provider '+str(child.pid)+'; skipping reserve',flush=True)
    release_requested=False
    def interrupted(sig,frame):
        nonlocal release_requested
        release_requested=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    tick=0;remote_poll=None
    from concurrent.futures import ThreadPoolExecutor
    remote_pool=ThreadPoolExecutor(max_workers=1)
    def poll_remote():
        s=remote(cfg,token,port,'status')
        logs=remote(cfg,token,port,'log',offset=offset)
        return s,logs['offset'],logs['data']
    def provider_ready():
        # The scheduler only knows "model process started" (LOADING). Confirm the
        # isolated local provider endpoint is healthy before displaying LOADED.
        try:
            with urlopen(f'http://127.0.0.1:{cfg.server.port}/health',timeout=1) as response:
                return response.status==200
        except OSError:
            return False
    try:
        while True:
            if release_requested:
                stop_model();remote(cfg,token,port,'release');state(phase='RELEASED');break
            try:client,_=sock.accept()
            except socket.timeout:client=None
            if client:
                with client:
                    client.settimeout(5)
                    action=''
                    try:
                        request=json.loads(client.makefile('rb').readline(1048576));action=request['action']
                        if action=='status': answer=data
                        elif action=='start':
                            if child is not None and child.poll() is None: raise ValueError('Session already has a running agent/provider')
                            allocation=remote(cfg,token,port,'status')
                            if not allocation.get('active'): raise ValueError('Allocation is not active')
                            cfg=replace(cfg,model=replace(cfg.model,name=request['model']['alias']),
                                llamacpp=replace(cfg.llamacpp,model_name=request['model']['name'],mtp=request['mtp'],
                                                 server_extra_args=request.get('server_extra_args',cfg.llamacpp.server_extra_args)))
                            if request['model'].get('context_size',0)>0:
                                cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,context_size=request['model']['context_size']))
                            # A preset without a draft model cannot run MTP; enforce it locally too.
                            if request['mtp']=='on' and not request['model'].get('mtp',{}).get('configured'):
                                request=dict(request);request['mtp']='off'
                                cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,mtp='off'))
                            state(config=asdict(cfg),model=cfg.model.name,provider_exit=None,client_pid=request.get('client_pid'),client_identity=identity(request.get('client_pid')))
                            child=subprocess.Popen([sys.executable,'-m','llm_away.resources','provider',str(path)],stdin=subprocess.DEVNULL)
                            state(provider_pid=child.pid,provider_identity=identity(child.pid))
                            answer={'port':cfg.server.port}
                        elif action in ('stop','release'):
                            stop_model(request.get('client_pid'))
                            if action=='release':remote(cfg,token,port,'release');state(phase='RELEASED')
                            answer={'ok':True}
                        else:raise ValueError('Unknown operation')
                    except Exception as exc:answer={'rpc_error':str(exc)}
                    client.sendall((json.dumps(answer)+'\n').encode())
                    if action=='release' and 'rpc_error' not in answer:break
            now=time.time()
            if now-tick>=5:
                tick=time.time()
                if remote_poll and not remote_poll.done():
                    print('Resource monitor: previous remote poll still in progress; skipping',flush=True)
                    continue
                remote_poll=remote_pool.submit(poll_remote)
                try:
                    s,new_offset,new_logs=remote_poll.result()
                    offset=new_offset
                    if child is not None and child.poll() is None and s.get('model_state')=='LOADING' and provider_ready():
                        s['model_state']='LOADED'
                    state(allocation=s,phase=s['slurm_state'],error='',provider_exit=child.poll() if child else None)
                    summary=(s['slurm_state'],s.get('model_state'),s.get('host'))
                    if summary!=last:print('Resource state:',summary,flush=True);last=summary
                    if new_logs:print(new_logs,end='',flush=True)
                except Exception as exc:state(error=str(exc));print('Monitor:',exc,flush=True)
    finally:
        if child is not None and child.poll() is None:child.terminate()
        sock.close();(path/'control.sock').unlink(missing_ok=True)
        remote_pool.shutdown(wait=False,cancel_futures=True)
        os.environ.pop('LLM_AWAY_SSH_CONTROL',None)


def allocate(args):
    STORE.mkdir(parents=True,exist_ok=True,mode=0o700)
    with open(STORE/'registry.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        configure_local(args.config,alias=args.host,connection=args.connection,restart=args.restart,resources_only=True)
        cfg=load_config(args.config)
        gpus=args.gpus if args.gpus is not None else int(ask('GPUs for this allocation',str(cfg.slurm.gpus or (0 if cfg.llamacpp.backend=='cpu' else 1))))
        if gpus<0 or (cfg.llamacpp.backend!='cpu' and gpus<1): raise ValueError('Invalid GPU count')
        cfg=replace(cfg,slurm=replace(cfg.slurm,gpus=gpus,nodes=1))
        if cfg.backend_type=='slurm_server':
            options=shlex.split(ask('Additional Slurm options for this allocation',shlex.join(cfg.slurm.custom_options)))
            cfg=replace(cfg,slurm=replace(cfg.slurm,custom_options=options))
        elif cfg.backend_type=='kubernetes':
            k=cfg.kubernetes
            selector=json.loads(ask('Kubernetes node selector (JSON)',json.dumps(k.node_selector)))
            tolerations=json.loads(ask('Kubernetes tolerations (JSON)',json.dumps(k.tolerations)))
            if not isinstance(selector,dict) or any(not isinstance(v,str) for v in selector.values()):raise ValueError('Node selector must map keys to strings')
            if not isinstance(tolerations,list) or any(not isinstance(v,dict) for v in tolerations):raise ValueError('Tolerations must be a JSON array of objects')
            cfg=replace(cfg,kubernetes=replace(k,node_selector=selector,tolerations=tolerations,
                cpu=ask('Kubernetes CPUs',k.cpu),memory=ask('Kubernetes memory',k.memory),
                priority_class=ask('Kubernetes priority class (blank for default)',k.priority_class),
                time_limit_seconds=int(ask('Kubernetes time limit in seconds (0: unlimited)',str(k.time_limit_seconds)))))
            if cfg.kubernetes.time_limit_seconds<0:raise ValueError('Time limit must not be negative')
        if cfg.backend_type=='direct' and gpus:
            devices=ask('GPU device IDs for this session',cfg.llamacpp.visible_devices or ','.join(map(str,range(gpus))))
            if len(devices.split(','))!=gpus or any(not n.isdigit() for n in devices.split(',')) or len(set(devices.split(',')))!=gpus:raise ValueError('Choose one distinct device ID per GPU')
            cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,visible_devices=devices))
            print('Direct mode selects devices; it cannot reserve them against other users.')
        if cfg.backend_type=='direct' and gpus:
            requested=set(cfg.llamacpp.visible_devices.split(','))
            for saved in STORE.glob('*/session.json'):
                old=json.loads(saved.read_text())
                if old.get('phase')=='RELEASED':continue
                other=config(old['config'])
                if other.backend_type=='direct' and other.ssh.destination==cfg.ssh.destination and requested.intersection(other.llamacpp.visible_devices.split(',')):
                    raise ValueError('GPU device overlap with resource session '+str(old['id'])+'; release it or select other devices')
        number=1+max([int(p.name) for p in STORE.iterdir() if p.name.isdigit()]+released_ids()+[0])
        path=STORE/str(number);path.mkdir(mode=0o700)
        ports=set()
        for saved in STORE.glob('*/session.json'):
            old=json.loads(saved.read_text())
            ports.update([old['remote_port'],old['config']['server']['port'],old['config']['gateway']['local_port']])
        remote_port=free_port()
        while remote_port in ports:remote_port=free_port()
        provider_port=free_port()
        while provider_port in ports or provider_port==remote_port:provider_port=free_port()
        tunnel_port=free_port()
        while tunnel_port in ports or tunnel_port in (remote_port,provider_port):tunnel_port=free_port()
        cfg=replace(cfg,server=replace(cfg.server,host='127.0.0.1',port=provider_port),
            gateway=replace(cfg.gateway,server_port=remote_port,local_port=tunnel_port,cancel_on_exit=False,cancel_reused_on_exit=False),
            hosts={},saved_hosts={},active_host='')
        write(path/'session.json',dict(id=number,token=uuid.uuid4().hex,config=asdict(cfg),remote_port=remote_port,
              host=cfg.ssh.destination,gpus=gpus,phase='STARTING',model=''))
        with (path/'session.log').open('a') as log:
            subprocess.Popen([sys.executable,'-m','llm_away.resources','daemon',str(path)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    print(f'Session {number} allocating in background. Use: res-mon --logs {number}\nThen: run --session {number}')


def run_agent(args):
    path=path_for(args.session)
    from . import terminals
    selection_path=path/'agent-selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text())
        data=json.loads((path/'session.json').read_text())
        existing=(terminals.engine()['alive'](path) if selection.get('location')=='local' else
                  remote(config(data['config']),data['token'],data['remote_port'],'status').get('terminal',{}).get('status')=='TERMINAL')
        if existing:
            if args.model and args.model not in (data.get('model'),data['config']['llamacpp'].get('model_name')):raise ValueError('Exit the agent terminal before switching models')
            if getattr(args,'agent_location','local')!=selection.get('location') or (args.cli and args.cli not in ('auto',selection.get('cli'))) or args.agent_workdir:
                raise ValueError('An agent terminal already exists; exit it before changing its configuration')
            if not args.serve:terminals.attach(path,data,selection)
            else:print('Agent terminal is already running; F4 attaches, Ctrl+B then D detaches.')
            return
    # Advisory lease prevents two foreground clients from sharing one model slot.
    with open(path/'client.lock','a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another run command owns this session')
        try:data=rpc(path,'status')
        except OSError:
            # A daemon can exit while its provider remains live. Restart it and
            # let daemon() adopt the provider before attaching the agent.
            with (path/'session.log').open('a') as log:
                subprocess.Popen([sys.executable,'-m','llm_away.resources','daemon',str(path)],
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            deadline=time.monotonic()+5
            while True:
                try:data=rpc(path,'status');break
                except OSError:
                    if time.monotonic()>=deadline:raise
                    time.sleep(0.2)
        if data.get('allocation',{}).get('prompt',{}).get('status') in ('QUEUED','RUNNING'):
            raise ValueError('A remote prompt is running; view its progress in res-mon before attaching')
        cfg=config(data['config'])
        settings=Path(os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/model.toml')))
        if settings.exists():
            current=load_config(settings)
            cfg=replace(cfg,codex=current.codex,agent=current.agent,claude=current.claude)
        reuse=False;resume=False
        loaded=bool(data.get('model') and data.get('provider_exit') is None and
                    data.get('provider_identity') and identity(data.get('provider_pid'))==data['provider_identity'])
        if loaded:
            reuse=(not args.model or args.model in (data['model'],cfg.llamacpp.model_name))
            if not args.model and not args.serve:
                reuse=choose_option([f"Keep loaded model: {data['model']}",'Load a different model'],'Model already running')==0
            if reuse:
                if args.serve:pass
                elif getattr(args,'resume',False):resume=True
                else:
                    mode=choose_option(['Reopen agent (saved-conversation picker)','Keep running in background'],
                                       'Session mode',default=1 if args.serve else 0)
                    args.serve=mode==1;resume=mode==0
        if reuse:
            model=dict(alias=data['model'],name=cfg.llamacpp.model_name,context_size=cfg.llamacpp.context_size)
        else:
            model=choose_model(discover_models(cfg),cfg.llamacpp.model_name,args.model)
        # Explicit preset context also applies to allocations created before the preset.
        if model.get('context_size',0)>0:
            cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,context_size=model['context_size']),
                        codex=replace(cfg.codex,context_window=model['context_size']))
        location=getattr(args,'agent_location','local')
        if location=='remote' and (args.rag or args.agent_args):
            raise ValueError('Remote mode accepts prompts via its console or res-mon; local --rag and extra CLI arguments are not supported')
        preference=getattr(args,'cli',None) or os.environ.get('LLM_AWAY_CLI') or cfg.agent.cli
        selected_cli=choose_cli(preference) if location=='local' else None
        mtp=cfg.llamacpp.mtp if reuse else choose_mtp(model,cfg.llamacpp.mtp,args.mtp)
        if not reuse and sys.stdin.isatty():
            cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,server_extra_args=shlex.split(
                ask('Additional llama.cpp server options (blank uses saved options)',
                    shlex.join(cfg.llamacpp.server_extra_args)))))
        rag_command=None
        if args.rag:
            from .rag import prepare
            rag_command=prepare(args.rag)
        started=False
        agent=None
        external_stop=False
        detached=False
        def interrupted(sig,frame):
            nonlocal external_stop
            external_stop=True
            if agent and agent.poll() is None:agent.terminate()
            raise KeyboardInterrupt
        def hangup(sig,frame):
            nonlocal detached
            detached=True
            raise KeyboardInterrupt
        previous=signal.signal(signal.SIGTERM,interrupted)
        previous_hup=signal.signal(signal.SIGHUP,hangup)
        try:
            if not reuse:
                if loaded:rpc(path,'stop',client_pid=os.getpid())
                rpc(path,'start',model=model,mtp=mtp,server_extra_args=cfg.llamacpp.server_extra_args,client_pid=os.getpid())
            started=True
            write(path/'attachment.json',dict(client_pid=os.getpid(),client_identity=identity(os.getpid())))
            deadline=time.time()+cfg.gateway.startup_timeout_seconds
            print(f'Loading model; telemetry: res-mon --logs {args.session}')
            while True:
                try:
                    with urlopen(f'http://127.0.0.1:{cfg.server.port}/health',timeout=1) as r:
                        if r.status==200:break
                except OSError:pass
                s=rpc(path,'status')
                if s.get('provider_exit') is not None or not s.get('allocation',{}).get('active',True) or s.get('allocation',{}).get('model_state')=='EXITED':raise RuntimeError('Backend stopped; inspect session log')
                if time.time()>deadline:raise TimeoutError('Model startup timed out')
                time.sleep(1)
            if location=='remote':
                info=remote(cfg,data['token'],data['remote_port'],'agent-info')
                if not info.get('clis'):raise ValueError('Install codex or claude on the compute host and start an updated allocation before selecting remote agents')
                selected_cli=choose_cli(preference,info['clis'])
            agent_cwd=getattr(args,'agent_workdir',None) or (os.getcwd() if location=='local' else '')
            if location=='remote' and agent_cwd and not agent_cwd.startswith('/'):
                raise ValueError('--agent-workdir must be an absolute remote path')
            selection={'location':location,'cli':selected_cli,'cwd':agent_cwd,'extra_args':args.agent_args,'rag_command':rag_command,
                       'instructions':cfg.claude.instructions or cfg.codex.instructions if selected_cli=='claude' else cfg.codex.instructions}
            if selected_cli=='codex' and cfg.codex.custom_metadata:
                selection['codex_catalog_content']=json.dumps(remote_model_catalog(
                    model,context_window=cfg.codex.context_window,settings=cfg.codex
                ),indent=2)+'\n'
            write(path/'agent-selection.json',selection)
            current=json.loads((path/'session.json').read_text())
            terminals.ensure(path,current,selection)
            if selection.get('foreground'):
                print('tmux unavailable; running the agent in this terminal. Closing it stops the agent.',flush=True)
            else:
                detached=True
                # The pane worker takes ownership of the agent lease after this handoff.
                fcntl.flock(lease,fcntl.LOCK_UN)
                print('Agent terminal running. F4 attaches; Ctrl+B then D detaches.')
            if not args.serve:terminals.attach(path,current,selection)
            return
        except KeyboardInterrupt:pass
        except Exception as exc:
            print(f'Launch failed: {exc}; see res-mon --logs {args.session}',file=sys.stderr,flush=True)
            raise
        finally:
            if agent and agent.poll() is None:
                agent.terminate()
                try:agent.wait(timeout=5)
                except subprocess.TimeoutExpired:agent.kill();agent.wait()
            signal.signal(signal.SIGTERM,previous)
            signal.signal(signal.SIGHUP,previous_hup)
            if started:(path/'attachment.json').unlink(missing_ok=True)
            if started and not external_stop and not detached:
                try:
                    choice=choose_option(['Keep model running in background','Unload model, keep allocation','Release allocation and stop background session'],'Session finished') if sys.stdin.isatty() else 0
                except (ValueError,KeyboardInterrupt):choice=0
                if choice:
                    from .serve_registration import remove
                    try:remove(path)
                    except (OSError,ValueError) as exc:print(f'MCP cleanup: {exc}',file=sys.stderr)
                    rpc(path,'stop' if choice==1 else 'release',client_pid=os.getpid())
                print(['Model and allocation retained.','Allocation retained.','Resources released.'][choice])


def release_session(number):
    path=path_for(number)
    data=json.loads((path/'session.json').read_text())
    # Stop the foreground run wrapper (which terminates its Codex child) first.
    current=data
    try:current=json.loads((path/'attachment.json').read_text())
    except (OSError,ValueError):pass
    pid=current.get('client_pid');born=current.get('client_identity')
    if pid and born and identity(pid)==born:
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
        deadline=time.monotonic()+7
        while time.monotonic()<deadline and identity(pid)==born:time.sleep(0.2)
        if identity(pid)==born:raise RuntimeError('Agent did not stop; allocation retained. Stop the agent and retry.')
    try:rpc(path,'release')
    except (ConnectionRefusedError,FileNotFoundError):
        pid=data.get('provider_pid');born=data.get('provider_identity')
        if pid and born and identity(pid)==born:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        release_error=''
        try:remote(config(data['config']),data['token'],data['remote_port'],'release')
        except Exception as exc:
            # The user explicitly asked to release this session and the local
            # daemon is gone. Preserve the remote error, but do not leave a
            # daemon-less allocation stuck in the monitor when SSH is flaky.
            release_error=str(exc)
        data['phase']='RELEASED';data['model']='';write(path/'session.json',data)
        if release_error:raise RuntimeError('Remote release failed; local session released. '+release_error) from None


def submit_prompt(number, text, location=None, cli=None, cwd=None):
    from . import terminals
    path=path_for(number)
    data=json.loads((path/'session.json').read_text())
    saved=path/'agent-selection.json'
    if not saved.exists():raise ValueError('Open the agent first with run or res-background')
    selection=json.loads(saved.read_text())
    if (location and location!=selection['location']) or (cli and cli!=selection['cli']) or (cwd and cwd!=selection.get('cwd')):
        raise ValueError('Use the existing terminal settings, or exit the agent before reconfiguring')
    if not text.strip():raise ValueError('Prompt is empty')
    return terminals.send(path,data,text)


def monitor(args):
    if args.logs:
        path=path_for(args.logs);subprocess.call(['tail','-n','60','-f',str(path/'session.log')]);return
    if not args.list and not args.kill and sys.stdin.isatty():
        if not sys.stdout.isatty():
            # Some terminal wrappers leave stdout piped while stdin remains the
            # controlling TTY. Curses needs the TTY, so attach stdout to it.
            try:
                tty=os.open('/dev/tty',os.O_WRONLY);os.dup2(tty,1);os.close(tty)
            except OSError:pass
        if not sys.stdout.isatty():return
        from .monitor_ui import show
        chosen=show(STORE,release_session,run_agent,submit_prompt)
        if chosen is not None:
            # Re-exec the run wrapper: guarantees a clean terminal handoff to Codex.
            selection=path_for(chosen)/'agent-selection.json'
            saved=json.loads(selection.read_text()) if selection.exists() else {}
            command=[sys.executable,'-m','llm_away.resources','run','--resume','--session',str(chosen),'--agent-location',saved.get('location','local')]
            if saved.get('cli'):command+=['--cli',saved['cli']]
            if saved.get('cwd'):command+=['--agent-workdir',saved['cwd']]
            raise SystemExit(subprocess.call(command))
        return
    entries=[]
    for p in sorted(STORE.glob('*/session.json'),key=lambda p:int(p.parent.name)):
        data=json.loads(p.read_text())
        if data.get('phase')=='RELEASED':continue
        try:data=rpc(p.parent,'status')
        except (OSError,RuntimeError):data['phase']='DAEMON OFFLINE'
        entries.append(data)
    if entries:
        rows=[['ID','HOST','GPUS','STATE','MODEL','JOB','ERROR']]
        for data in entries:
            rows.append([str(data['id']),data['host'],str(data['gpus']),data['phase'],
                         data.get('model') or '-',str(data.get('allocation',{}).get('job_id') or '-'),
                         data.get('error') or ''])
        widths=[max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
        header=rows[0]
        print('\x1b[1;94m'+'    '.join(header[col].ljust(widths[col]) for col in range(len(header)))+'\x1b[0m')
        for data,row in zip(entries,rows[1:]):
            phase=row[3].upper()
            color={'RUNNING':'\x1b[1;92m','READY':'\x1b[1;92m','LOADING':'\x1b[1;93m','ALLOCATING':'\x1b[1;93m'}.get(
                phase,'\x1b[1;91m' if ('ERROR' in phase or 'OFFLINE' in phase or 'EXIT' in phase) else '\x1b[36m')
            cells=[row[col].ljust(widths[col]) for col in range(len(row))]
            print('    '.join(cells[:3])+f'    {color}{cells[3]}\x1b[0m    '+'    '.join(cells[4:]))
        print()
    number=args.kill
    if not number and not args.list and entries and sys.stdin.isatty():
        idx=choose_option(['Exit']+[str(x['id'])+' — '+x['host'] for x in entries],'Manage session')
        if not idx:return
        number=entries[idx-1]['id']
    if number:
        path=path_for(number)
        release=args.release or (sys.stdin.isatty() and choose_option(['Stop model, keep allocation','Release resources and stop background session'],'Stop session')==1)
        if release:
            release_session(number)
        else:rpc(path,'stop')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    alloc=sub.add_parser('allocate');alloc.add_argument('--config',default=os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/model.toml')))
    alloc.add_argument('--host');alloc.add_argument('--connection',choices=['ssh','local']);alloc.add_argument('--restart',action='store_true');alloc.add_argument('--gpus',type=int)
    run=sub.add_parser('run');run.add_argument('--session','-s',required=True,type=int);run.add_argument('--model');run.add_argument('--mtp',choices=['auto','on','off']);run.add_argument('--rag',action='append',metavar='FOLDER',help='Local code/docs folder; repeat for multiple folders');run.add_argument('agent_args',nargs=argparse.REMAINDER)
    run.add_argument('--agent-location',choices=['local','remote'],default='local')
    run.add_argument('--agent-workdir',help='Project directory on the selected agent host')
    run.add_argument('--cli',choices=['auto','codex','claude'],help='Agent CLI; auto asks only when both are installed')
    run.add_argument('--serve',action='store_true',help='Keep model loaded for MCP delegation without launching an agent')
    run.add_argument('--resume',action='store_true',help='Reconnect directly to the saved agent conversation when a model is already loaded')
    prompt=sub.add_parser('prompt');prompt.add_argument('--session','-s',required=True,type=int);prompt.add_argument('text')
    prompt.add_argument('--agent-location',choices=['local','remote']);prompt.add_argument('--cli',choices=['codex','claude']);prompt.add_argument('--agent-workdir')
    mon=sub.add_parser('monitor');mon.add_argument('--list',action='store_true');mon.add_argument('--logs',type=int);mon.add_argument('--kill',type=int);mon.add_argument('--release',action='store_true')
    for name in ('daemon','provider'):sub.add_parser(name).add_argument('path',type=Path)
    args=parser.parse_args()
    try:
        if args.command=='allocate':allocate(args)
        elif args.command=='run':
            if args.agent_args[:1]==['--']:args.agent_args=args.agent_args[1:]
            run_agent(args)
        elif args.command=='prompt':print(json.dumps(submit_prompt(args.session,args.text,args.agent_location,args.cli,args.agent_workdir)))
        elif args.command=='monitor':monitor(args)
        elif args.command=='daemon':daemon(args.path)
        else:provider(args.path)
    except (ValueError,RuntimeError,OSError,subprocess.SubprocessError,KeyboardInterrupt) as exc:
        print(str(exc) or 'Canceled',file=sys.stderr);return 1
    return 0

if __name__=='__main__':sys.exit(main())

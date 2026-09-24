"""Numbered background allocations with isolated providers and foreground agents."""
from __future__ import annotations
import argparse
from argparse import Namespace
import contextlib
from dataclasses import asdict, is_dataclass, replace
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import threading
import uuid
from urllib.request import Request, urlopen

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

def allocation_pending(status):
    """True while a scheduler allocation has no compute node to run a model."""
    return status.get('slurm_state') in ('PENDING','CONFIGURING') or not status.get('host')

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
    saved=path/'session.json'
    if saved.exists() and json.loads(saved.read_text()).get('native'):
        from .native_sessions import control
        return control(path,action)
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
            while not self._closing.is_set():
                state=self.serverctl('status',model)
                if not state.get('active'): raise BackendError('Resource allocation ended')
                if state.get('model_state')=='EXITED': raise BackendError('Model exited; see res-mon logs')
                if state.get('slurm_state') in ('PENDING','CONFIGURING') or not state.get('host'):
                    # Scheduler queue time is not model startup time. The desired
                    # model is already recorded and the worker starts it once the
                    # allocation receives a node.
                    deadline=time.time()+self.config.gateway.startup_timeout_seconds
                if state.get('host') and state.get('model_state')=='LOADING':
                    self.ensure_tunnel(state['host'])
                    if self.http_ready(model): self._ready_model=model;return
                if time.time()>=deadline:break
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


SLURM_ENDED = {'COMPLETED','CANCELLED','TIMEOUT','FAILED','OUT_OF_MEMORY',
               'NODE_FAIL','BOOT_FAIL','DEADLINE','REVOKED'}


def allocation_ended(cfg, status):
    phase=str(status.get('slurm_state','')).split(' ',1)[0].rstrip('+')
    return cfg.backend_type=='slurm_server' and status.get('active') is False and phase in SLURM_ENDED


def cleanup_ended_allocation(path, data):
    """Retire local dependents, keeping scheduler state, logs and conversations."""
    from .terminals import engine
    from .serve_registration import remove
    from .session_guard import stop_process
    errors=[]
    def attempt(fn):
        try:fn()
        except Exception as exc:errors.append(str(exc))
    attempt(lambda:engine()['stop'](path))
    attempt(lambda:remove(path))
    for filename,pid_key,born_key in [('attachment.json','client_pid','client_identity'),
                                      ('tunnel.json','pid','identity')]:
        try:record=json.loads((path/filename).read_text())
        except FileNotFoundError:continue
        except (OSError,ValueError) as exc:errors.append(str(exc));continue
        attempt(lambda:stop_process(record.get(pid_key),record.get(born_key)))
    attempt(lambda:stop_process(data.get('provider_pid'),data.get('provider_identity')))
    if errors:raise RuntimeError('Allocation ended; cleanup will retry: '+'; '.join(errors))
    data.update(allocation_cleaned=True,client_pid=None,client_identity='',provider_pid=None,
                provider_identity='',provider_exit=0,error='')
    write(path/'session.json',data)


def daemon(path):
    data=json.loads((path/'session.json').read_text());cfg=config(data['config'])
    if data.get('allocation_cleaned'):return
    token=data['token'];port=data['remote_port'];child=None;offset=0;last=None
    if cfg.ssh.connection!='local':
        os.environ['LLM_AWAY_SSH_CONTROL']=str((path/'ssh.sock').resolve())
    if (path/'control.sock').exists():
        if rpc_alive(path):raise RuntimeError('Daemon already owns this session')
        (path/'control.sock').unlink(missing_ok=True)
    from .session_guard import register as guard_runner
    guard_runner(path)
    pid=data.get('provider_pid');born=data.get('provider_identity')
    if pid and born and identity(pid)==born:
        # Adopt a live provider after a daemon restart so status and stop remain accurate.
        child=AdoptedProcess(pid)
    sock=socket.socket(socket.AF_UNIX);sock.bind(str(path/'control.sock'));os.chmod(path/'control.sock',0o600)
    sock.listen(4);sock.settimeout(1)
    def state(**kwargs):
        data.update(kwargs);write(path/'session.json',data)
        if 'allocation' in kwargs:
            from .gpu_history import append
            try:append(path,(kwargs['allocation'] or {}).get('gpu_telemetry'))
            except Exception as exc:print(f'GPU history: {exc}',file=sys.stderr,flush=True)
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
        before=remote(cfg,token,port,'status')
        remote(cfg,token,port,'stop')
        if allocation_pending(before):
            # No model process exists to acknowledge the stop. The changed
            # desired generation prevents a later start if Slurm assigns a node.
            state(model='',client_pid=None,client_identity='',provider_pid=None,provider_identity='')
            return
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
            allocation=remote(cfg,token,port,'reserve',session_id=data['id']);state(allocation=allocation,phase=allocation['slurm_state'])
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
        try:logs=remote(cfg,token,port,'log',offset=offset)
        except Exception:
            if not allocation_ended(cfg,s):raise
            logs={'offset':offset,'data':''}
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
                            allocation=remote(cfg,token,port,'start')
                            state(allocation=allocation,config=asdict(cfg),model=cfg.model.name,provider_exit=None,client_pid=request.get('client_pid'),client_identity=identity(request.get('client_pid')))
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
                    if 'rag_log' in s:(path/'rag.log').write_text(s['rag_log'])
                    if allocation_ended(cfg,s):
                        cleanup_ended_allocation(path,data)
                        print('Allocation ended; local processes cleaned up. History retained.',flush=True)
                        break
                except Exception as exc:state(error=str(exc));print('Monitor:',exc,flush=True)
    finally:
        if child is not None and child.poll() is None:child.terminate()
        sock.close();(path/'control.sock').unlink(missing_ok=True)
        remote_pool.shutdown(wait=False,cancel_futures=True)
        os.environ.pop('LLM_AWAY_SSH_CONTROL',None)


def allocate(args):
    connection=args.connection or ('ssh' if args.host else
        ['local','ssh'][choose_option(['local','ssh'],'Location',default=0)])
    mode=getattr(args,'mode',None) or ['native','custom'][choose_option(
        ['Native CLI','Custom model'],'Session type',default=1)]
    if mode=='native':
        cli=getattr(args,'cli',None) or ['claude','codex'][choose_option(
            ['claude','codex'],'Native CLI',default=1)]
        host=None
        if connection=='ssh':
            from .onboarding import ssh_hosts, select_host
            aliases,patterns=ssh_hosts()
            host=select_host(aliases,'',args.host,patterns)
        elif not shutil.which(cli):
            raise ValueError(f'{cli} is not available on PATH')
        from .native_sessions import allocate as allocate_native
        return allocate_native(STORE,cli,host)
    STORE.mkdir(parents=True,exist_ok=True,mode=0o700)
    with open(STORE/'registry.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        configure_local(args.config,alias=args.host,connection=connection,restart=args.restart,resources_only=True)
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
    from .serve_registration import register
    helper=getattr(args,'helper',False)
    if helper:args.detach=True
    path=path_for(args.session)
    if json.loads((path/'session.json').read_text()).get('allocation_cleaned'):
        raise ValueError('Allocation ended; allocate new resources to run a model')
    if json.loads((path/'session.json').read_text()).get('native'):
        from .native_sessions import run
        return run(path,args)
    from . import terminals
    selection_path=path/'agent-selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text())
        data=json.loads((path/'session.json').read_text())
        existing=(terminals.engine()['alive'](path) if selection.get('location')=='local' else
                  remote(config(data['config']),data['token'],data['remote_port'],'status').get('terminal',{}).get('status')=='TERMINAL')
        if existing:
            if helper:
                if args.model and args.model!=data.get('model'):raise ValueError('Exit the agent before switching its model')
                register(args.session,args.rag,getattr(args,'log_helper',True))
                return
            if args.model and args.model not in (data.get('model'),data['config']['llamacpp'].get('model_name'),selection.get('loading',{}).get('model',{}).get('alias')):raise ValueError('Exit the agent terminal before switching models')
            if getattr(args,'agent_location','local')!=selection.get('target_location',selection.get('location')) or (args.cli and args.cli not in ('auto',selection.get('cli'))) or (args.agent_workdir and args.agent_workdir!=selection.get('cwd')):
                raise ValueError('An agent terminal already exists; exit it before changing its configuration')
            if not args.detach:terminals.attach(path,data,selection)
            else:print('Agent terminal is already running; F2 attaches, Ctrl+B then D detaches.')
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
            cfg=replace(cfg,codex=current.codex,agent=current.agent,claude=current.claude,
                        llamacpp=replace(cfg.llamacpp,model_batch_defaults=current.llamacpp.model_batch_defaults))
        location=getattr(args,'agent_location','local')
        reuse=False;resume=bool(getattr(args,'resume',False) and not helper)
        # Local agents can use their own native Codex/Claude account (no remote
        # model) instead of loading a session model.
        if (location=='local' and cfg.ssh.connection=='local' and not helper and not args.model and not reuse
                and sys.stdin.isatty() and not getattr(args,'quiet',False)
                and not (data.get('model') and data.get('provider_exit') is None)):
            choice=choose_option(['Session model (AWAY inference)','Native CLI (own model/account)'],
                                 'Local agent model',default=0)
            if choice==1:
                available=[name for name in ('codex','claude') if shutil.which(name)]
                if not available:raise ValueError('No codex or claude CLI on PATH for native mode')
                preference=getattr(args,'cli',None) or os.environ.get('LLM_AWAY_CLI') or cfg.agent.cli
                native_cli=choose_cli(preference,available) if len(available)>1 else available[0]
                agent_cwd=getattr(args,'agent_workdir',None) or os.getcwd()
                rag_config=({'paths':args.rag,'threads':getattr(args,'rag_threads',2),
                             'memory_gb':getattr(args,'rag_memory_gb',0),'gpu':getattr(args,'rag_gpu',False),
                             'compute':'local','paths_location':'local'}
                            if args.rag else None)
                instructions=cfg.claude.instructions or cfg.codex.instructions
                if rag_config:
                    instructions+=(('\n' if instructions else '')+
                                   'RAG is available through the exact function '
                                   '`mcp__project_search__search_project`. Invoke that full function name with '
                                   '`{"query":"what to find","limit":6}`. Never call `mcp__project_search` by itself, '
                                   'and do not look for RAG in MCP resource listings.')
                selection={'location':'local','cli':native_cli,'cwd':agent_cwd,'extra_args':args.agent_args,
                           'rag_config':rag_config,'rag_command':None,'native':True,
                           'instructions':instructions}
                write(path/'agent-selection.json',selection)
                current=json.loads((path/'session.json').read_text())
                terminals.ensure(path,current,selection)
                if not args.detach:terminals.attach(path,current,selection)
                return
        loaded=bool(data.get('model') and data.get('provider_exit') is None and
                    data.get('provider_identity') and identity(data.get('provider_pid'))==data['provider_identity'])
        if loaded:
            reuse=(not args.model or args.model in (data['model'],cfg.llamacpp.model_name))
            if not args.model and not args.detach and not getattr(args,'resume',False):
                from .monitor_ui import dropdown_win
                keep=f"Keep loaded model: {data['model']}"
                choice=dropdown_win('Model already running',[keep,'Load a different model'])
                if choice is None:return
                reuse=choice==keep
            if reuse:
                if args.detach:pass
                elif getattr(args,'resume',False):resume=True
                else:
                    mode=choose_option(['Reopen agent (saved-conversation picker)','Keep running in background'],
                                       'Session mode',default=1 if args.detach else 0)
                    args.detach=mode==1;resume=mode==0
        if reuse:
            model=dict(alias=data['model'],name=cfg.llamacpp.model_name,context_size=cfg.llamacpp.context_size)
        else:
            models=discover_models(cfg)
            if not args.model and sys.stdin.isatty() and not getattr(args,'quiet',False):
                from .monitor_ui import mtp_select_win
                model,args.mtp=mtp_select_win(models,args.session,cfg.llamacpp.model_name)
                if model is None:return
                args.mtp_prompted=True
            else:model=choose_model(models,cfg.llamacpp.model_name,args.model)
        # Explicit preset context also applies to allocations created before the preset.
        if model.get('context_size',0)>0:
            cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,context_size=model['context_size']),
                        codex=replace(cfg.codex,context_window=min(cfg.codex.context_window,model['context_size'])))
        location=getattr(args,'agent_location','local')
        if not helper and location=='remote' and args.agent_args:
            raise ValueError('Remote mode does not support extra agent CLI arguments')
        preference=getattr(args,'cli',None) or os.environ.get('LLM_AWAY_CLI') or cfg.agent.cli
        selected_cli=None
        if location=='local' and not helper:
            if getattr(args,'quiet',False) and preference=='auto':
                preference=next((cli for cli in ('codex','claude') if shutil.which(cli)), 'codex')
            selected_cli=choose_cli(preference)
        mtp=cfg.llamacpp.mtp if reuse else args.mtp if getattr(args,'mtp_prompted',False) else choose_mtp(model,cfg.llamacpp.mtp,args.mtp)
        if not reuse:
            cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,server_extra_args=model_server_options(cfg,model['name'])))
        if not reuse and sys.stdin.isatty() and not getattr(args,'quiet',False):
            from .monitor_ui import _form_screen
            saved_options=shlex.join(cfg.llamacpp.server_extra_args)
            options=_form_screen('Model server options',[('text','Options',None,saved_options)],[saved_options])
            if options is None:return
            cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,server_extra_args=shlex.split(options[0])))
        if getattr(args,'server_extra_args',None) is not None:
            cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,server_extra_args=args.server_extra_args))
        rag_command=None
        rag_config=({'paths':args.rag,'threads':getattr(args,'rag_threads',2),
                     'memory_gb':getattr(args,'rag_memory_gb',0),'gpu':getattr(args,'rag_gpu',False),
                     'compute':getattr(args,'rag_compute',None) or location,
                     'paths_location':getattr(args,'rag_paths_location',None) or location}
                    if args.rag else None)
        agent_instructions=(cfg.claude.instructions or cfg.codex.instructions
                            if selected_cli=='claude' else cfg.codex.instructions)
        if rag_config:
            agent_instructions+=('\nRAG is available through the exact function '
                                 '`mcp__project_search__search_project`. Invoke that full function name with '
                                 '`{"query":"what to find","limit":6}`. Never call `mcp__project_search` by itself, '
                                 'and do not look for RAG in MCP resource listings.')
        # Native/local allocations keep RAG local. Scheduled allocations prepare
        # it after the job is ready and bridge its stdio back to a local agent.
        if not helper:
            # Load inside the retained local terminal, then open the selected CLI.
            # A remote CLI is attached through this terminal after readiness.
            terminals.engine()['tmux']()
            if loaded and not reuse:rpc(path,'stop',client_pid=os.getpid())
            agent_cwd=getattr(args,'agent_workdir',None) or (os.getcwd() if location=='local' else '')
            if location=='remote' and agent_cwd and not agent_cwd.startswith('/'):
                raise ValueError('--agent-workdir must be an absolute remote path')
            selection={'location':'local','target_location':location,'cli':selected_cli or preference,
                       'cwd':agent_cwd,'extra_args':args.agent_args,'rag_command':rag_command,
                       'rag_config':rag_config,'resume':resume,
                       'instructions':agent_instructions,
                       'context_window':cfg.codex.context_window,
                       'auto_compact_token_limit':min(cfg.codex.auto_compact_token_limit,int(cfg.codex.context_window*0.7)),
                       'loading':{'config':asdict(cfg),'model':model,'reuse':reuse,'mtp':mtp}}
            if (selected_cli or preference) in ('codex','auto') and cfg.codex.custom_metadata:
                selection['codex_catalog_content']=json.dumps(remote_model_catalog(
                    model['alias'],context_window=cfg.codex.context_window,settings=cfg.codex),indent=2)+'\n'
            write(path/'agent-selection.json',selection)
            current=json.loads((path/'session.json').read_text())
            terminals.ensure(path,current,selection)
            fcntl.flock(lease,fcntl.LOCK_UN)
            print('Model loading in tmux. Ctrl+B then D detaches; run --session '+str(args.session)+' or F2 reattaches.',flush=True)
            if not args.detach:terminals.attach(path,current,selection)
            return
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
                allocation=s.get('allocation') or {}
                if s.get('provider_exit') is not None or not allocation.get('active',True) or allocation.get('model_state')=='EXITED':raise RuntimeError('Backend stopped; inspect session log')
                if allocation.get('slurm_state') in ('PENDING','CONFIGURING') or not allocation.get('host'):
                    deadline=time.time()+cfg.gateway.startup_timeout_seconds
                if time.time()>deadline:raise TimeoutError('Model startup timed out')
                time.sleep(1)
            detached=True
            register(args.session,args.rag,getattr(args,'log_helper',True))
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
    if data.get('allocation_cleaned'):
        data['phase']='RELEASED';write(path/'session.json',data);return
    if data.get('native'):
        return rpc(path,'release')
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
    try:
        if allocation_pending(data.get('allocation') or {}):
            raise RuntimeError('Pending allocation has no model process; bypassing daemon model stop')
        return rpc(path,'release')
    except Exception as rpc_exc:
        # Release is stronger than model stop. An old/stuck daemon may reject
        # model shutdown even though the scheduler job can still be cancelled.
        # Fall back to the remote release primitive for every RPC failure.
        pid=data.get('provider_pid');born=data.get('provider_identity')
        if pid and born and identity(pid)==born:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        release_error=''
        try:remote(config(data['config']),data['token'],data['remote_port'],'release')
        except Exception as exc:release_error=str(exc)
        if release_error:
            data.update(phase='RELEASE FAILED',error='Daemon release failed: '+str(rpc_exc)+'; remote cancellation failed: '+release_error)
            write(path/'session.json',data)
            raise RuntimeError(data['error']) from None
        runner_path=path/'runner.json'
        try:
            runner=json.loads(runner_path.read_text());runner['finished']=True;write(runner_path,runner)
            from .session_guard import stop_process
            stop_process(data.get('pid'),runner.get('identity'))
        except (OSError,ValueError):pass
        latest=data
        try:latest=json.loads((path/'session.json').read_text())
        except (OSError,ValueError):pass
        allocation=dict(latest.get('allocation') or {})
        allocation.update(active=False,model_state='STOPPED')
        latest.update(phase='RELEASED',model='',error='',allocation=allocation,
                      provider_pid=None,provider_identity='',client_pid=None,client_identity='')
        write(path/'session.json',latest);(path/'control.sock').unlink(missing_ok=True)
        return {'released':True,'fallback':True}


def submit_prompt(number, text, location=None, cli=None, cwd=None):
    from . import terminals
    path=path_for(number)
    data=json.loads((path/'session.json').read_text())
    saved=path/'agent-selection.json'
    if not saved.exists():raise ValueError('Open the agent first with run')
    selection=json.loads(saved.read_text())
    if (location and location!=selection.get('target_location',selection['location'])) or (cli and cli!=selection['cli']) or (cwd and cwd!=selection.get('cwd')):
        raise ValueError('Use the existing terminal settings, or exit the agent before reconfiguring')
    if not text.strip():raise ValueError('Prompt is empty')
    return terminals.send(path,data,text)


def restart_session(number):
    """Retry failed/cleaned allocations with current host settings, retaining history."""
    from .host_store import apply_profile, read_store
    from .session_guard import locked, stop_process
    path=path_for(number)
    with locked(path):
        data=json.loads((path/'session.json').read_text())
        if data.get('native') or data.get('phase')=='RELEASED':
            raise ValueError('Restart is for failed or ended resource allocations')
        if data.get('provider_pid') or data.get('model') and not data.get('allocation_cleaned'):
            raise ValueError('A model may still be running; release it before restarting')
        if data.get('allocation') and not data.get('allocation_cleaned'):
            raise ValueError('Allocation may still be active; wait for confirmed cleanup before restarting')
        config_path=os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/model.toml'))
        old=config(data['config']);cfg=load_config(config_path)
        if old.ssh.connection=='ssh':
            profile=read_store(config_path)['hosts'].get(old.ssh.host)
            if profile is None:raise ValueError('No saved settings for '+old.ssh.host)
            cfg=apply_profile(cfg,profile)
        else:
            cfg=replace(old,codex=cfg.codex,agent=cfg.agent,claude=cfg.claude)
        cfg=replace(cfg,ssh=old.ssh,server=old.server,gateway=old.gateway,
                    slurm=replace(cfg.slurm,gpus=data['gpus'],nodes=old.slurm.nodes,
                                  custom_options=old.slurm.custom_options),
                    hosts={},saved_hosts={},active_host='')
        # Stop only the verified local runner. Mark its watchdog finished first,
        # preventing it from releasing a replacement allocation.
        runner_path=path/'runner.json'
        runner=json.loads(runner_path.read_text()) if runner_path.exists() else {}
        pid=data.get('pid');born=runner.get('identity')
        if pid:
            current=identity(pid)
            if not current:
                try:os.kill(pid,0)
                except ProcessLookupError:pass
                else:raise ValueError('Cannot verify the old daemon identity; restart from an unrestricted terminal')
            elif current==born and runner.get('pid')==pid and runner.get('token')==data['token']:
                runner['finished']=True;write(runner_path,runner)
                try:stop_process(pid,born)
                except Exception:
                    runner['finished']=False;write(runner_path,runner);raise
                if identity(pid)==born:raise RuntimeError('Old daemon has not stopped; retry later')
            else:raise ValueError('Runner identity changed; refusing to stop an unrelated process')
        data=json.loads((path/'session.json').read_text())
        history=data.setdefault('restart_history',[])
        history.append({k:data.get(k) for k in ('token','phase','error','allocation')})
        # Never allocate a second job while recovering an unacknowledged reserve:
        # reuse its token. A confirmed ended job needs a fresh token.
        if data.get('allocation_cleaned'):data['token']=uuid.uuid4().hex
        data.update(config=asdict(cfg),phase='STARTING',error='',model='',pid=None,
                    allocation_cleaned=False,provider_pid=None,provider_identity='',provider_exit=None,
                    client_pid=None,client_identity='')
        data.pop('allocation',None)
        write(path/'session.json',data)
        (path/'control.sock').unlink(missing_ok=True)
        with (path/'session.log').open('a') as log:
            log.write('Restart requested using current host settings.\n');log.flush()
            subprocess.Popen([sys.executable,'-m','llm_away.resources','daemon',str(path)],
                             stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    return number


def refresh_session(number, model=None):
    """Restart the session agent as a fresh Codex conversation, keeping helpers."""
    path=path_for(number)
    data=json.loads((path/'session.json').read_text())
    if data.get('native'):
        raise ValueError('Native sessions refresh by exiting the CLI and reopening with F2')
    saved=path/'agent-selection.json'
    if not saved.exists():raise ValueError('Open the agent first with run')
    selection=json.loads(saved.read_text())
    rag=selection.get('rag_config') or {}
    from . import terminals
    engine=terminals.engine()
    if engine['alive'](path):
        engine['stop'](path)
        deadline=time.monotonic()+5
        while engine['alive'](path) and time.monotonic()<deadline:time.sleep(0.2)
    args=Namespace(session=number,detach=True,model=model,mtp=None,rag=rag.get('paths',[]),helper=False,quiet=True,resume=False,
                   rag_threads=rag.get('threads',2),rag_memory_gb=rag.get('memory_gb',0),rag_gpu=rag.get('gpu',False),
                   rag_compute=rag.get('compute'),rag_paths_location=rag.get('paths_location'),
                   log_helper=True,agent_location=selection.get('target_location',selection.get('location','local')),
                   cli=selection.get('cli'),agent_workdir=selection.get('cwd'),
                   agent_args=selection.get('extra_args',[]))
    run_agent(args)


def reconnect_session(number):
    """Repair the selected provider's SSH tunnel without running an inference."""
    path=path_for(number)
    data=rpc(path,'status')
    if data.get('native'):
        raise ValueError('Native sessions do not use a model SSH tunnel')
    if not data.get('model') or data.get('provider_exit') is not None:
        raise ValueError('Load a model before reconnecting')
    server=data['config']['server']
    payload={'model':data['model'],'messages':[{'role':'user','content':'.'}]}
    request=Request(f"http://127.0.0.1:{int(server['port'])}/v1/messages/count_tokens",
                    data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=60) as response:
        if response.status!=200:raise RuntimeError('Provider did not acknowledge reconnect')
        response.read(4096)
    gateway=data['config']['gateway']
    with urlopen(f"http://127.0.0.1:{int(gateway['local_port'])}/health",timeout=5) as response:
        if response.status!=200:raise RuntimeError('Tunnel did not become ready')
    return f'Session {number} tunnel reconnected.'


def model_server_options(cfg, name):
    options=list(cfg.llamacpp.server_extra_args)
    defaults=cfg.llamacpp.model_batch_defaults.get(name)
    if not defaults:return options
    result=[];skip=False
    for value in options:
        if skip:skip=False;continue
        replaced=('--batch-size','-b','--ubatch-size','-ub')
        if name=='glm-5.3-flash-q4':replaced+=('--cache-type-k','--cache-type-v')
        if value in replaced:skip=True;continue
        if value.split('=',1)[0] in replaced:continue
        result.append(value)
    if name=='glm-5.3-flash-q4':result+=['--cache-type-k','q4_0','--cache-type-v','q4_0']
    return result+['--batch-size',str(defaults[0]),'--ubatch-size',str(defaults[1])]


def refresh_monitor():
    removed=[];kept=[]
    for saved in sorted(STORE.glob('[0-9]*/session.json'),key=lambda p:int(p.parent.name)):
        data=json.loads(saved.read_text())
        if data.get('phase')=='RELEASED' or data.get('native'):continue
        try:
            status=remote(config(data['config']),data['token'],data['remote_port'],'status')
            if allocation_ended(config(data['config']),status) or data.get('allocation_cleaned'):
                release_session(data['id']);removed.append(str(data['id']))
        except Exception as exc:kept.append(str(data['id'])+': '+str(exc))
    return 'Removed ended sessions: '+(', '.join(removed) or 'none')+('; retained uncertain sessions: '+'; '.join(kept) if kept else '')


def cleanup_monitor():
    from . import cleanup
    from .monitor_ui import cleanup_confirm, terminal_operation
    items=[]
    for saved in sorted(STORE.glob('[0-9]*/session.json'),key=lambda p:int(p.parent.name)):
        data=json.loads(saved.read_text())
        if data.get('phase')=='RELEASED' and not cleanup.busy(saved.parent):
            cleanup.add_session_items(items,saved.parent,False)
    if not items:return 'No inactive released-session leftovers.'
    plan=terminal_operation(cleanup.preview,items)
    if not cleanup_confirm(plan['paths']):return 'Cleanup canceled.'
    import io
    def perform():
        if cleanup.preview(items)!=plan:raise ValueError('Files changed; review cleanup again')
        output=io.StringIO()
        with contextlib.redirect_stdout(output):cleanup.execute(items,list(range(len(items))),expected_remote=plan['remote_paths'])
        return output.getvalue()
    return terminal_operation(perform)


def allocate_and_run(mode,session=None):
    """Allocate resources and/or a model, then return control to res-mon."""
    from argparse import Namespace
    from .monitor_ui import terminal_operation, dropdown_win
    def report(text):
        dropdown_win(text, ['Back to monitor'])
    if mode=='model':
        if session is not None:
            try:
                data=json.loads((path_for(session)/'session.json').read_text())
                if data.get('phase')=='RELEASED' or data.get('native'):session=None
            except (OSError,ValueError):session=None
        if session is None:
            chosen=[]
            for p in sorted(STORE.glob('*/session.json'),key=lambda p:int(p.parent.name)):
                data=json.loads(p.read_text())
                if data.get('phase')!='RELEASED' and not data.get('native'):chosen.append(int(p.parent.name))
            if not chosen:
                report('No eligible resource allocation.');return
            session=max(chosen)
        number=session
    if mode=='allocate':
        from .monitor_ui import allocation_wizard
        settings=allocation_wizard()
        if settings is None:return
        try:number=terminal_operation(allocate_from_wizard,settings)
        except (ValueError,RuntimeError,OSError,subprocess.SubprocessError) as exc:
            report('Allocation failed: '+str(exc));return
    path=path_for(number)
    data=json.loads((path/'session.json').read_text())
    if data.get('native') or data.get('phase')=='RELEASED':return
    from .monitor_ui import model_select_win
    cfg=config(data['config'])
    try:models=terminal_operation(discover_models,cfg)
    except Exception as exc:report('Model discovery failed: '+str(exc));return
    from .monitor_ui import mtp_select_win
    try:
        model,mtp=mtp_select_win(models,number,current=str(data.get('model') or cfg.llamacpp.model_name))
    except KeyboardInterrupt:return
    if model is None:return
    cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,server_extra_args=model_server_options(cfg,model['name'])))
    from .monitor_ui import _form_screen
    options=_form_screen('Model server options', [('text','Options',None,shlex.join(cfg.llamacpp.server_extra_args))], [shlex.join(cfg.llamacpp.server_extra_args)])
    if options is None:return
    try:extra=shlex.split(options[0])
    except ValueError as exc:report(str(exc));return
    args=Namespace(session=number,model=model['name'],mtp=mtp,rag=[],helper=False,quiet=True,server_extra_args=extra,
                   mtp_prompted=True,
                   log_helper=True,agent_location='local',cli=None,agent_workdir=None,
                   agent_args=[],resume=True,detach=True)
    try:terminal_operation(run_agent,args)
    except (ValueError,RuntimeError,OSError,subprocess.SubprocessError) as exc:
        report('Model allocation failed: '+str(exc));return
    return number


def allocate_from_wizard(settings):
    """Create the allocation from wizard settings, without terminal prompts."""
    from dataclasses import asdict
    from .config import AppConfig, load_config, replace
    if settings.get('native'):
        cli=settings['cli'];host=None
        if settings['connection']=='ssh':
            host=settings.get('host') or None
            if not host:raise ValueError('SSH host required for a native SSH session')
            config_path=os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/model.toml'))
            host=load_config(config_path).host_config(host).ssh_host
        elif not shutil.which(cli):
            raise ValueError(f'{cli} is not available on PATH')
        from .native_sessions import allocate as allocate_native
        return allocate_native(STORE,cli,host)
    config_path=os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/model.toml'))
    STORE.mkdir(parents=True,exist_ok=True,mode=0o700)
    with open(STORE/'registry.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cfg=load_config(config_path)
        if settings['connection']=='ssh':
            host=settings.get('host') or cfg.ssh.host
            cfg=cfg.with_host(host)
            cfg=replace(cfg,ssh=replace(cfg.ssh,connection='ssh'),active_host='')
        else:
            cfg=replace(cfg,ssh=replace(cfg.ssh,connection='local',host='localhost'))
        gpus=int(settings.get('gpus') or 0)
        if gpus<0 or (cfg.llamacpp.backend!='cpu' and gpus<1):raise ValueError('Invalid GPU count')
        cfg=replace(cfg,slurm=replace(cfg.slurm,gpus=gpus,nodes=1,
            custom_options=shlex.split(settings.get('slurm_options') or '')))
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
    print(f'Session {number} allocating in background.')
    return number


def monitor(args):
    if args.logs:
        path=path_for(args.logs);subprocess.call(['tail','-n','60','-f',str(path/'session.log')]);return
    if getattr(args,'refresh',None):
        refresh_session(args.refresh,getattr(args,'model',None));return
    if not args.list and not args.kill and sys.stdin.isatty():
        if not sys.stdout.isatty():
            # Some terminal wrappers leave stdout piped while stdin remains the
            # controlling TTY. Curses needs the TTY, so attach stdout to it.
            try:
                tty=os.open('/dev/tty',os.O_WRONLY);os.dup2(tty,1);os.close(tty)
            except OSError:pass
        if not sys.stdout.isatty():return
        from .monitor_ui import show
        from .serve_registration import register as set_helper
        while True:
            chosen=show(STORE,release_session,run_agent,submit_prompt,refresh_session,allocate_and_run,lambda number:set_helper(number,log_helper=True),restart_session,lambda number:rpc(path_for(number),'stop'),refresh_monitor,cleanup_monitor,reconnect_session)
            if chosen=='monitor':continue
            if chosen is None:break
            # Re-exec the run wrapper: guarantees a clean terminal handoff to Codex.
            selection=path_for(chosen)/'agent-selection.json'
            saved=json.loads(selection.read_text()) if selection.exists() else {}
            command=[sys.executable,'-m','llm_away.resources','run','--resume','--session',str(chosen),'--agent-location',saved.get('target_location',saved.get('location','local'))]
            if saved.get('cli'):command+=['--cli',saved['cli']]
            if saved.get('cwd'):command+=['--agent-workdir',saved['cwd']]
            subprocess.call(command)
        return
    entries=[]
    for p in sorted(STORE.glob('*/session.json'),key=lambda p:int(p.parent.name)):
        data=json.loads(p.read_text())
        if data.get('phase')=='RELEASED':continue
        try:data=rpc(p.parent,'status')
        except (OSError,RuntimeError):
            if not data.get('allocation_cleaned'):data['phase']='DAEMON OFFLINE'
        try:
            from .session_guard import ensure
            data['ps']=ensure(p.parent,data)
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError):data['ps']=None
        entries.append(data)
    if entries:
        from .helper_relations import annotate
        annotate(STORE,entries)
        rows=[['ID','HOST','GPUS','STATE','MODEL','JOB','TIME LEFT (dd-hh:mm)','IS MASTER','IS HELPER','PS','ERROR']]
        for data in entries:
            from .monitor_ui import time_left_text
            rows.append([str(data['id']),data['host'],str(data['gpus']),data['phase'],
                         data.get('model') or ('native '+data['native_cli'] if data.get('native') else '-'),str(data.get('allocation',{}).get('job_id') or '-'),
                         time_left_text(data),data['is_master'],data['is_slave'],str(data.get('ps') or '—'),data.get('error') or ''])
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
    alloc.add_argument('--mode',choices=['native','custom'],help='Native account CLI or custom model allocation')
    alloc.add_argument('--cli',choices=['claude','codex'],help='CLI to launch in native mode')
    run=sub.add_parser('run');run.add_argument('--session','-s',required=True,type=int);run.add_argument('--model');run.add_argument('--mtp',choices=['auto','on','off']);run.add_argument('--rag',action='append',metavar='PATH',help='Local code/docs file or folder; repeat for multiple paths');run.add_argument('agent_args',nargs=argparse.REMAINDER)
    run.add_argument('--agent-location',choices=['local','remote'],default='local')
    run.add_argument('--agent-workdir',help='Project directory on the selected agent host')
    run.add_argument('--rag-threads',type=int,default=2,help='CPU cores used on the selected RAG compute host')
    run.add_argument('--rag-compute',choices=['local','remote'],help='Host for the RAG embedding model (default: agent host)')
    run.add_argument('--rag-paths-location',choices=['local','remote','shared'],help='Where RAG source paths are visible')
    run.add_argument('--rag-memory-gb',type=float,default=0,help='Local RAG memory limit in GiB; 0 is unlimited')
    run.add_argument('--rag-gpu',action='store_true',help='Use an available local CoreML, CUDA, or ROCm provider for RAG')
    run.add_argument('--cli',choices=['auto','codex','claude'],help='Agent CLI; auto asks only when both are installed')
    run.add_argument('--detach',action='store_true',help='Start or reuse the tmux agent without attaching')
    run.add_argument('--helper',action='store_true',help='Load/reuse the model and register it as a Codex MCP helper; --rag selects folders, default current directory')
    run.add_argument('--log-helper',dest='log_helper',action='store_true',default=True,help='Log helper questions, excerpts, and responses to run/resources/ID/helper.log (default)')
    run.add_argument('--no-log-helper',dest='log_helper',action='store_false',help='Disable helper traffic logging')
    run.add_argument('--resume',action='store_true',help='Open the saved-conversation picker when reopening an exited agent')
    prompt=sub.add_parser('prompt');prompt.add_argument('--session','-s',required=True,type=int);prompt.add_argument('text')
    prompt.add_argument('--agent-location',choices=['local','remote']);prompt.add_argument('--cli',choices=['codex','claude']);prompt.add_argument('--agent-workdir')
    mon=sub.add_parser('monitor');mon.add_argument('--list',action='store_true');mon.add_argument('--logs',type=int);mon.add_argument('--kill',type=int);mon.add_argument('--release',action='store_true');mon.add_argument('--refresh',type=int);mon.add_argument('--model')
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

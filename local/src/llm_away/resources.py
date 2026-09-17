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
import uuid
from urllib.request import urlopen

from .config import AppConfig, load_config
from .backends import SlurmServerBackend, KubernetesBackend, BackendError
from .models import discover_models, choose_model, choose_mtp
from .onboarding import configure_local, ask
from .prompt import choose_option

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
        command=['ssh','-x','-o','BatchMode=yes','-o',f'ConnectTimeout={cfg.ssh.connect_timeout_seconds}',
                 cfg.ssh.destination,shlex.join(command)]
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

def path_for(number):
    if not str(number).isdigit(): raise ValueError('Session ID must be a number')
    path=STORE/str(int(number))
    if not (path/'session.json').exists(): raise ValueError('Unknown session '+str(number))
    return path

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
    def close(self):
        self._closing.set()
        if self._tunnel:
            self._tunnel.terminate()
            try:self._tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:self._tunnel.kill()
            self._tunnel=None


def provider(path):
    from .server import serve
    data=json.loads((path/'session.json').read_text());cfg=config(data['config'])
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
    sock=socket.socket(socket.AF_UNIX);sock.bind(str(path/'control.sock'));os.chmod(path/'control.sock',0o600)
    sock.listen(4);sock.settimeout(1)
    def state(**kwargs):
        data.update(kwargs);write(path/'session.json',data)
    def stop_model(caller=None):
        nonlocal child,cfg
        client=data.get('client_pid')
        if client and client!=caller and data.get('client_identity') and identity(client)==data['client_identity']:
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
    state(pid=os.getpid(),phase='ALLOCATING',error='')
    try:
        allocation=remote(cfg,token,port,'reserve');state(allocation=allocation,phase=allocation['slurm_state'])
        print('Allocation:',allocation,flush=True)
    except Exception as exc:
        state(phase='ERROR',error=str(exc));print(exc,flush=True)
    release_requested=False
    def interrupted(sig,frame):
        nonlocal release_requested
        release_requested=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    tick=0
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
                                llamacpp=replace(cfg.llamacpp,model_name=request['model']['name'],mtp=request['mtp']))
                            if request['model'].get('context_size',0)>0:
                                cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,context_size=request['model']['context_size']))
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
            if time.time()-tick>=5:
                tick=time.time()
                try:
                    s=remote(cfg,token,port,'status');state(allocation=s,phase=s['slurm_state'],error='',provider_exit=child.poll() if child else None)
                    summary=(s['slurm_state'],s.get('model_state'),s.get('host'))
                    if summary!=last:print('Resource state:',summary,flush=True);last=summary
                    logs=remote(cfg,token,port,'log',offset=offset);offset=logs['offset']
                    if logs['data']:print(logs['data'],end='',flush=True)
                except Exception as exc:state(error=str(exc));print('Monitor:',exc,flush=True)
    finally:
        if child is not None and child.poll() is None:child.terminate()
        sock.close();(path/'control.sock').unlink(missing_ok=True)


def allocate(args):
    STORE.mkdir(parents=True,exist_ok=True,mode=0o700)
    with open(STORE/'registry.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        configure_local(args.config,alias=args.host,connection=args.connection,restart=args.restart,resources_only=True)
        cfg=load_config(args.config)
        gpus=args.gpus if args.gpus is not None else int(ask('GPUs for this allocation',str(cfg.slurm.gpus or (0 if cfg.llamacpp.backend=='cpu' else 1))))
        if gpus<0 or (cfg.llamacpp.backend!='cpu' and gpus<1): raise ValueError('Invalid GPU count')
        cfg=replace(cfg,slurm=replace(cfg.slurm,gpus=gpus,nodes=1))
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
        number=1+max([int(p.name) for p in STORE.iterdir() if p.name.isdigit()]+[0])
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
    # Advisory lease prevents two foreground clients from sharing one model slot.
    with open(path/'client.lock','a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another run command owns this session')
        data=rpc(path,'status');cfg=config(data['config'])
        model=choose_model(discover_models(cfg),cfg.llamacpp.model_name,args.model)
        # Explicit preset context also applies to allocations created before the preset.
        if model.get('context_size',0)>0:
            cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,context_size=model['context_size']),
                        codex=replace(cfg.codex,context_window=model['context_size']))
        mtp=choose_mtp(model,cfg.llamacpp.mtp,args.mtp)
        started=False
        agent=None
        external_stop=False
        def interrupted(sig,frame):
            nonlocal external_stop
            external_stop=True
            if agent and agent.poll() is None:agent.terminate()
            raise KeyboardInterrupt
        previous=signal.signal(signal.SIGTERM,interrupted)
        try:
            rpc(path,'start',model=model,mtp=mtp,client_pid=os.getpid());started=True
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
            from .cli import remote_model_catalog
            catalog=path/'models.json';write(catalog,remote_model_catalog(model['alias'],min(cfg.codex.context_window,cfg.llamacpp.context_size),cfg.codex))
            # Per-process overrides keep simultaneous sessions out of global Codex settings.
            command=['codex','--no-alt-screen','-c','model_provider="away_resource"','-c','model="away"',
                     '-c','model_providers.away_resource.name="AWAY resource"',
                     '-c',f'model_providers.away_resource.base_url="http://127.0.0.1:{cfg.server.port}/v1"',
                     '-c','model_providers.away_resource.wire_api="responses"',
                     '-c','model_providers.away_resource.requires_openai_auth=false',
                     '-c','model_catalog_json='+json.dumps(str(catalog)),
                     '-c','model_context_window='+str(min(cfg.codex.context_window,cfg.llamacpp.context_size)),
                     '-c','model_auto_compact_token_limit='+str(int(min(cfg.codex.context_window,cfg.llamacpp.context_size)*0.7))]
            agent=subprocess.Popen(command+args.agent_args)
            agent.wait()
        except KeyboardInterrupt:pass
        finally:
            if agent and agent.poll() is None:
                agent.terminate()
                try:agent.wait(timeout=5)
                except subprocess.TimeoutExpired:agent.kill();agent.wait()
            signal.signal(signal.SIGTERM,previous)
            if started and not external_stop:
                try:
                    choice=choose_option(['Keep allocation for later (unload model)','Release allocation and stop background session'],'Session finished') if sys.stdin.isatty() else 0
                except (ValueError,KeyboardInterrupt):choice=0
                rpc(path,'stop' if choice==0 else 'release',client_pid=os.getpid())
                print('Allocation retained.' if choice==0 else 'Resources released.')


def monitor(args):
    if args.logs:
        path=path_for(args.logs);subprocess.call(['tail','-n','60','-f',str(path/'session.log')]);return
    entries=[]
    for p in sorted(STORE.glob('*/session.json'),key=lambda p:int(p.parent.name)):
        data=json.loads(p.read_text())
        if data.get('phase')=='RELEASED':continue
        try:data=rpc(p.parent,'status')
        except (OSError,RuntimeError):data['phase']='DAEMON OFFLINE'
        entries.append(data)
        print(f"{data['id']:>3}  {data['host']}  GPUs={data['gpus']}  {data['phase']}  model={data.get('model') or '-'}  job={data.get('allocation',{}).get('job_id','-')}"+('  '+data['error'] if data.get('error') else ''))
    number=args.kill
    if not number and not args.list and entries and sys.stdin.isatty():
        idx=choose_option(['Exit']+[str(x['id'])+' — '+x['host'] for x in entries],'Manage session')
        if not idx:return
        number=entries[idx-1]['id']
    if number:
        path=path_for(number)
        release=args.release or (sys.stdin.isatty() and choose_option(['Stop model, keep allocation','Release resources and stop background session'],'Stop session')==1)
        if release:
            try:rpc(path,'release')
            except (ConnectionRefusedError,FileNotFoundError):
                data=json.loads((path/'session.json').read_text());remote(config(data['config']),data['token'],data['remote_port'],'release')
                for prefix in ('provider','client'):
                    pid=data.get(prefix+'_pid');born=data.get(prefix+'_identity')
                    if pid and born and identity(pid)==born:
                        try:os.kill(pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                data['phase']='RELEASED';write(path/'session.json',data)
        else:rpc(path,'stop')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    alloc=sub.add_parser('allocate');alloc.add_argument('--config',default=os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/away.toml')))
    alloc.add_argument('--host');alloc.add_argument('--connection',choices=['ssh','local']);alloc.add_argument('--restart',action='store_true');alloc.add_argument('--gpus',type=int)
    run=sub.add_parser('run');run.add_argument('--session','-s',required=True,type=int);run.add_argument('--model');run.add_argument('--mtp',choices=['auto','on','off']);run.add_argument('agent_args',nargs=argparse.REMAINDER)
    mon=sub.add_parser('monitor');mon.add_argument('--list',action='store_true');mon.add_argument('--logs',type=int);mon.add_argument('--kill',type=int);mon.add_argument('--release',action='store_true')
    for name in ('daemon','provider'):sub.add_parser(name).add_argument('path',type=Path)
    args=parser.parse_args()
    try:
        if args.command=='allocate':allocate(args)
        elif args.command=='run':
            if args.agent_args[:1]==['--']:args.agent_args=args.agent_args[1:]
            run_agent(args)
        elif args.command=='monitor':monitor(args)
        elif args.command=='daemon':daemon(args.path)
        else:provider(args.path)
    except (ValueError,RuntimeError,OSError,KeyboardInterrupt) as exc:
        print(str(exc) or 'Canceled',file=sys.stderr);return 1
    return 0

if __name__=='__main__':sys.exit(main())

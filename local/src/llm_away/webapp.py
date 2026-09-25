"""Optional loopback dashboard and PTY terminal for LLM-AWAY."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import hashlib
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import pty
import secrets
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import threading
import time
import sys
import webbrowser
from urllib.request import build_opener, ProxyHandler

# Provider and tunnel traffic must stay local, even on hosts with HTTP proxies.
urlopen = build_opener(ProxyHandler({})).open
from urllib.parse import parse_qs, urlparse

from . import cleanup, monitor_ui, resources

ROOT=Path(__file__).resolve().parents[2]
WEB_ROOT=ROOT/'web'/'dist'
LOGO_PATH=ROOT.parent/'docs'/'llm-away-logo.jpeg'
JOBS={}
JOBS_LOCK=threading.Lock()


def source_revision():
    digest=hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(source.name.encode());digest.update(source.read_bytes())
    return digest.hexdigest()


# Capture the code generation at import, not when an HTTP request arrives.
RUNTIME_REVISION=source_revision()


def start_job(label, function):
    identifier=secrets.token_urlsafe(12)
    with JOBS_LOCK:JOBS[identifier]={'state':'running','label':label,'message':''}
    def run():
        try:
            result=function()
            message=str(result if result is not None else label+' completed.')
            with JOBS_LOCK:JOBS[identifier].update(state='done',message=message)
        except Exception as exc:
            with JOBS_LOCK:JOBS[identifier].update(state='error',message=str(exc))
    threading.Thread(target=run,daemon=True).start()
    return identifier


def job_status(identifier):
    with JOBS_LOCK:item=JOBS.get(identifier)
    if item is None:raise ValueError('Unknown operation')
    return dict(item)


def allocation_options():
    config_path=os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/model.toml'))
    cfg=resources.load_config(config_path)
    from .host_store import read_store
    store=read_store(config_path)
    hosts=[]
    for host in cfg.host_configs():
        if host.name not in hosts:hosts.append(host.name)
    return {
        'hosts':hosts,
        'default_connection':'ssh' if hosts else 'local',
        'default_host':store.get('active') or (cfg.active_host if cfg.active_host in hosts else '') or (hosts[0] if hosts else ''),
        'default_gpus':cfg.slurm.gpus or (0 if cfg.llamacpp.backend=='cpu' else 1),
        'slurm_options':shlex.join(cfg.slurm.custom_options),
        'backend':cfg.backend_type,
        'available_clis':[name for name in ('codex','claude') if shutil.which(name)],
    }


def model_options(number):
    path=session_path(number);data=json.loads((path/'session.json').read_text())
    if data.get('native'):
        selection={}
        try:selection=json.loads((path/'agent-selection.json').read_text())
        except (OSError,ValueError):pass
        rag=selection.get('rag_config') or {}
        return {'native':True,'models':[],'current':'','server_options':'',
                'agent_cli':selection.get('cli','auto'),'agent_workdir':selection.get('cwd',''),
                'rag':os.pathsep.join(rag.get('paths') or []),'rag_threads':rag.get('threads',2),
                'rag_memory_gb':rag.get('memory_gb',0),'rag_gpu':bool(rag.get('gpu',False))}
    cfg=resources.config(data['config'])
    current=resources.load_config(os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/model.toml')))
    try:current=current.with_host(cfg.ssh.host)
    except ValueError:pass
    cfg=resources.replace(cfg,llamacpp=resources.replace(cfg.llamacpp,
        model_batch_defaults=current.llamacpp.model_batch_defaults,
        model_host_batch_defaults=current.llamacpp.model_host_batch_defaults))
    models=resources.discover_models(cfg)
    public=[]
    for model in models:
        mtp=model.get('mtp',{})
        public.append({'name':model['name'],'alias':model['alias'],'size_bytes':model['size_bytes'],
                       'context_size':model.get('context_size',0),'mtp':{
                           'configured':bool(mtp.get('configured')),'available':bool(mtp.get('available')),
                           'toggle_supported':bool(mtp.get('toggle_supported')),
                           'embedded':bool(mtp.get('embedded'))}})
    from . import shared_sessions
    allocation=shared_sessions.current(path)
    owner=allocation.get('owner') or {}
    foreign=bool(owner and owner.get('id')!=shared_sessions.client_identity())
    loaded=allocation.get('model_state') in ('STARTING','LOADING','LOADED','READY','RUNNING')
    selection={}
    try:selection=json.loads((path/'agent-selection.json').read_text())
    except (OSError,ValueError):pass
    rag=selection.get('rag_config') or {}
    from .model_preferences import read
    preferences=read(cfg)
    return {'remote_rags':allocation.get('remote_rags',{}),'preferences':preferences,'container_path':preferences.get('_container_path',cfg.llamacpp.container),
            'build_mode':'container' if cfg.llamacpp.container else 'native','native':False,'models':public,'current':data.get('model') or cfg.llamacpp.model_name,'loaded':loaded,
            'remote_owner':owner.get('label','') if foreign else '',
            'expected_owner':owner.get('id',''),'expected_generation':allocation.get('generation',''),
            'can_attach':allocation.get('model_state') in ('STARTING','LOADING','LOADED','READY') and bool(allocation.get('session')),
            'agent_location':selection.get('target_location',selection.get('location','local')),
            'agent_cli':selection.get('cli','auto'),'agent_workdir':selection.get('cwd',''),
            'rag':os.pathsep.join(rag.get('paths') or []),'rag_threads':rag.get('threads',2),
            'rag_memory_gb':rag.get('memory_gb',0),'rag_gpu':bool(rag.get('gpu',False)),
            'rag_compute':rag.get('compute'),'rag_paths_location':rag.get('paths_location'),
            'server_options_by_model':{m['name']:shlex.join(resources.model_server_options(cfg,m['name'])) for m in models}}


def allocate_browser(settings):
    number=resources.allocate_from_wizard(settings)
    return f'Session {number} is allocating.'


def load_browser_model(number, settings):
    number=int(number);path=session_path(number)
    data=json.loads((path/'session.json').read_text())
    mode=settings.get('build_mode')
    container=None
    if mode is not None:
        if mode not in ('container','native'):raise ValueError('Choose container or native build')
        container=str(settings.get('container_path','')).strip() if mode=='container' else ''
        if mode=='container' and not container:raise ValueError('A container path is required')
    from . import shared_sessions
    shared_sessions.ensure_daemon(path)
    allocation=shared_sessions.current(path)
    owner=allocation.get('owner') or {}
    foreign=bool(owner and owner.get('id')!=shared_sessions.client_identity())
    attach=bool(settings.get('attach_existing'))
    if foreign and not (settings.get('replace_loaded') or attach):
        raise ValueError('Session is allocated on another machine; acknowledge replacement or attach to its model')
    if attach:
        allocation=shared_sessions.attach(path,settings.get('expected_generation',''))
    elif foreign:
        allocation=shared_sessions.claim(path,settings.get('expected_owner',''),settings.get('expected_generation',''))
    loaded=allocation.get('model_state') in ('STARTING','LOADING','LOADED','READY','RUNNING')
    if attach:
        if not loaded or not allocation.get('session'):raise ValueError('No running model to attach to')
        live=data.get('provider_identity') and resources.identity(data.get('provider_pid'))==data['provider_identity']
        if not live:resources.rpc(path,'adopt-config',expected_generation=allocation['generation'])
        shared=allocation['session']
        settings=dict(settings,model=shared['llamacpp']['model_name'] or shared['model']['name'],
                      mtp=shared['llamacpp']['mtp'],server_options=shlex.join(shared['llamacpp']['server_extra_args']))
    else:
        if loaded and not settings.get('replace_loaded'):
            raise ValueError('A model is already loaded; acknowledge replacement before starting another configuration')
        if loaded:resources.rpc(path,'stop')
    selected_rag=settings.get('remote_rag_id')
    if selected_rag:
        saved=(allocation.get('remote_rags') or {}).get(selected_rag)
        if not saved:raise ValueError('Remote RAG configuration changed; refresh the attachment dialog')
        settings=dict(settings,rag=os.pathsep.join(saved['paths']),rag_compute='remote',
                      rag_paths_location='remote',rag_threads=saved.get('threads',2),
                      rag_memory_gb=saved.get('memory_gb',0),rag_gpu='yes' if saved.get('gpu') else 'no')
    extra=shlex.split(str(settings.get('server_options','')))
    rag=[item.strip() for item in str(settings.get('rag','')).split(os.pathsep) if item.strip()]
    rag_threads=max(1,int(settings.get('rag_threads') or 2))
    rag_memory_gb=max(0,float(settings.get('rag_memory_gb') or 0))
    args=argparse.Namespace(session=number,model=settings.get('model') or None,
        mtp=settings.get('mtp','auto'),rag=rag,helper=False,quiet=True,
        rag_threads=rag_threads,rag_memory_gb=rag_memory_gb,rag_gpu=settings.get('rag_gpu')=='yes',
        rag_compute=settings.get('rag_compute') or None,rag_paths_location=settings.get('rag_paths_location') or None,
        server_extra_args=extra,mtp_prompted=True,log_helper=True,
        agent_location=settings.get('agent_location','local'),cli=settings.get('cli') or 'auto',
        agent_workdir=settings.get('agent_workdir') or None,agent_args=[],resume=True,detach=True)
    args.container=container
    args.reconfigure_agent=attach
    resources.run_agent(args)
    if data.get('config'):
        from .model_preferences import save
        save(resources.config(data['config']),args.model,settings)
    return f'Session {number} started. Use Attach to open its agent terminal.'


def restart_native_session(number, settings):
    from . import native_sessions
    path=session_path(number)
    data=json.loads((path/'session.json').read_text())
    if not data.get('native'):raise ValueError('This session is not a native CLI session')
    instructions=None
    cli=settings.get('cli')
    if cli and cli!='auto':
        config_path=os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/model.toml'))
        cfg=resources.load_config(config_path)
        base=cfg.claude.instructions if cli=='claude' else cfg.codex.instructions
        instructions=base
    native_sessions.update_settings(path,dict(settings,instructions=instructions))
    return f'Session {number} agent restarted with the saved settings.'


def session_rows():
    """Return monitor information without exposing session tokens or config."""
    rows=[]
    for row in monitor_ui.snapshots(resources.STORE):
        allocation=row.get('allocation') or {}
        telemetry=allocation.get('gpu_telemetry') or {}
        agent=allocation.get('prompt') or {}
        model_state=display_state(row,allocation)
        rows.append({
            'id':row['id'], 'host':allocation.get('worker_host') or row.get('host',''), 'gpus':row.get('gpus',0),
            'phase':row.get('phase',''), 'model':row.get('model',''),
            'native':row.get('native',False), 'native_cli':row.get('native_cli',''),
            'job_id':'' if row.get('config',{}).get('backend_type')=='direct' else allocation.get('job_id',''), 'node':allocation.get('worker_host') or allocation.get('host',''),
            'model_state':model_state, 'busy':row.get('_busy',False),
            'terminal':row.get('_terminal',False), 'error':row.get('error',''),
            'agent_text':str(agent.get('text') or '')[-32768:],
            'gpu_lines':telemetry.get('lines',[]), 'gpu_timestamp':telemetry.get('timestamp'),
            'time_left':monitor_ui.time_left_text(row),
            'details':safe_details(row),
        })
    return rows


def safe_details(row):
    """Useful allocation detail, deliberately excluding config and credentials."""
    allocation=row.get('allocation') or {}
    cfg=row.get('config') or {}
    slurm=cfg.get('slurm') or {}
    llama=cfg.get('llamacpp') or {}
    slurm_flags=['--nodes='+str(slurm.get('nodes',1)),'--gres=gpu:'+str(slurm.get('gpus',row.get('gpus',0)))]
    if slurm.get('partition'):slurm_flags.append('--partition='+str(slurm['partition']))
    if slurm.get('exclusive'):slurm_flags.append('--exclusive')
    slurm_flags.extend(slurm.get('custom_options') or [])
    pending=display_state(row,allocation)=='PENDING'
    if pending and (row.get('_busy') or row.get('_terminal')):agent_state='queued; waiting for resources/model'
    elif row.get('_terminal'):agent_state='running'
    elif row.get('_busy'):agent_state='starting'
    else:agent_state='not started'
    lines=[
        'Session: '+str(row.get('id','—')),
        'Host: '+str(allocation.get('worker_host') or row.get('host') or '—'),
        'Shared allocation: '+str(row.get('token','')[:12] or '—'),
        'Backend: '+str(row.get('config',{}).get('backend_type','—')),
        'Assigned node: '+str(allocation.get('worker_host') or allocation.get('host') or 'pending'),
        ('Worker PID: '+str(allocation.get('worker_pid') or allocation.get('job_id') or '—') if row.get('config',{}).get('backend_type')=='direct' else 'Job: '+str(allocation.get('job_id') or '—')),
        'GPUs: '+str(row.get('gpus',0)),
        'Scheduler state: '+str(allocation.get('slurm_state') or row.get('phase') or '—'),
        'Model: '+str(row.get('model') or 'none'),
        'Model state: '+display_state(row,allocation),
        'Local monitor PID: '+str(row.get('pid') or '—'),
        'Model loader PID: '+str(row.get('provider_pid') or '—'),
        'Model flags: '+(shlex.join(llama.get('server_extra_args') or []) if row.get('model') else ''),
        'Slurm flags: '+(shlex.join(slurm_flags) if cfg.get('backend_type')=='slurm_server' else '—'),
        'Time left: '+monitor_ui.time_left_text(row),
        'Agent: '+agent_state,
        'RAG: '+('enabled' if row.get('rag_enabled') else 'disabled'),
    ]
    if row.get('_terminal'):lines.append('Agent terminal: open'+(' (waiting for model)' if pending else ''))
    if row.get('_stale'): lines.append('Status: daemon may be offline or stale')
    if row.get('error'): lines.append('Error: '+str(row['error']))
    return lines


def display_state(row, allocation):
    """Use model states only after the scheduler has assigned resources."""
    slurm_state=str(allocation.get('slurm_state') or '').upper()
    phase=str(row.get('phase') or '').upper()
    if slurm_state in ('PENDING','CONFIGURING') or (not allocation.get('host') and phase in ('PENDING','STARTING')):
        return 'PENDING'
    return str(allocation.get('model_state') or row.get('phase') or 'IDLE').upper()


def session_path(number):
    number=int(number)
    if number<0:raise ValueError('Invalid session')
    path=resources.path_for(number)
    if not (path/'session.json').exists():raise ValueError('Unknown session')
    return path


def read_log(number, name):
    if name not in ('session','helper','agent','rag','model-queries'):raise ValueError('Unknown log')
    session=session_path(number);path=session/(name+'.log')
    if name=='rag':
        try:
            remote_text=json.loads((session/'session.json').read_text()).get('allocation',{}).get('rag_log','')
            if remote_text:return {'path':'remote allocation: rag.log','content':remote_text}
        except (OSError,ValueError):pass
    if not path.exists():return {'path':str(path),'content':'No '+name+' log yet.'}
    with path.open('rb') as stream:
        stream.seek(0,os.SEEK_END);size=stream.tell();stream.seek(max(0,size-1_000_000))
        content=stream.read().decode('utf-8','replace')
    return {'path':str(path),'content':content}


def cleanup_items():
    items=[]
    for saved in sorted(resources.STORE.glob('[0-9]*/session.json'),key=lambda p:int(p.parent.name)):
        try:
            data=json.loads(saved.read_text())
            if data.get('phase')=='RELEASED' and not cleanup.busy(saved.parent):
                cleanup.add_session_items(items,saved.parent,False)
        except (OSError,ValueError):continue
    return items


CLEANUP_PLANS={}


def cleanup_preview():
    items=cleanup_items()
    plan=cleanup.preview(items)
    token=secrets.token_urlsafe(24)
    for old in list(CLEANUP_PLANS):
        if time.monotonic()-CLEANUP_PLANS[old][0]>600:CLEANUP_PLANS.pop(old,None)
    CLEANUP_PLANS[token]=(time.monotonic(),items,plan)
    return {'token':token,'paths':plan['paths'],'actions':[label for _,_,label in items]}


def cleanup_released(token=None):
    saved=CLEANUP_PLANS.pop(token,None)
    if not saved or time.monotonic()-saved[0]>600:raise ValueError('Review cleanup files again; preview expired')
    _,items,plan=saved
    if cleanup.preview(items)!=plan:raise ValueError('Files changed since preview; review cleanup again')
    output=io.StringIO()
    with contextlib.redirect_stdout(output):cleanup.execute(items,list(range(len(items))),expected_remote=plan['remote_paths'])
    return output.getvalue().strip() or 'Cleanup completed.'


def run_action(action, number=None):
    if action=='discover-remote':
        from .shared_sessions import discover
        return discover()
    if action=='refresh-monitor':return resources.refresh_monitor()
    if action=='cleanup':raise ValueError('Cleanup requires a reviewed preview')
    if number is None:raise ValueError('Select an allocation first')
    number=int(number);path=session_path(number)
    if action=='refresh-session':return resources.refresh_session(number) or f'Session {number} refreshed.'
    if action=='set-helper':
        from .serve_registration import register
        register(number,log_helper=True);return f'Session {number} is available as a helper.'
    if action=='restart':return resources.restart_session(number) or f'Session {number} restarting.'
    if action=='reconnect':return resources.reconnect_session(number)
    if action=='unload':resources.rpc(path,'stop');return f'Model unloaded from session {number}.'
    if action=='release':resources.release_session(number);return f'Session {number} released.'
    raise ValueError('Unknown action')


class BrowserTerminal:
    def __init__(self):
        master,slave=pty.openpty()
        self.master=master;self.lock=threading.Lock();self.data='';self.base=0
        env=dict(os.environ,TERM='xterm-256color',COLORTERM='truecolor')
        shell=env.get('SHELL','/bin/zsh')
        self.process=subprocess.Popen([shell,'-l'],cwd=ROOT,stdin=slave,stdout=slave,stderr=slave,
                                      start_new_session=True,env=env,close_fds=True)
        os.close(slave)
        threading.Thread(target=self._read,daemon=True).start()

    def _read(self):
        while True:
            try:chunk=os.read(self.master,65536)
            except OSError:break
            if not chunk:break
            with self.lock:
                self.data+=chunk.decode('utf-8','replace')
                if len(self.data)>1_000_000:
                    trim=len(self.data)-1_000_000;self.data=self.data[trim:];self.base+=trim

    def write(self,data):
        raw=data.encode('utf-8')
        if len(raw)>16384:raise ValueError('Terminal input is too large')
        os.write(self.master,raw)

    def resize(self,cols,rows):
        cols=max(40,min(int(cols),400));rows=max(10,min(int(rows),200))
        fcntl.ioctl(self.master,termios.TIOCSWINSZ,struct.pack('HHHH',rows,cols,0,0))

    def output(self,offset):
        with self.lock:
            offset=max(self.base,min(int(offset),self.base+len(self.data)))
            return {'data':self.data[offset-self.base:], 'offset':self.base+len(self.data),
                    'alive':self.process.poll() is None}

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGKILL)
                self.process.wait(timeout=2)
        try:os.close(self.master)
        except OSError:pass


# Imported lazily so this module remains importable on platforms without a PTY.
import termios
TERMINALS={}
TERMINALS_LOCK=threading.Lock()


def terminal(identifier):
    with TERMINALS_LOCK:
        item=TERMINALS.get(identifier)
    if item is None:raise ValueError('Unknown terminal')
    return item


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(WEB_ROOT),**kwargs)

    def log_message(self,*args):
        return

    def end_headers(self):
        asset_path=urlparse(self.path).path
        if asset_path in ('','/') or Path(asset_path).suffix in ('.html','.js','.css'):
            self.send_header('Cache-Control','no-cache, no-store, must-revalidate')
        super().end_headers()

    def _json(self,payload,status=HTTPStatus.OK):
        data=json.dumps(payload).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(data)

    def _body(self):
        size=min(int(self.headers.get('Content-Length','0')),65536)
        return json.loads(self.rfile.read(size) or b'{}')

    def do_GET(self):
        parsed=urlparse(self.path)
        try:
            if parsed.path=='/api/runtime':
                self._json({'revision':RUNTIME_REVISION});return
            if parsed.path=='/api/sessions':self._json({'sessions':session_rows(),'time':time.time()});return
            if parsed.path=='/api/gpu-history':
                from .gpu_history import append, read
                query=parse_qs(parsed.query);path=session_path(query.get('session',[''])[0])
                data=json.loads((path/'session.json').read_text())
                append(path,(data.get('allocation') or {}).get('gpu_telemetry'))
                self._json(read(path,query.get('after',['0'])[0]));return
            if parsed.path in ('/logo','/favicon.jpeg'):
                data=LOGO_PATH.read_bytes();self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type','image/jpeg');self.send_header('Content-Length',str(len(data)))
                self.send_header('Cache-Control','public, max-age=86400');self.end_headers();self.wfile.write(data);return
            if parsed.path=='/api/options':self._json(allocation_options());return
            if parsed.path=='/api/cleanup-preview':self._json(cleanup_preview());return
            if parsed.path=='/api/models':
                query=parse_qs(parsed.query);self._json(model_options(query.get('session',[''])[0]));return
            if parsed.path=='/api/job':
                query=parse_qs(parsed.query);self._json(job_status(query.get('id',[''])[0]));return
            if parsed.path=='/api/log':
                query=parse_qs(parsed.query)
                self._json(read_log(query.get('session',[''])[0],query.get('name',['session'])[0]));return
            if parsed.path=='/api/terminal/output':
                query=parse_qs(parsed.query);item=terminal(query.get('id',[''])[0])
                self._json(item.output(query.get('offset',['0'])[0]));return
            super().do_GET()
        except Exception as exc:self._json({'error':str(exc)},HTTPStatus.BAD_REQUEST)

    def do_POST(self):
        try:
            body=self._body()
            if self.path in ('/api/browser/ping','/api/browser/close'):
                identifier=str(body.get('id',''))[:100]
                with self.server.browser_lock:
                    if self.path.endswith('/close'):
                        self.server.browsers.pop(identifier,None)
                    else:
                        self.server.browsers[identifier]=time.monotonic()
                        self.server.browser_seen=True
                    self.server.browser_activity=time.monotonic()
                self._json({'ok':True});return
            if self.path=='/api/terminal':
                item=BrowserTerminal();identifier=secrets.token_urlsafe(18)
                with TERMINALS_LOCK:TERMINALS[identifier]=item
                self._json({'id':identifier});return
            if self.path=='/api/terminal/input':
                terminal(body['id']).write(str(body.get('data','')));self._json({'ok':True});return
            if self.path=='/api/terminal/resize':
                terminal(body['id']).resize(body.get('cols',120),body.get('rows',30));self._json({'ok':True});return
            if self.path=='/api/actions':
                action=str(body.get('action',''))
                if action in ('cleanup','release') and not body.get('confirmed'):
                    raise ValueError('Confirmation required')
                print(json.dumps(dict(event='dashboard_action',time=time.time(),action=action,
                    session=body.get('session'),peer=self.client_address[0],pid=os.getpid())),flush=True)
                if action=='cleanup':
                    identifier=start_job(action,lambda:cleanup_released(body.get('preview_token')))
                else:identifier=start_job(action,lambda:run_action(action,body.get('session')))
                self._json({'job':identifier});return
            if self.path=='/api/allocate':
                settings=dict(body.get('settings') or {})
                identifier=start_job('Allocate resources',lambda:allocate_browser(settings))
                self._json({'job':identifier});return
            if self.path=='/api/load':
                number=body.get('session');settings=dict(body.get('settings') or {})
                identifier=start_job('Start session',lambda:load_browser_model(number,settings))
                self._json({'job':identifier});return
            if self.path=='/api/native':
                number=body.get('session');settings=dict(body.get('settings') or {})
                identifier=start_job('Restart native agent',lambda:restart_native_session(number,settings))
                self._json({'job':identifier});return
            if self.path=='/api/chat':
                resources.submit_prompt(int(body['session']),str(body.get('prompt','')))
                self._json({'ok':True});return
            self._json({'error':'Not found'},HTTPStatus.NOT_FOUND)
        except Exception as exc:self._json({'error':str(exc)},HTTPStatus.BAD_REQUEST)

    def do_DELETE(self):
        try:
            identifier=parse_qs(urlparse(self.path).query).get('id',[''])[0]
            with TERMINALS_LOCK:item=TERMINALS.pop(identifier,None)
            if item:item.close()
            self._json({'ok':True})
        except Exception as exc:self._json({'error':str(exc)},HTTPStatus.BAD_REQUEST)


def port_listening(port):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.5):return True
    except OSError:return False


def dashboard_running(port):
    # Session enumeration can block on SSH. A busy dashboard still owns its port.
    for endpoint,key,kind in (('runtime','revision',str),('sessions','sessions',list)):
        try:
            with urlopen(f'http://127.0.0.1:{port}/api/{endpoint}',timeout=.5) as response:
                payload=json.loads(response.read())
            if isinstance(payload.get(key),kind):return True
        except Exception:pass
    return False


def dashboard_current(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/api/runtime',timeout=1) as response:
            return json.loads(response.read()).get('revision')==RUNTIME_REVISION
    except Exception:return False


def dashboard_processes(output,port):
    """Select only this user's dedicated dashboard processes on the requested port."""
    matches=[]
    for line in output.splitlines():
        try:
            pid,uid,command=line.strip().split(None,2)
            args=shlex.split(command)
            module=args.index('-m')
            if not Path(args[0]).name.startswith(('python','pypy')):continue
            if args[module+1]!='llm_away.webapp' or '--restart' in args:continue
            actual=int(args[args.index('--port')+1]) if '--port' in args else int(next((a.split('=',1)[1] for a in args if a.startswith('--port=')),'8766'))
            if int(uid)==os.getuid() and int(pid)!=os.getpid() and actual==port:matches.append(int(pid))
        except (ValueError,IndexError):continue
    return matches


def stop_dashboard(port):
    output=subprocess.check_output(['ps','-ww','-eo','pid=,uid=,args='],text=True)
    processes=dashboard_processes(output,port)
    if not processes:
        if not port_listening(port):return  # Browser shutdown won the restart race.
        listener=''
        command=(['ss','-ltnp',f'sport = :{port}'] if shutil.which('ss') else
                 ['lsof','-nP',f'-iTCP:{port}','-sTCP:LISTEN'] if shutil.which('lsof') else None)
        if command:
            try:listener=subprocess.run(command,capture_output=True,text=True,timeout=3).stdout.strip()[-1500:]
            except (OSError,subprocess.TimeoutExpired):pass
        raise RuntimeError(f'Port {port} is occupied but no owned LLM-AWAY dashboard process was found. '
                           f'Check its listener with ss -ltnp \'sport = :{port}\' or use res-mon-web --port {port+1}. '
                           'No unrelated process was stopped.'+('\nListener: '+listener if listener else ''))
    for pid in processes:
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+5
    while port_listening(port):
        if time.monotonic()>=deadline:raise RuntimeError('Dashboard has not stopped yet; retry shortly')
        time.sleep(.1)


def open_dashboard(port):
    url=f'http://127.0.0.1:{port}/'
    print(f'LLM-AWAY dashboard: {url}',flush=True)
    quiet=None;saved=None
    try:
        saved=os.dup(2);quiet=os.open(os.devnull,os.O_WRONLY);os.dup2(quiet,2)
    except OSError:
        if quiet is not None:os.close(quiet);quiet=None
        if saved is not None:os.close(saved);saved=None
    try:webbrowser.open(url)
    finally:
        if saved is not None and quiet is not None:os.dup2(saved,2)
        if saved is not None:os.close(saved)
        if quiet is not None:os.close(quiet)


def serve(port=8766,open_browser=True):
    if not WEB_ROOT.exists():raise FileNotFoundError('Dashboard assets are missing')
    if dashboard_running(port):
        if open_browser:open_dashboard(port)
        return
    server=ThreadingHTTPServer(('127.0.0.1',port),DashboardHandler)
    server.browser_lock=threading.Lock()
    server.browsers={}
    server.browser_seen=False
    server.browser_activity=time.monotonic()
    def browser_watch():
        started=time.monotonic()
        while True:
            time.sleep(1)
            now=time.monotonic()
            with server.browser_lock:
                server.browsers={key:seen for key,seen in server.browsers.items() if now-seen<180}
                idle=not server.browsers and now-server.browser_activity>5 and (server.browser_seen or now-started>30)
            if idle:
                server.shutdown();return
    threading.Thread(target=browser_watch,daemon=True).start()
    if open_browser:open_dashboard(port)
    else:print(f'LLM-AWAY dashboard: http://127.0.0.1:{port}',flush=True)
    try:server.serve_forever()
    finally:
        server.server_close()
        with TERMINALS_LOCK:items=list(TERMINALS.values());TERMINALS.clear()
        for item in items:item.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument("--restart",action="store_true",help="Restart the dashboard process while retaining resource allocations")
    parser.add_argument("--serve",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('--port must be 1024–65535')
    if args.serve:
        serve(args.port,open_browser=False)
        return
    running=dashboard_running(args.port)
    if (args.restart and port_listening(args.port)) or (running and not dashboard_current(args.port)):
        print('Restarting dashboard to load current code; resource allocations are retained.',flush=True)
        stop_dashboard(args.port)
        running=False
    if not running:
        log=Path(__file__).resolve().parents[2]/'run/dashboard.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        def log_tail():
            try:return log.read_text(errors='replace').strip()[-2000:]
            except OSError:return ''
        with log.open('a') as output:
            child=subprocess.Popen([sys.executable,'-m','llm_away.webapp','--serve','--port',str(args.port)],
                stdin=subprocess.DEVNULL,stdout=output,stderr=output,start_new_session=True)
        for _ in range(100):
            if dashboard_running(args.port):break
            if child.poll() is not None:
                raise RuntimeError(f'Dashboard failed; see {log}\n{log_tail()}')
            time.sleep(.1)
        else:raise RuntimeError(f'Dashboard startup timed out; see {log}\n{log_tail()}')
    open_dashboard(args.port)


if __name__=='__main__':main()

"""Optional loopback dashboard and PTY terminal for LLM-AWAY."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
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
import struct
import subprocess
import threading
import time
import webbrowser
from urllib.request import urlopen
from urllib.parse import parse_qs, urlparse

from . import cleanup, monitor_ui, resources

ROOT=Path(__file__).resolve().parents[2]
WEB_ROOT=ROOT/'web'/'dist'
JOBS={}
JOBS_LOCK=threading.Lock()


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
        return {'native':True,'models':[],'current':'','server_options':''}
    cfg=resources.config(data['config']);models=resources.discover_models(cfg)
    public=[]
    for model in models:
        mtp=model.get('mtp',{})
        public.append({'name':model['name'],'alias':model['alias'],'size_bytes':model['size_bytes'],
                       'context_size':model.get('context_size',0),'mtp':{
                           'configured':bool(mtp.get('configured')),'available':bool(mtp.get('available')),
                           'toggle_supported':bool(mtp.get('toggle_supported')),
                           'embedded':bool(mtp.get('embedded'))}})
    allocation=data.get('allocation') or {}
    loaded=bool(data.get('model') and allocation.get('model_state') in ('LOADING','LOADED','READY','RUNNING'))
    return {'native':False,'models':public,'current':data.get('model') or cfg.llamacpp.model_name,'loaded':loaded,
            'server_options_by_model':{m['name']:shlex.join(resources.model_server_options(cfg,m['name'])) for m in models}}


def allocate_browser(settings):
    number=resources.allocate_from_wizard(settings)
    return f'Session {number} is allocating.'


def load_browser_model(number, settings):
    number=int(number);path=session_path(number)
    data=json.loads((path/'session.json').read_text())
    allocation=data.get('allocation') or {}
    loaded=bool(data.get('model') and allocation.get('model_state') in ('LOADING','LOADED','READY','RUNNING'))
    if loaded and not settings.get('replace_loaded'):
        raise ValueError('A model is already loaded; acknowledge replacement before starting another configuration')
    if loaded:resources.rpc(path,'stop')
    extra=shlex.split(str(settings.get('server_options','')))
    rag=[item.strip() for item in str(settings.get('rag','')).split(os.pathsep) if item.strip()]
    args=argparse.Namespace(session=number,model=settings.get('model') or None,
        mtp=settings.get('mtp','auto'),rag=rag,helper=False,quiet=True,
        server_extra_args=extra,mtp_prompted=True,log_helper=True,
        agent_location=settings.get('agent_location','local'),cli=settings.get('cli') or 'auto',
        agent_workdir=settings.get('agent_workdir') or None,agent_args=[],resume=True,detach=True)
    resources.run_agent(args)
    return f'Session {number} started. Use Attach to open its agent terminal.'


def session_rows():
    """Return monitor information without exposing session tokens or config."""
    rows=[]
    for row in monitor_ui.snapshots(resources.STORE):
        allocation=row.get('allocation') or {}
        telemetry=allocation.get('gpu_telemetry') or {}
        agent=allocation.get('prompt') or {}
        model_state=display_state(row,allocation)
        rows.append({
            'id':row['id'], 'host':row.get('host',''), 'gpus':row.get('gpus',0),
            'phase':row.get('phase',''), 'model':row.get('model',''),
            'job_id':allocation.get('job_id',''), 'node':allocation.get('host',''),
            'model_state':model_state, 'busy':row.get('_busy',False),
            'terminal':row.get('_terminal',False), 'error':row.get('error',''),
            'agent_text':str(agent.get('text') or '')[-32768:],
            'gpu_lines':telemetry.get('lines',[]), 'gpu_timestamp':telemetry.get('timestamp'),
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
        'Host: '+str(row.get('host') or '—'),
        'Assigned node: '+str(allocation.get('host') or 'pending'),
        'Job: '+str(allocation.get('job_id') or '—'),
        'GPUs: '+str(row.get('gpus',0)),
        'Scheduler state: '+str(allocation.get('slurm_state') or row.get('phase') or '—'),
        'Model: '+str(row.get('model') or 'none'),
        'Model state: '+display_state(row,allocation),
        'Local monitor PID: '+str(row.get('pid') or '—'),
        'Model loader PID: '+str(row.get('provider_pid') or '—'),
        'Model flags: '+(shlex.join(llama.get('server_extra_args') or []) if row.get('model') else ''),
        'Slurm flags: '+(shlex.join(slurm_flags) if cfg.get('backend_type')=='slurm_server' else '—'),
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
    if name not in ('session','helper','agent'):raise ValueError('Unknown log')
    path=session_path(number)/(name+'.log')
    if not path.exists():return {'path':str(path),'content':'No '+name+' log yet.'}
    with path.open('rb') as stream:
        stream.seek(0,os.SEEK_END);size=stream.tell();stream.seek(max(0,size-1_000_000))
        content=stream.read().decode('utf-8','replace')
    return {'path':str(path),'content':content}


def cleanup_released():
    """Clean verified inactive leftovers without the curses confirmation UI."""
    items=[]
    for saved in sorted(resources.STORE.glob('[0-9]*/session.json'),key=lambda p:int(p.parent.name)):
        try:
            data=json.loads(saved.read_text())
            if data.get('phase')=='RELEASED' and not cleanup.busy(saved.parent):
                cleanup.add_session_items(items,saved.parent,False)
        except (OSError,ValueError):continue
    if not items:return 'No safely identifiable released-session leftovers.'
    output=io.StringIO()
    with contextlib.redirect_stdout(output):cleanup.execute(items,list(range(len(items))))
    return output.getvalue().strip() or 'Cleanup completed.'


def run_action(action, number=None):
    if action=='refresh-monitor':return resources.refresh_monitor()
    if action=='cleanup':return cleanup_released()
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
            if parsed.path=='/api/sessions':self._json({'sessions':session_rows(),'time':time.time()});return
            if parsed.path=='/logo':
                path=Path.home()/'Desktop'/'vit_man.jpeg'
                data=path.read_bytes();self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type','image/jpeg');self.send_header('Content-Length',str(len(data)))
                self.send_header('Cache-Control','no-cache');self.end_headers();self.wfile.write(data);return
            if parsed.path=='/api/options':self._json(allocation_options());return
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
                identifier=start_job(action,lambda:run_action(action,body.get('session')))
                self._json({'job':identifier});return
            if self.path=='/api/allocate':
                settings=dict(body.get('settings') or {})
                identifier=start_job('Allocate resources',lambda:allocate_browser(settings))
                self._json({'job':identifier});return
            if self.path=='/api/load':
                number=body.get('session');settings=dict(body.get('settings') or {})
                identifier=start_job('Start session',lambda:load_browser_model(number,settings))
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


def dashboard_running(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/api/sessions',timeout=.5) as response:
            payload=json.loads(response.read())
        return isinstance(payload.get('sessions'),list)
    except Exception:return False


def open_dashboard(port):
    url=f'http://127.0.0.1:{port}/'
    print(f'LLM-AWAY dashboard: {url}',flush=True)
    webbrowser.open(url)


def serve(port=8766,open_browser=True):
    if not WEB_ROOT.exists():raise FileNotFoundError('Dashboard assets are missing')
    if dashboard_running(port):
        if open_browser:open_dashboard(port)
        return
    server=ThreadingHTTPServer(('127.0.0.1',port),DashboardHandler)
    if open_browser:open_dashboard(port)
    else:print(f'LLM-AWAY dashboard: http://127.0.0.1:{port}',flush=True)
    try:server.serve_forever()
    finally:
        with TERMINALS_LOCK:items=list(TERMINALS.values());TERMINALS.clear()
        for item in items:item.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('--port must be 1024–65535')
    serve(args.port)


if __name__=='__main__':main()

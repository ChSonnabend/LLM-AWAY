"""One CPU-only integration check: two independent retained allocations."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from dataclasses import asdict, replace
from llm_away.config import AppConfig
from llm_away.resources import rpc, write

REPO=Path(__file__).resolve().parents[2]

class ResourceLifecycleTests(unittest.TestCase):
    def test_two_sessions_stop_reuse_and_release_independently(self):
        with tempfile.TemporaryDirectory(prefix='away-res-') as directory:
            root=Path(directory); (root/'bin').mkdir()
            for name in ('resource-control','resource-worker'):
                shutil.copy2(REPO/'remote/bin'/name,root/'bin'/name)
            fake=root/'bin/run-server'
            fake.write_text('#!'+sys.executable+'\n'+'''import json,os
from http.server import BaseHTTPRequestHandler,HTTPServer
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  data={'status':'ok'} if self.path=='/health' else {'data':[{'id':'fixture'}]}
  body=json.dumps(data).encode();self.send_response(200);self.end_headers();self.wfile.write(body)
 def log_message(self,*a):pass
HTTPServer(('127.0.0.1',int(os.environ['LLAMACPP_PORT'])),Handler).serve_forever()
''');fake.chmod(0o755)
            processes=[];paths=[];logs=[];sockets=[];expected_exits={}
            for i in range(6):
                s=socket.socket();s.bind(('127.0.0.1',0));sockets.append(s)
            ports=[s.getsockname()[1] for s in sockets]
            for s in sockets:s.close()
            try:
                for i in range(2):
                    path=root/str(i);path.mkdir();paths.append(path)
                    cfg=AppConfig();cfg=replace(cfg,backend_type='direct',ssh=replace(cfg.ssh,connection='local'),
                        remote=replace(cfg.remote,workdir=str(root)),model=replace(cfg.model,name='fixture'),
                        llamacpp=replace(cfg.llamacpp,backend='cpu',visible_devices='',model_name='fixture',container=''),
                        slurm=replace(cfg.slurm,gpus=0,nodes=1),server=replace(cfg.server,port=ports[3*i]),
                        gateway=replace(cfg.gateway,server_port=ports[3*i+1],local_port=ports[3*i+2],startup_timeout_seconds=20,cancel_on_exit=False))
                    write(path/'session.json',dict(id=i,token=uuid.uuid4().hex,config=asdict(cfg),remote_port=ports[3*i+1],host='local',gpus=0,phase='STARTING',model=''))
                    log=(path/'session.log').open('w');logs.append(log)
                    processes.append(subprocess.Popen([sys.executable,'-m','llm_away.resources','daemon',str(path)],stdout=log,stderr=log))
                def wait(predicate):
                    end=time.time()+30
                    while time.time()<end:
                        try:
                            result=predicate()
                            if result:return result
                        except (OSError,RuntimeError):pass
                        time.sleep(.2)
                    self.fail('Timed out: '+ '\n'.join((p/'session.log').read_text() for p in paths))
                for p in paths:wait(lambda:rpc(p,'status').get('allocation',{}).get('active'))
                original=[rpc(p,'status')['allocation']['job_id'] for p in paths]
                for p in paths:rpc(p,'start',model={'alias':'fixture','name':'fixture'},mtp='off',client_pid=os.getpid())
                from urllib.request import urlopen
                def healthy(port):
                    with urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as r:return r.status==200
                for i in range(2):wait(lambda:healthy(ports[3*i]))
                for path in paths:
                    token=json.loads((path/'session.json').read_text())['token']
                    wait(lambda: 'LOADED' in (root/'.state/resources'/token/'worker.state').read_text())
                # Neither graceful monitor shutdown nor a crash may unload the model/job.
                for i in range(2):
                    old=processes[i]
                    if i==0:old.terminate();expected_exits[old.pid]=0
                    else:old.kill();expected_exits[old.pid]=-signal.SIGKILL
                    self.assertEqual(old.wait(timeout=15),expected_exits[old.pid])
                    wait(lambda:json.loads((paths[i]/'session.json').read_text()).get('monitor_detached'))
                    saved=json.loads((paths[i]/'session.json').read_text())
                    self.assertTrue(healthy(ports[3*i]))
                    self.assertNotEqual(saved['phase'],'RELEASED')
                    self.assertFalse((root/'.state/resources'/saved['token']/'release').exists())
                    processes.append(subprocess.Popen([sys.executable,'-m','llm_away.resources','daemon',str(paths[i])],stdout=logs[i],stderr=logs[i]))
                    wait(lambda:rpc(paths[i],'status').get('allocation',{}).get('active'))
                    self.assertEqual(rpc(paths[i],'status')['allocation']['job_id'],original[i])
                    self.assertEqual(rpc(paths[i],'status')['provider_pid'],saved['provider_pid'])
                rpc(paths[0],'stop',client_pid=os.getpid())
                self.assertTrue(healthy(ports[3]))
                self.assertEqual(rpc(paths[0],'status')['allocation']['job_id'],original[0])
                rpc(paths[0],'start',model={'alias':'fixture','name':'fixture'},mtp='off',client_pid=os.getpid())
                wait(lambda:healthy(ports[0]))
                for p in paths:rpc(p,'release',client_pid=os.getpid())
                for child in processes:self.assertEqual(child.wait(timeout=10),expected_exits.get(child.pid,0))
                for p in paths:self.assertEqual(json.loads((p/'session.json').read_text())['phase'],'RELEASED')
            finally:
                for p in paths:
                    try:rpc(p,'release',client_pid=os.getpid())
                    except (OSError,RuntimeError):pass
                for p in (root/'.state/resources').glob('*'):
                    (p/'release').touch()
                for child in processes:
                    if child.poll() is None:child.terminate();child.wait(timeout=15)
                for log in logs:log.close()

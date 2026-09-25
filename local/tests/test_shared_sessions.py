"""Exercise remote arbitration without SSH, a scheduler, or a model download."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from llm_away.config import AppConfig
from llm_away import resources, shared_sessions, loading

CONTROL=Path(__file__).resolve().parents[2]/'remote/bin/resource-control'

class RemoteOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.token='a'*32
        self.state=self.root/'.state/resources'/self.token;self.state.mkdir(parents=True)
        cfg=AppConfig();self.cfg=replace(cfg,backend_type='direct',remote=replace(cfg.remote,workdir=str(self.root)),llamacpp=replace(cfg.llamacpp,container=''))
        (self.state/'allocation.json').write_text(json.dumps({'job_id':'123','created':time.time(),'server_port':1234}))
        (self.state/'heartbeat').touch()
        (self.state/'worker.state').write_text('IDLE host')
        self.call('status') # migrate descriptor for an existing allocation

    def call(self,action,client='one',ok=True,**extra):
        payload=dict(config=asdict(self.cfg),token=self.token,port=1234,client_id=client,client_label=client,**extra)
        result=subprocess.run([sys.executable,str(CONTROL),action],input=json.dumps(payload),text=True,capture_output=True)
        if ok:
            self.assertEqual(result.returncode,0,result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode,0)
        return result.stderr

    def test_discovery_and_reuse_preserve_running_generation(self):
        started=self.call('start');generation=started['generation']
        (self.state/'worker.state').write_text(generation+' LOADING host')
        listed=self.call('list',client='two')['sessions']
        self.assertEqual([s['token'] for s in listed],[self.token])
        self.assertNotIn('codex',listed[0]['session'])
        self.assertIn('acknowledge',self.call('stop',client='two',ok=False))
        self.call('claim',client='two',expected_owner='one',expected_generation=generation)
        self.assertEqual(self.call('start',client='two')['generation'],generation)
        self.assertEqual((self.state/'desired').read_text(),generation)
        for action in ('start','stop','prompt'):
            self.assertIn('acknowledge',self.call(action,client='one',ok=False))
        self.call('stop',client='two')
        self.assertNotEqual((self.state/'desired').read_text(),generation)

    def test_shared_attachment_preserves_owner_and_registers_remote_rag(self):
        generation=self.call('start')['generation']
        self.call('attach',client='two',expected_generation=generation)
        self.assertEqual(self.call('status')['owner']['id'],'one')
        self.assertEqual((self.state/'desired').read_text(),generation)
        config={'paths':['/remote/project'],'threads':4,'compute':'remote','paths_location':'remote'}
        registered=self.call('rag-register',client='two',rag_config=config)
        self.assertEqual(self.call('status',client='one')['remote_rags'][registered['id']]['paths'],config['paths'])
        self.assertEqual(self.call('rag-register',client='one',rag_config=config)['id'],registered['id'])
        self.assertIn('Only remote RAG',self.call('rag-register',client='two',rag_config=dict(config,compute='local'),ok=False))
        self.assertIn('acknowledge',self.call('stop',client='two',ok=False))
        self.call('stop')
        self.assertIn('Session changed',self.call('rag-register',client='two',rag_config=config,ok=False))
        self.assertIn('No running model',self.call('attach',client='two',expected_generation=generation,ok=False))

    def test_remote_terminal_and_logs_are_private_to_each_client(self):
        import hashlib
        generation=self.call('start')['generation']
        (self.state/'worker.state').write_text(generation+' LOADED host')
        (self.state/'prompt-worker.ready').write_text(json.dumps({'version':4,'tmux':True,'clis':['codex']}))
        self.call('attach',client='two',expected_generation=generation)
        terminal_ids=[]
        for client in ('one','two'):
            self.call('terminal-start',client=client,spec={'cli':'codex'})
            target=self.state/'clients'/hashlib.sha256(client.encode()).hexdigest()[:16]
            terminal_ids.append(json.loads((target/'terminal-launch.json').read_text())['terminal_id'])
            (target/'terminal-status.json').write_text(json.dumps({'status':'TERMINAL','text':client}))
            (target/'rag.log').write_text(client+' RAG')
        self.assertEqual(len(set(terminal_ids)),2)
        for client in ('one','two'):
            status=self.call('status',client=client)
            self.assertEqual(status['terminal']['text'],client)
            self.assertEqual(status['rag_log'],client+' RAG')
        self.call('terminal-stop',client='two')
        one=self.state/'clients'/hashlib.sha256(b'one').hexdigest()[:16]
        self.assertFalse((one/'terminal-stop').exists())

    def test_two_remote_agents_survive_independent_rag_reconfiguration(self):
        import os
        cli=self.root/'codex'
        cli.write_text('#!/usr/bin/env python3\nimport sys\nprint("READY",flush=True)\nfor line in sys.stdin: print("RECEIVED:"+line.strip(),flush=True)\n')
        cli.chmod(0o755)
        generation=self.call('start')['generation']
        (self.state/'worker.state').write_text(generation+' LOADED host')
        self.call('attach',client='two',expected_generation=generation)
        env=dict(os.environ,PATH=str(self.root)+os.pathsep+os.environ['PATH'])
        worker=subprocess.Popen([sys.executable,str(CONTROL.with_name('resource-prompts')),str(self.state)],
                                env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        def wait_for(check):
            deadline=time.monotonic()+8
            while time.monotonic()<deadline:
                if check():return
                time.sleep(.1)
            self.fail('Terminal worker did not reach expected state')
        try:
            wait_for(lambda:(self.state/'prompt-worker.ready').exists())
            for client in ('one','two'):
                self.call('terminal-start',client=client,spec={'cli':'codex','model':'test','cwd':str(self.root)})
                wait_for(lambda:'READY' in self.call('status',client=client).get('terminal',{}).get('text',''))
                self.call('terminal-send',client=client,prompt='hello '+client)
                wait_for(lambda:'RECEIVED:hello '+client in self.call('status',client=client)['terminal']['text'])
            self.call('terminal-stop',client='two')
            wait_for(lambda:self.call('status',client='two')['terminal']['status']=='EXITED')
            self.assertEqual(self.call('status',client='one')['terminal']['status'],'TERMINAL')
            self.assertNotIn('hello two',self.call('status',client='one')['terminal']['text'])
            self.assertEqual((self.state/'desired').read_text(),generation)
        finally:
            worker.terminate();worker.communicate(timeout=10)

    def test_legacy_loading_worker_uses_live_health_without_proxy(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading
        class Health(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200);self.end_headers();self.wfile.write(b'{"status":"ok"}')
            def log_message(self,*args):pass
        server=HTTPServer(('127.0.0.1',0),Health)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            allocation=json.loads((self.state/'allocation.json').read_text())
            allocation['server_port']=server.server_port
            (self.state/'allocation.json').write_text(json.dumps(allocation))
            generation=self.call('start')['generation']
            (self.state/'worker.state').write_text(generation+' LOADING host')
            with patch.dict('os.environ',{'http_proxy':'http://127.0.0.1:1','HTTP_PROXY':'http://127.0.0.1:1','no_proxy':'','NO_PROXY':''}):
                self.assertEqual(self.call('status')['model_state'],'LOADED')
            self.assertEqual((self.state/'desired').read_text(),generation)
        finally:server.shutdown();server.server_close();thread.join()

    def test_direct_identity_is_compute_host_and_worker_pid(self):
        (self.state/'worker.state').write_text('IDLE epn000')
        item=dict(self.call('status'),token=self.token)
        self.assertEqual(item['backend_type'],'direct')
        self.assertEqual(item['worker_host'],'epn000')
        self.assertEqual(item['worker_pid'],'123')
        self.assertEqual(item['job_id'],'')
        with patch.object(resources,'STORE',self.root/'client'),patch.object(shared_sessions,'ensure_daemon'):
            shared_sessions.import_session(self.cfg,item)
            path=resources.STORE/'1/session.json'
            data=json.loads(path.read_text())
            self.assertEqual(data['config']['ssh']['host'],'epn000')
            self.assertEqual(data['token'],self.token)
            data['host']='gateway';data['config']['ssh']['host']='gateway';path.write_text(json.dumps(data))
            self.assertEqual(shared_sessions.import_session(self.cfg,item),0)
            self.assertEqual(json.loads(path.read_text())['config']['ssh']['host'],'epn000')

    def test_managed_models_use_one_shared_inference_slot(self):
        self.cfg=replace(self.cfg,llamacpp=replace(self.cfg.llamacpp,server_extra_args=['--parallel','4']))
        generation=self.call('start')['generation']
        script=(self.state/(generation+'.sh')).read_text()
        self.assertLess(script.index('--parallel 4'),script.rindex('--parallel 1'))

    def test_release_requires_explicit_intent_and_is_audited(self):
        self.call('start')
        self.assertIn('Explicit release required',self.call('release',ok=False))
        self.assertFalse((self.state/'release').exists())
        self.assertIn('active',self.call('release',explicit_release=True,only_if_inactive=True,ok=False))
        self.assertFalse((self.state/'release').exists())
        self.assertTrue(self.call('release',explicit_release=True)['released'])
        audit=json.loads((self.state/'release-audit.jsonl').read_text())
        self.assertEqual(audit['client_id'],'one')
        self.assertTrue(audit['explicit_release'])

    def test_other_client_can_release_but_cannot_stop_model(self):
        self.call('start')
        self.assertIn('acknowledge takeover',self.call('stop',client='two',ok=False))
        self.assertIn('Explicit release required',self.call('release',client='two',ok=False))
        self.assertTrue(self.call('release',client='two',explicit_release=True)['released'])
        audit=json.loads((self.state/'release-audit.jsonl').read_text())
        self.assertEqual(audit['client_id'],'two')

    def test_shared_api_key_is_private_and_absent_from_discovery(self):
        current=self.call('start')
        self.assertTrue(current['api_key_required'])
        key=self.call('credential',client='two')['api_key']
        self.assertGreaterEqual(len(key),40)
        self.assertEqual((self.state/'api-key').stat().st_mode & 0o777,0o600)
        self.assertNotIn(key,json.dumps(current))
        self.assertNotIn(key,json.dumps(self.call('list',client='two')))
        script=(self.state/(current['generation']+'.sh')).read_text()
        self.assertIn('--api-key-file',script)
        self.assertNotIn(key,script)
        self.assertEqual(self.call('credential')['api_key'],key)

    def test_missing_optional_default_does_not_reject_running_model(self):
        generation=self.call('start')['generation']
        descriptor=json.loads((self.state/'session.json').read_text())
        descriptor['llamacpp'].pop('mtp_draft_tokens',None)
        (self.state/'session.json').write_text(json.dumps(descriptor))
        self.assertEqual(self.call('start')['generation'],generation)
        self.cfg=replace(self.cfg,llamacpp=replace(self.cfg.llamacpp,mtp_draft_tokens=4))
        self.assertIn('Stop the existing model',self.call('start',ok=False))

    def test_stale_takeover_rejected(self):
        current=self.call('start')
        self.call('claim',client='two',expected_owner='one',expected_generation=current['generation'])
        self.assertIn('Session changed',self.call('claim',client='three',expected_owner='one',expected_generation=current['generation'],ok=False))
        self.assertEqual(self.call('status')['owner']['id'],'two')

    def test_concurrent_takeovers_have_one_winner(self):
        current=self.call('start')
        children=[]
        for client in ('two','three'):
            payload=dict(config=asdict(self.cfg),token=self.token,port=1234,client_id=client,
                         expected_owner='one',expected_generation=current['generation'])
            process=subprocess.Popen([sys.executable,str(CONTROL),'claim'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            process.stdin.write(json.dumps(payload));process.stdin.close();process.stdin=None
            children.append(process)
        for process in children:process.communicate(timeout=10)
        self.assertEqual(sorted(process.returncode for process in children),[0,1])

    def test_other_configuration_requires_explicit_stop(self):
        self.call('start')
        self.cfg=replace(self.cfg,model=replace(self.cfg.model,name='different'))
        self.assertIn('Stop the existing model',self.call('start',ok=False))

    def test_import_deduplicates_and_keeps_local_credentials_and_ports(self):
        item=dict(self.call('status'),token=self.token)
        store=self.root/'client'
        with patch.object(resources,'STORE',store),patch.object(shared_sessions,'ensure_daemon'):
            self.assertEqual(shared_sessions.import_session(self.cfg,item),1)
            self.assertEqual(shared_sessions.import_session(self.cfg,item),0)
        data=json.loads((store/'1/session.json').read_text())
        self.assertEqual(data['remote_port'],1234)
        self.assertNotEqual(data['config']['gateway']['local_port'],data['config']['server']['port'])
        self.assertEqual(data['config']['codex'],asdict(self.cfg.codex))

class StartupTests(unittest.TestCase):
    def test_terminal_does_not_time_out_in_scheduler_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            from unittest.mock import MagicMock
            cfg=AppConfig();cfg=replace(cfg,gateway=replace(cfg.gateway,startup_timeout_seconds=1))
            spec={'loading':{'config':asdict(cfg),'model':{'alias':'test'},'reuse':True,'mtp':'off'}}
            response=MagicMock();response.__enter__.return_value.status=200
            with patch.object(loading,'urlopen',side_effect=[OSError(),response]),patch.object(loading.time,'sleep'),patch.object(loading.time,'monotonic',side_effect=[0,10,10]),patch.object(resources,'identity',return_value='live'),patch.object(resources,'rpc',return_value={'allocation':{'active':True,'slurm_state':'PENDING'}}):
                self.assertFalse(loading.prepare(Path(tmp),spec))

    def test_transient_status_connection_failure_is_retried(self):
        backend=resources.ResourceBackend(AppConfig(),'token',1)
        state={'active':True,'host':'node','model_state':'LOADED'}
        with patch.object(backend,'serverctl',side_effect=[state,RuntimeError('SSH broken pipe'),state]),patch.object(backend,'ensure_tunnel'),patch.object(backend,'http_ready',return_value=True),patch.object(backend._closing,'wait') as wait:
            backend.ensure_ready('test')
        wait.assert_called_once_with(2)

    def test_attached_provider_never_tries_to_restart_the_model(self):
        backend=resources.ResourceBackend(AppConfig(),'token',1);backend.attachment_generation='generation'
        with patch.object(resources,'remote',return_value={}) as remote:
            backend.serverctl('ensure','test')
        self.assertEqual(remote.call_args.args[3],'attach')
        self.assertEqual(remote.call_args.kwargs,{'expected_generation':'generation'})

    def test_backend_accepts_remote_ready_state(self):
        backend=resources.ResourceBackend(AppConfig(),'token',1)
        state={'active':True,'host':'node','model_state':'READY'}
        with patch.object(backend,'serverctl',return_value=state),patch.object(backend,'ensure_tunnel') as tunnel,patch.object(backend,'http_ready',return_value=True):
            backend.ensure_ready('test')
        tunnel.assert_called_once_with('node')

class BrowserOwnershipTests(unittest.TestCase):
    def test_foreign_replacement_requires_acknowledgement(self):
        from llm_away import webapp
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'session.json').write_text('{}')
            with patch.object(webapp,'session_path',return_value=path),patch.object(shared_sessions,'ensure_daemon'),patch.object(shared_sessions,'current',return_value={'owner':{'id':'other','label':'machine two'},'model_state':'LOADED'}),patch.object(shared_sessions,'client_identity',return_value='me'),patch.object(shared_sessions,'claim') as claim:
                with self.assertRaisesRegex(ValueError,'another machine'):webapp.load_browser_model(1,{})
                claim.assert_not_called()

    def test_attach_shares_model_without_transferring_ownership(self):
        from llm_away import webapp
        shared=asdict(AppConfig())
        allocation={'owner':{'id':'other'},'model_state':'LOADED','generation':'generation','session':shared}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'session.json').write_text('{}')
            with patch.object(webapp,'session_path',return_value=path),patch.object(shared_sessions,'ensure_daemon'),patch.object(shared_sessions,'current',return_value=allocation),patch.object(shared_sessions,'client_identity',return_value='me'),patch.object(shared_sessions,'attach',return_value=allocation) as attach,patch.object(shared_sessions,'claim') as claim,patch.object(resources,'rpc') as rpc,patch.object(resources,'run_agent') as run:
                webapp.load_browser_model(1,{'attach_existing':True,'expected_owner':'other','expected_generation':'generation'})
                attach.assert_called_once_with(path,'generation')
                claim.assert_not_called()
                rpc.assert_called_once_with(path,'adopt-config',expected_generation='generation')
                self.assertEqual(run.call_args.args[0].model,shared['llamacpp']['model_name'] or shared['model']['name'])

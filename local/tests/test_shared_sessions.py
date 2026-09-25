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
        for action in ('start','stop','release','terminal-start','prompt'):
            self.assertIn('acknowledge',self.call(action,client='one',ok=False))
        self.call('stop',client='two')
        self.assertNotEqual((self.state/'desired').read_text(),generation)

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

    def test_attach_transfers_ownership_without_stopping_model(self):
        from llm_away import webapp
        shared=asdict(AppConfig())
        allocation={'owner':{'id':'other'},'model_state':'LOADED','generation':'generation','session':shared}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'session.json').write_text('{}')
            with patch.object(webapp,'session_path',return_value=path),patch.object(shared_sessions,'ensure_daemon'),patch.object(shared_sessions,'current',return_value=allocation),patch.object(shared_sessions,'client_identity',return_value='me'),patch.object(shared_sessions,'claim',return_value=allocation) as claim,patch.object(resources,'rpc') as rpc,patch.object(resources,'run_agent') as run:
                webapp.load_browser_model(1,{'attach_existing':True,'expected_owner':'other','expected_generation':'generation'})
                claim.assert_called_once_with(path,'other','generation')
                rpc.assert_called_once_with(path,'adopt-config')
                self.assertEqual(run.call_args.args[0].model,shared['llamacpp']['model_name'] or shared['model']['name'])

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_away import webapp


class WebAppTests(unittest.TestCase):
    def test_cleanup_requires_preview_and_rejects_changed_paths(self):
        with self.assertRaisesRegex(ValueError,'preview expired'):
            webapp.cleanup_released('unknown')
        with patch.object(webapp,'cleanup_items',return_value=[]),patch.object(webapp.cleanup,'preview',return_value={'paths':['/tmp/example'],'remote_paths':{}}):
            plan=webapp.cleanup_preview()
        with patch.object(webapp.cleanup,'preview',return_value={'paths':['/tmp/example','/tmp/new'],'remote_paths':{}}),patch.object(webapp.cleanup,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'changed'):
                webapp.cleanup_released(plan['token'])
            execute.assert_not_called()

    def test_project_logo_is_used_for_the_favicon(self):
        self.assertTrue(webapp.LOGO_PATH.is_file())
        self.assertEqual(webapp.LOGO_PATH.name,'llm-away-logo.jpeg')
        self.assertIn('/favicon.jpeg?v=2',(webapp.WEB_ROOT/'index.html').read_text())

    def test_load_browser_model_passes_colon_separated_rag_paths(self):
        settings={'model':'model','rag':f' ~/project/src {os.pathsep}/tmp/README.md{os.pathsep} ',
                  'rag_threads':'6','rag_memory_gb':'12.5','rag_gpu':'yes',
                  'rag_compute':'remote','rag_paths_location':'local',
                  'server_options':''}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'session.json').write_text(json.dumps({'model':'','allocation':{}}))
            with patch.object(webapp,'session_path',return_value=path),patch.object(webapp.resources,'run_agent') as run:
                webapp.load_browser_model(4,settings)
        args=run.call_args.args[0]
        self.assertEqual(args.rag,['~/project/src','/tmp/README.md'])
        self.assertEqual(args.rag_threads,6)
        self.assertEqual(args.rag_memory_gb,12.5)
        self.assertTrue(args.rag_gpu)
        self.assertEqual((args.rag_compute,args.rag_paths_location),('remote','local'))

    def test_load_browser_model_replaces_loaded_model_after_acknowledgement(self):
        settings={'model':'new-model','replace_loaded':True,'server_options':''}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'session.json').write_text(json.dumps({'model':'old-model','allocation':{'model_state':'LOADED'}}))
            with patch.object(webapp,'session_path',return_value=path),patch.object(webapp.resources,'rpc') as rpc,patch.object(webapp.resources,'run_agent') as run:
                webapp.load_browser_model(4,settings)
            rpc.assert_called_once_with(path,'stop');self.assertEqual(run.call_args.args[0].model,'new-model')

    def test_load_browser_model_rejects_unacknowledged_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'session.json').write_text(json.dumps({'model':'old-model','allocation':{'model_state':'LOADED'}}))
            with patch.object(webapp,'session_path',return_value=path),patch.object(webapp.resources,'rpc') as rpc:
                with self.assertRaisesRegex(ValueError,'acknowledge replacement'):
                    webapp.load_browser_model(4,{'model':'new-model','server_options':''})
            rpc.assert_not_called()

    def test_session_rows_exclude_secrets(self):
        row={'id':7,'token':'secret','config':{'ssh':{'password':'secret'}},'host':'host','gpus':4,
             'phase':'RUNNING','model':'test','_busy':False,'_terminal':True,
             'allocation':{'job_id':'44','host':'node','model_state':'LOADED'}}
        with patch.object(webapp.monitor_ui,'snapshots',return_value=[row]):
            result=webapp.session_rows()
        text=json.dumps(result)
        self.assertNotIn('secret',text)
        self.assertEqual(result[0]['id'],7)

    def test_allocating_session_without_allocation_payload(self):
        row={'id':8,'host':'host','gpus':4,'phase':'STARTING','model':'',
             '_busy':False,'_terminal':False,'allocation':None}
        with patch.object(webapp.monitor_ui,'snapshots',return_value=[row]):
            result=webapp.session_rows()
        self.assertEqual(result[0]['node'],'')
        self.assertEqual(result[0]['model_state'],'PENDING')
        self.assertIn('Assigned node: pending',result[0]['details'])

    def test_scheduler_pending_overrides_early_model_state(self):
        row={'id':8,'host':'host','gpus':4,'phase':'STARTING','model':'test',
             '_busy':False,'_terminal':False,
             'allocation':{'slurm_state':'PENDING','host':'','model_state':'STARTING'}}
        with patch.object(webapp.monitor_ui,'snapshots',return_value=[row]):
            result=webapp.session_rows()
        self.assertEqual(result[0]['model_state'],'PENDING')

    def test_model_state_used_after_allocation(self):
        row={'id':8,'host':'host','gpus':4,'phase':'STARTING','model':'',
             '_busy':False,'_terminal':False,
             'allocation':{'slurm_state':'RUNNING','host':'node','model_state':'IDLE'}}
        with patch.object(webapp.monitor_ui,'snapshots',return_value=[row]):
            result=webapp.session_rows()
        self.assertEqual(result[0]['model_state'],'IDLE')

    def test_details_explain_scheduler_and_waiting_agent(self):
        row={'id':8,'host':'hydra','gpus':4,'phase':'PENDING','model':'test','pid':123,
             'provider_pid':456,'_busy':True,'_terminal':False,
             'allocation':{'job_id':'44','slurm_state':'PENDING','host':'','model_state':'STARTING'},
             'config':{'backend_type':'slurm_server','slurm':{'gpus':4,'nodes':1,'partition':'gpu','custom_options':['--mem=64G']},
                       'llamacpp':{'server_extra_args':['--batch-size','2048']}}}
        details=webapp.safe_details(row)
        self.assertIn('Assigned node: pending',details)
        self.assertIn('Scheduler state: PENDING',details)
        self.assertIn('Local monitor PID: 123',details)
        self.assertIn('Model flags: --batch-size 2048',details)
        self.assertIn('Slurm flags: --nodes=1 --gres=gpu:4 --partition=gpu --mem=64G',details)
        self.assertIn('Agent: queued; waiting for resources/model',details)

    def test_terminal_round_trip(self):
        item=webapp.BrowserTerminal()
        try:
            item.write('printf web-terminal-ok\\n')
            for _ in range(30):
                output=item.output(0)
                if 'web-terminal-ok' in output['data']:break
                import time;time.sleep(.1)
            self.assertIn('web-terminal-ok',output['data'])
        finally:item.close()

    def test_background_job_reports_completion(self):
        identifier=webapp.start_job('test',lambda:'finished')
        for _ in range(50):
            status=webapp.job_status(identifier)
            if status['state']!='running':break
            time.sleep(.01)
        self.assertEqual(status,{'state':'done','label':'test','message':'finished'})

    def test_background_job_reports_failure(self):
        def fail():raise ValueError('broken')
        identifier=webapp.start_job('test',fail)
        for _ in range(50):
            status=webapp.job_status(identifier)
            if status['state']!='running':break
            time.sleep(.01)
        self.assertEqual(status['state'],'error')
        self.assertEqual(status['message'],'broken')

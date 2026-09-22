import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_away import webapp


class WebAppTests(unittest.TestCase):
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

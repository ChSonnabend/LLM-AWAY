import ast
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch,MagicMock
from llm_away import resources,session_guard
from llm_away.config import AppConfig

class ExpiredAllocations(unittest.TestCase):
    def test_only_confirmed_terminal_slurm_states(self):
        cfg=replace(AppConfig(),backend_type='slurm_server')
        for state in ('TIMEOUT','CANCELLED by 42','FAILED','COMPLETED','OUT_OF_MEMORY'):
            self.assertTrue(resources.allocation_ended(cfg,{'active':False,'slurm_state':state}))
        for state in ('UNKNOWN','STOPPED','RUNNING','PENDING','SUSPENDED','PREEMPTED','REQUEUED'):
            self.assertFalse(resources.allocation_ended(cfg,{'active':False,'slurm_state':state}))
        self.assertFalse(resources.allocation_ended(cfg,{'slurm_state':'TIMEOUT'}))
        self.assertFalse(resources.allocation_ended(replace(cfg,backend_type='direct'),{'active':False,'slurm_state':'FAILED'}))

    def test_cleanup_retains_history_and_retries_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);data={'id':1,'phase':'TIMEOUT','model':'glm','provider_pid':123,'provider_identity':'born'}
            (path/'session.log').write_text('history')
            (path/'agent-selection.json').write_text('{"cli":"codex"}')
            (path/'tunnel.json').write_text('{"pid":456,"identity":"tunnel"}')
            stop=MagicMock()
            with patch('llm_away.terminals.engine',return_value={'stop':stop}),patch('llm_away.serve_registration.remove',side_effect=[OSError('retry'),None]),patch.object(session_guard,'stop_process') as kill,patch.object(resources,'remote',side_effect=AssertionError('No remote release')):
                with self.assertRaisesRegex(RuntimeError,'cleanup will retry'):resources.cleanup_ended_allocation(path,data)
                self.assertFalse(data.get('allocation_cleaned'))
                resources.cleanup_ended_allocation(path,data)
                self.assertTrue(data['allocation_cleaned'])
                self.assertEqual(data['phase'],'TIMEOUT')
                self.assertEqual((path/'session.log').read_text(),'history')
                self.assertTrue((path/'agent-selection.json').exists())
                kill.assert_any_call(123,'born');kill.assert_any_call(456,'tunnel')
                self.assertIsNone(session_guard.ensure(path,data))
                with patch.object(resources,'path_for',return_value=path):resources.release_session(1)
                self.assertEqual(json.loads((path/'session.json').read_text())['phase'],'RELEASED')

    def test_accounting_confirms_exact_job_not_steps(self):
        source=Path(__file__).resolve().parents[2]/'remote/bin/resource-control'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='status')
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);meta=state/'allocation.json';meta.write_text('{"job_id":"123"}')
            call=MagicMock(side_effect=['','123.batch|FAILED|\n123|TIMEOUT|'])
            scope={'json':json,'meta':meta,'state':state,'kind':'slurm_server','call':call}
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
            self.assertEqual(scope['status']()['slurm_state'],'TIMEOUT')
            call.side_effect=['','123.batch|FAILED|']
            self.assertEqual(scope['status']()['slurm_state'],'UNKNOWN')
            call.side_effect=[RuntimeError('SSH unavailable')]
            with self.assertRaisesRegex(RuntimeError,'SSH unavailable'):scope['status']()

    def test_daemon_cleans_ended_job_even_if_final_log_read_fails(self):
        import socket
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);cfg=replace(AppConfig(),backend_type='slurm_server')
            data={'id':1,'token':'test','config':asdict(cfg),'remote_port':123,'model':'glm'}
            (path/'session.json').write_text(json.dumps(data))
            status={'active':False,'slurm_state':'TIMEOUT','job_id':'123'}
            def remote(cfg,token,port,action,**kw):
                if action in ('reserve','status'):return status
                if action=='log':raise OSError('log no longer readable')
                self.fail('Unexpected remote action: '+action)
            sock=MagicMock();sock.accept.side_effect=socket.timeout
            sock.bind.side_effect=lambda name:Path(name).touch()
            with patch.object(resources.socket,'socket',return_value=sock),patch.object(resources,'remote',side_effect=remote),patch.object(session_guard,'register'),patch.object(resources.signal,'signal'),patch('llm_away.terminals.engine',return_value={'stop':MagicMock()}),patch('llm_away.serve_registration.remove'):
                resources.daemon(path)
            result=json.loads((path/'session.json').read_text())
            self.assertTrue(result['allocation_cleaned'])
            self.assertEqual(result['phase'],'TIMEOUT')

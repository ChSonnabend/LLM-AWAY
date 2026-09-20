import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from dataclasses import asdict

from llm_away import cleanup
from llm_away.config import AppConfig


class RemoteCleanupTests(unittest.TestCase):
    def test_remote_failure_keeps_local_record_and_success_allows_removal(self):
        for fails in (True,False):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root=Path(temp);session=root/'1';session.mkdir()
                (session/'session.json').write_text(json.dumps({'phase':'RELEASED','config':asdict(AppConfig()),'token':'a'*32,'remote_port':8080}))
                stack.enter_context(patch.object(cleanup,'STORE',root))
                stack.enter_context(patch.object(cleanup,'released_ids',return_value=[1]))
                stack.enter_context(patch.object(cleanup,'busy',return_value=False))
                stack.enter_context(patch('sys.argv',['res-clean','--session','1']))
                stack.enter_context(patch('builtins.input',side_effect=['all','y']))
                remote=stack.enter_context(patch.object(cleanup,'remote',side_effect=RuntimeError('offline') if fails else None,return_value={'cleaned':True}))
                cleanup.main()
                self.assertEqual(session.exists(),fails)
                self.assertEqual(remote.call_args.args[3],'cleanup')

    def test_remote_cleanup_handles_empty_retry_and_lock_outside_state(self):
        import subprocess,sys
        script=Path(__file__).resolve().parents[2]/'remote/bin/resource-control'
        for empty in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);token='a'*32;state=root/'resources'/token;state.mkdir(parents=True)
                if not empty:
                    (state/'allocation.json').write_text('{}')
                    (state/'release').touch()
                    (state/'worker.state').write_text('STOPPED host')
                cfg=asdict(AppConfig());cfg['remote']['resource_state_dir']=tmp;cfg['backend_type']='direct'
                payload=json.dumps({'config':cfg,'token':token,'port':1})
                for _ in range(2):
                    result=subprocess.run([sys.executable,str(script),'cleanup'],input=payload,text=True,capture_output=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertTrue(json.loads(result.stdout)['cleaned'])
                    self.assertFalse(state.exists())
                self.assertTrue((root/'resources/.locks'/ (token+'.lock')).exists())

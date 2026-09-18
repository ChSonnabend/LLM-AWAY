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

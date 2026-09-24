import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from llm_away import session_guard as guard

class GuardTests(unittest.TestCase):
    def test_legacy_german_runner_survives_english_monitor(self):
        with patch('llm_away.resources.identity',return_value='Thu Sep 24 14:30:23 2026'):
            self.assertTrue(guard.live(os.getpid(),'Do Sep 24 14:30:23 2026'))
            self.assertFalse(guard.live(os.getpid(),'Do Sep 24 14:30:22 2026'))

    def test_all_german_month_aliases(self):
        for de,en in [('Mär','Mar'),('Mrz','Mar'),('Mai','May'),('Okt','Oct'),('Dez','Dec')]:
            self.assertTrue(guard.same_birth(f'Do {de} 24 14:30:23 2026',f'Thu {en} 24 14:30:23 2026'))
        self.assertFalse(guard.same_birth('unknown','different'))

    def test_real_locale_change(self):
        values=[]
        for loc in ('de_DE.utf8','C'):
            values.append(subprocess.check_output(['ps','-p',str(os.getpid()),'-o','lstart='],text=True,env=dict(os.environ,LC_ALL=loc)).strip())
        self.assertTrue(guard.same_birth(*values))

    def test_process_key_survives_timestamp_changes(self):
        key=guard.process_key(os.getpid())
        self.assertTrue(key)
        with patch('llm_away.resources.identity',return_value='unrelated locale/timezone'):
            self.assertTrue(guard.live(os.getpid(),'old timestamp',key))
            self.assertFalse(guard.live(os.getpid(),'old timestamp',key+'0'))

    def test_missing_process_and_unavailable_query(self):
        with patch.object(guard.os,'kill',side_effect=ProcessLookupError):
            self.assertFalse(guard.live(123,'born','key'))
        with patch('llm_away.resources.identity',return_value=''):
            self.assertTrue(guard.live(os.getpid(),'born'))
        with patch.object(guard,'process_key',return_value=''):
            self.assertTrue(guard.live(os.getpid(),'born','key'))

    def test_ensure_does_not_replace_live_legacy_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'runner.json').write_text(json.dumps(dict(token='test',pid=os.getpid(),identity='Do Sep 24 14:30:23 2026',guard_pid=os.getpid(),guard_identity='Do Sep 24 14:30:23 2026')))
            with patch('llm_away.resources.identity',return_value='Thu Sep 24 14:30:23 2026'),patch.object(guard.subprocess,'Popen') as spawn:
                self.assertEqual(guard.ensure(root,dict(token='test')),os.getpid())
                spawn.assert_not_called()

    def test_stopped_and_released_sessions_are_not_adopted(self):
        for phase in ('STOPPED','RELEASED'):
            self.assertIsNone(guard.ensure(Path('/does-not-exist'),dict(phase=phase)))

    def test_watch_keeps_live_legacy_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'session.json').write_text(json.dumps(dict(token='test',phase='RUNNING')))
            (root/'runner.json').write_text(json.dumps(dict(token='test',generation='one',pid=os.getpid(),identity='Do Sep 24 14:30:23 2026')))
            with patch('llm_away.resources.identity',return_value='Thu Sep 24 14:30:23 2026'),patch.object(guard,'cleanup') as cleanup,patch.object(guard.time,'sleep',side_effect=InterruptedError):
                with self.assertRaises(InterruptedError):guard.watch(root,'one')
                cleanup.assert_not_called()

    def test_watch_cleans_up_changed_process_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'session.json').write_text(json.dumps(dict(token='test',phase='RUNNING')))
            (root/'runner.json').write_text(json.dumps(dict(token='test',generation='one',pid=os.getpid(),identity='old',process_key='different-boot:1')))
            with patch.object(guard,'cleanup') as cleanup,patch('builtins.print') as log:
                guard.watch(root,'one')
                cleanup.assert_called_once()
                self.assertEqual(json.loads(log.call_args.args[0])['event'],'runner_lost')

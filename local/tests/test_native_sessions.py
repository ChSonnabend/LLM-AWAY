import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from unittest.mock import patch, MagicMock

from llm_away import native_sessions as native, resources, monitor_ui

ROOT=Path(__file__).resolve().parents[2]

class NativeSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=Path(self.temp.name);self.path=self.store/'1'
        self.engine={'tmux':MagicMock(), 'capture':MagicMock(return_value={'status':'TERMINAL','text':'hello'}),
                     'alive':MagicMock(return_value=True),'stop':MagicMock()}
        self.engine_patch=patch('llm_away.terminals.engine',return_value=self.engine)
        self.engine_patch.start();self.addCleanup(self.engine_patch.stop)
        self.ensure=patch('llm_away.terminals.ensure').start()
        self.attach=patch('llm_away.terminals.attach').start()
        self.addCleanup(patch.stopall)

    def allocate(self,host=None):
        with redirect_stdout(io.StringIO()):native.allocate(self.store,'codex',host)
        return json.loads((self.path/'session.json').read_text())

    def test_registered_local_session_visible_and_not_stale(self):
        data=self.allocate()
        self.assertEqual(data['gpus'],0);self.assertEqual(data['model'],'')
        self.assertTrue(data['remote_cleanup_done'])
        rows=monitor_ui.snapshots(self.store)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['phase'],'RUNNING')
        self.assertTrue(rows[0]['_terminal']);self.assertFalse(rows[0].get('_stale'))
        with patch.object(resources,'STORE',self.store), redirect_stdout(io.StringIO()) as output:
            resources.monitor(Namespace(logs=None,list=True,kill=None,release=False))
        self.assertIn('native codex',output.getvalue());self.assertNotIn('OFFLINE',output.getvalue())

    def test_ssh_saved_and_run_reattaches_without_model_or_daemon(self):
        self.allocate('hydra')
        spec=json.loads((self.path/'agent-selection.json').read_text())
        self.assertEqual(spec['native_host'],'hydra');self.assertEqual(spec['location'],'local')
        with patch.object(resources,'STORE',self.store), patch.object(resources,'rpc') as rpc:
            resources.run_agent(Namespace(session=1,helper=False,detach=False))
        rpc.assert_not_called();self.assertEqual(self.attach.call_count,2)

    def test_exit_failure_stop_release_and_restart(self):
        self.allocate()
        self.engine['capture'].return_value={'status':'EXITED','text':''}
        self.assertEqual(resources.rpc(self.path,'status')['phase'],'EXITED')
        (self.path/'terminal-error.txt').write_text('missing executable')
        self.assertEqual(resources.rpc(self.path,'status')['phase'],'FAILED')
        resources.rpc(self.path,'stop')
        self.assertEqual(resources.rpc(self.path,'status')['phase'],'STOPPED')
        native.run(self.path,Namespace(detach=True))
        with patch.object(resources,'STORE',self.store):resources.release_session(1)
        self.assertEqual(monitor_ui.snapshots(self.store),[])
        with self.assertRaisesRegex(ValueError,'released'):native.run(self.path)
        self.assertEqual(self.engine['stop'].call_count,2)

    def test_failed_start_is_visible(self):
        self.ensure.side_effect=RuntimeError('tmux failed')
        with self.assertRaisesRegex(RuntimeError,'tmux failed'):self.allocate()
        self.engine['capture'].return_value={'status':'EXITED','text':''}
        self.assertEqual(monitor_ui.snapshots(self.store)[0]['phase'],'FAILED')

    def test_helper_and_model_options_rejected(self):
        self.allocate()
        for option in ('helper','model','rag','mtp'):
            with self.subTest(option=option), self.assertRaises(ValueError):
                native.run(self.path,Namespace(**{option:True}))

    def test_unique_ids_even_after_release(self):
        self.allocate();native.control(self.path,'release')
        self.allocate()
        self.assertTrue((self.store/'2'/'session.json').exists())

    def test_worker_launches_real_native_child_and_ssh_command(self):
        for host in (None,'hydra'):
            with self.subTest(host=host):
                binary=self.store/('ssh' if host else 'codex')
                binary.write_text('#!'+sys.executable+'\nimport json,sys,os\nfrom pathlib import Path\nPath("called.json").write_text(json.dumps({"args":sys.argv[1:],"env":os.environ.get("NATIVE_TEST")}))\n')
                binary.chmod(0o755)
                (self.store/'terminal.json').write_text(json.dumps(dict(native=True,local=True,cli='codex',native_host=host,cwd=str(self.store),base_url='')))
                env=dict(os.environ,PATH=str(self.store)+os.pathsep+os.environ['PATH'],NATIVE_TEST='preserved')
                subprocess.run([sys.executable,str(ROOT/'remote/bin/resource-terminal'),'worker',str(self.store)],env=env,check=True,timeout=10)
                call=json.loads((self.store/'called.json').read_text())
                self.assertEqual(call['env'],'preserved')
                self.assertEqual(call['args'],['-tt','hydra','codex'] if host else [])
                self.assertFalse((self.store/'terminal-error.txt').exists())

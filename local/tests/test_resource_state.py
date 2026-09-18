import json
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from llm_away.config import AppConfig

CONTROL = Path(__file__).resolve().parents[2]/'remote/bin/resource-control'


class ResourceStateTests(unittest.TestCase):
    def test_private_state_and_legacy_location(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'shared';root.mkdir()
            cfg=AppConfig()
            token='a'*32
            for name in ('user-one','user-two',''):
                base=Path(temp)/name if name else root/'.state'
                remote=replace(cfg.remote,workdir=str(root),resource_state_dir=str(base) if name else '')
                payload={'config':asdict(replace(cfg,remote=remote,backend_type='direct')),'token':token}
                result=subprocess.run([sys.executable,str(CONTROL),'status'],input=json.dumps(payload),text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertFalse(json.loads(result.stdout)['active'])
                self.assertTrue((base/'resources'/token/'lock').exists())
                if name:self.assertFalse((root/'.state').exists())

    def test_cleanup_requires_release_and_shutdown_and_preserves_other_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'shared';root.mkdir()
            (root/'model.gguf').write_text('keep')
            base=Path(temp)/'cache'
            cfg=AppConfig()
            cfg=replace(cfg,backend_type='direct',remote=replace(cfg.remote,workdir=str(root),resource_state_dir=str(base)))
            token='b'*32
            payload={'config':asdict(cfg),'token':token}
            def call(action):
                return subprocess.run([sys.executable,str(CONTROL),action],input=json.dumps(payload),text=True,capture_output=True)
            self.assertEqual(call('status').returncode,0)
            state=base/'resources'/token
            (state/'allocation.json').write_text('{}')
            self.assertNotEqual(call('cleanup').returncode,0)
            (state/'release').touch()
            (state/'worker.state').write_text('abc LOADING host')
            self.assertNotEqual(call('cleanup').returncode,0)
            (state/'worker.state').write_text('abc STOPPED host')
            sibling=state.parent/('c'*32);sibling.mkdir()
            self.assertEqual(call('cleanup').returncode,0)
            self.assertFalse(state.exists())
            self.assertTrue(sibling.exists())
            self.assertTrue((root/'model.gguf').exists())
            self.assertEqual(call('cleanup').returncode,0)
            self.assertFalse(state.exists())

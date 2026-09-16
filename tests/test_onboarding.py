from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from llm_away.config import load_config
from llm_away.onboarding import configure_local, SHARED_WORKDIR

class OnboardingTests(unittest.TestCase):
    def test_alias_uses_own_account_and_shared_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.toml'
            path.write_text('[ssh]\nhost="old"\nuser="previous-user"\n[model]\nname="keep-me"\n')
            with patch('llm_away.onboarding.subprocess.run', return_value=subprocess.CompletedProcess([],0,'','')) as run:
                configure_local(str(path), 'my-away')
            cfg=load_config(path)
            self.assertEqual(cfg.ssh.host,'my-away')
            self.assertEqual(cfg.ssh.user,'')
            self.assertEqual(cfg.remote.workdir,SHARED_WORKDIR)
            self.assertEqual(cfg.remote.state_dir,'$HOME/.cache/llm-away')
            self.assertEqual(cfg.remote.serverctl,SHARED_WORKDIR+'/scripts/remote/llm-away-serverctl')
            self.assertEqual(cfg.model.name,'keep-me')
            self.assertFalse(cfg.llamacpp.build_before_run)
            self.assertIn('my-away',run.call_args.args[0])
            self.assertTrue(list(Path(tmp).glob('*.backup-*')))

    def test_inaccessible_shared_installation_does_not_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.toml';path.write_text('[ssh]\nhost="old"\n')
            before=path.read_bytes()
            with patch('llm_away.onboarding.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','Permission denied')):
                with self.assertRaisesRegex(ValueError,'Cannot access'):
                    configure_local(str(path),'my-away')
            self.assertEqual(path.read_bytes(),before)

    def test_interactive_alias_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.toml';path.write_text('[ssh]\nhost="epnh"\n')
            with patch('llm_away.onboarding.sys.stdin.isatty',return_value=True), patch('builtins.input',return_value='alice-away') as prompt, patch('llm_away.onboarding.subprocess.run',return_value=subprocess.CompletedProcess([],0,'','')):
                configure_local(str(path))
            self.assertIn('~/.ssh/config',prompt.call_args.args[0])
            self.assertEqual(load_config(path).ssh.host,'alice-away')

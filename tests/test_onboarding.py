from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from llm_away.config import load_config
from llm_away.onboarding import configure_local, select_host, pattern_matches


class SelectHostTests(unittest.TestCase):
    def test_question_mark_pattern_is_recognized(self):
        self.assertTrue(pattern_matches('epn???', 'epn000'))
        self.assertTrue(pattern_matches('epn???', 'epn280'))
        self.assertFalse(pattern_matches('epn???', 'epnh'))

    def test_requested_node_under_pattern_is_accepted(self):
        with patch('llm_away.onboarding.sys.stdin.isatty', return_value=True), \
             patch('builtins.input', return_value='mi50'):
            self.assertEqual(select_host(['epnh'], '', requested='epn000', patterns=['epn???']), 'epn000')

    def test_node_matching_pattern_is_accepted_interactively(self):
        with patch('llm_away.onboarding.sys.stdin.isatty', return_value=True), \
             patch('builtins.input', side_effect=['epn000', 'mi50']):
            self.assertEqual(select_host(['epnh'], 'epnh', patterns=['epn???']), 'epn000')

    def test_unknown_alias_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not configured'):
            select_host(['epnh'], '', requested='nope123', patterns=['epn???'])

    def test_node_class_question_is_asked(self):
        with patch('llm_away.onboarding.sys.stdin.isatty', return_value=True), \
             patch('builtins.input', side_effect=['mi50']) as prompt:
            select_host(['epnh'], '', requested='epn000', patterns=['epn???'])
        self.assertIn('which node class', prompt.call_args.args[0])
        self.assertIn('epn000', prompt.call_args.args[0])

    def test_boundary_node_has_no_default(self):
        with patch('llm_away.onboarding.sys.stdin.isatty', return_value=True), \
             patch('builtins.input', side_effect=['mi100']) as prompt:
            select_host(['epnh'], '', requested='epn280', patterns=['epn???'])
        self.assertNotIn('[mi50]', prompt.call_args.args[0])
        self.assertNotIn('[mi100]', prompt.call_args.args[0])

    def test_non_away_name_under_pattern_is_accepted_without_question(self):
        with patch('llm_away.onboarding.sys.stdin.isatty', return_value=True), \
             patch('builtins.input', return_value='epnxyz') as prompt:
            self.assertEqual(select_host(['epnh'], '', requested='epnxyz', patterns=['epn???']), 'epnxyz')
        self.assertEqual(prompt.call_count, 0)


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

from contextlib import ExitStack, redirect_stdout
from pathlib import Path
import io
import tempfile
import unittest
from unittest.mock import patch

from llm_away.config import load_config
from llm_away.host_store import read_store, store_path
from llm_away.onboarding import configure_local, select_host, pattern_matches


class SelectHostTests(unittest.TestCase):
    def test_wildcards_and_stanza_exclusions(self):
        self.assertTrue(pattern_matches('epn???', 'epn137'))
        self.assertFalse(pattern_matches('epn???', 'epnh'))
        self.assertTrue(pattern_matches('gpu-* !gpu-admin', 'gpu-12'))
        self.assertFalse(pattern_matches('gpu-* !gpu-admin', 'gpu-admin'))

    def test_requested_nodes_have_no_special_class_questions(self):
        with patch('llm_away.onboarding.ask') as ask:
            for alias in ('epn000', 'epn137', 'epn279', 'epn280', 'epn349'):
                self.assertEqual(select_host([], '', requested=alias, patterns=['epn???']), alias)
            ask.assert_not_called()

    def test_pattern_selection_asks_for_concrete_hostname(self):
        with patch('llm_away.onboarding.choose_option', return_value=1), \
             patch('llm_away.onboarding.ask', return_value='epn137'):
            self.assertEqual(select_host(['epnh'], 'epnh', patterns=['epn???']), 'epn137')

    def test_unknown_or_excluded_alias_is_rejected(self):
        for alias in ('other123', 'gpu-admin'):
            with self.assertRaisesRegex(ValueError, 'not configured'):
                select_host([], '', requested=alias, patterns=['gpu-* !gpu-admin'])


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'config.toml'
        self.path.write_text('[ssh]\nhost="epnh"\nuser="previous-user"\n')
        self.probe = {'tools': dict.fromkeys(
            ('apptainer', 'docker', 'sbatch', 'kubectl', 'nvidia-smi', 'rocminfo'), False),
            'partitions': []}
        self.model = {'name': 'm', 'alias': 'm', 'path': '/p/m.gguf', 'size_bytes': 1}

    def mocks(self, stack, answers):
        stack.enter_context(patch('llm_away.onboarding.ask', side_effect=answers))
        stack.enter_context(patch('llm_away.onboarding.ssh_hosts', return_value=([], ['epn???'])))
        remote = stack.enter_context(patch('llm_away.onboarding.remote_json', return_value=self.probe))
        stack.enter_context(patch('llm_away.onboarding.discover_models', return_value=[self.model]))
        stack.enter_context(patch('llm_away.onboarding.choose_model', return_value=self.model))
        stack.enter_context(patch('llm_away.onboarding.choose_mtp', return_value='auto'))
        return remote

    def configure_direct(self):
        with ExitStack() as stack:
            self.mocks(stack, ['/scratch/x/remote', '/shared/models', '/shared/llama', 'no', 'direct', 'rocm', 'gfx906', '8', '0,1,2,3,4,5,6,7'])
            configure_local(str(self.path), alias='epn137', connection='ssh')

    def test_direct_profile_is_saved_without_changing_base_config(self):
        original = self.path.read_bytes()
        output = io.StringIO()
        with redirect_stdout(output):
            self.configure_direct()
        cfg = load_config(self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(cfg.ssh.host, 'epn137')
        self.assertEqual(cfg.ssh.user, '')
        self.assertEqual(cfg.backend_type, 'direct')
        self.assertEqual(cfg.remote.serverctl, '/scratch/x/remote/scripts/remote/llm-away-directctl')
        self.assertEqual(cfg.model.name, 'm')
        self.assertEqual(cfg.llamacpp.models_dir, '/shared/models')
        self.assertEqual(cfg.llamacpp.installation_dir, '/shared/llama')
        self.assertIn('no Slurm job will be submitted', output.getvalue())
        self.assertIn('epn137', output.getvalue())

    def test_failed_inspection_does_not_save(self):
        original = self.path.read_bytes()
        with ExitStack() as stack:
            remote = self.mocks(stack, ['/scratch/x/remote'])
            remote.side_effect = ValueError('Remote inspection failed: Permission denied')
            with self.assertRaisesRegex(ValueError, 'Permission denied'):
                configure_local(str(self.path), alias='epn137', connection='ssh')
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(store_path(self.path).exists())

    def test_known_host_reuses_settings_without_setup_questions(self):
        self.configure_direct()
        previous = read_store(self.path)['hosts']['epn137']
        with ExitStack() as stack:
            remote = self.mocks(stack, [])
            configure_local(str(self.path), alias='epn137', connection='ssh')
            remote.assert_not_called()
        self.assertEqual(read_store(self.path)['hosts']['epn137'], previous)

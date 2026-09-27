import unittest
from unittest.mock import patch
from llm_away import resources


class LocalAttachmentTests(unittest.TestCase):
    def test_remote_model_alone_does_not_attach_this_machine(self):
        self.assertFalse(resources.locally_attached({'model':'glm','allocation':{'model_state':'LOADED'}}))

    def test_provider_must_be_live_and_match_generation(self):
        row={'model':'glm','provider_pid':123,'provider_identity':'original',
             'attachment_generation':'a','allocation':{'generation':'a'}}
        with patch.object(resources,'identity',return_value='original'):
            self.assertTrue(resources.locally_attached(row))
            self.assertFalse(resources.locally_attached(dict(row,provider_exit=1)))
            self.assertFalse(resources.locally_attached(dict(row,allocation={'generation':'b'})))
        with patch.object(resources,'identity',return_value='reused PID'):
            self.assertFalse(resources.locally_attached(row))

class EndedModelTests(unittest.TestCase):
    def test_live_provider_does_not_make_unloaded_model_attached(self):
        with patch.object(resources,'identity',return_value='live'):
            for state in ('IDLE','EXITED','FAILED','STOPPED'):
                self.assertFalse(resources.locally_attached(dict(model='glm',provider_pid=1,provider_identity='live',provider_exit=None,allocation={'model_state':state})))
    def test_legacy_attachment_generation_is_checked(self):
        with patch.object(resources,'identity',return_value='live'):
            self.assertFalse(resources.locally_attached(dict(model='glm',provider_pid=1,provider_identity='live',provider_exit=None,allocation={'model_state':'LOADED','generation':'new','attachment':{'generation':'old'}})))

class RefreshRecoveryTests(unittest.TestCase):
    def test_refresh_reloads_saved_model_and_clears_stale_marker(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import Mock
        from llm_away import terminals
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'session.json').write_text(json.dumps({'config':{},'token':'t','remote_port':1,'model':'old'}))
            (root/'agent-selection.json').write_text(json.dumps({'location':'local','loading':{'model':{'alias':'saved'}}}))
            (root/'agent-ready').touch()
            stop=Mock()
            with patch.object(resources,'path_for',return_value=root),patch.object(resources,'config'),patch.object(resources,'remote',return_value={'active':True,'model_state':'IDLE'}),patch.object(terminals,'engine',return_value={'alive':lambda p:False,'stop':stop}),patch.object(resources,'run_agent') as run,patch.object(resources,'rpc') as rpc:
                resources.refresh_session(29)
                rpc.assert_called_once_with(root,'stop')
                self.assertEqual(run.call_args.args[0].model,'saved')
                self.assertTrue(run.call_args.args[0].reconfigure_agent)
                self.assertFalse((root/'agent-ready').exists())
    def test_refresh_does_not_interrupt_training(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'session.json').write_text(json.dumps({'config':{},'token':'t','remote_port':1}))
            (root/'agent-selection.json').write_text('{}')
            with patch.object(resources,'path_for',return_value=root),patch.object(resources,'config'),patch.object(resources,'remote',return_value={'active':True,'training':{'phase':'EVALUATING'}}),patch.object(resources,'run_agent') as run:
                with self.assertRaisesRegex(ValueError,'active'):resources.refresh_session(29)
                run.assert_not_called()

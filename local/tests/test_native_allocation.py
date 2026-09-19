import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from llm_away import resources

class NativeAllocationTests(unittest.TestCase):
    def args(self, **kw):
        return Namespace(**dict(dict(connection='local',host=None,mode='native',cli='codex'),**kw))

    def test_local_native_uses_registered_session(self):
        with patch.object(resources,'configure_local') as configure, patch.object(resources.shutil,'which',return_value='/bin/codex'), patch('llm_away.native_sessions.allocate') as allocate:
            resources.allocate(self.args())
            allocate.assert_called_once_with(resources.STORE,'codex',None)
            configure.assert_not_called()

    def test_remote_native_registers_host(self):
        with patch('llm_away.onboarding.ssh_hosts',return_value=(['hydra'],[])), patch('llm_away.onboarding.select_host',return_value='hydra'), patch('llm_away.native_sessions.allocate') as allocate:
            resources.allocate(self.args(connection='ssh',host='hydra',cli='claude'))
            allocate.assert_called_once_with(resources.STORE,'claude','hydra')

    def test_interactive_prompt_order(self):
        with patch.object(resources,'choose_option',side_effect=[0,0,1]) as choose, patch.object(resources.shutil,'which',return_value='/bin/codex'), patch('llm_away.native_sessions.allocate'):
            resources.allocate(self.args(connection=None,mode=None,cli=None))
            self.assertEqual([c.args[1] for c in choose.call_args_list],['Location','Session type','Native CLI'])

    def test_custom_continues_existing_setup(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(resources,'STORE',Path(temp)), patch.object(resources,'configure_local',side_effect=RuntimeError('setup reached')) as configure:
            args=self.args(mode='custom');args.config='settings';args.restart=False
            with self.assertRaisesRegex(RuntimeError,'setup reached'):resources.allocate(args)
            configure.assert_called_once_with('settings',alias=None,connection='local',restart=False,resources_only=True)

    def test_run_unloaded_session_has_initialized_reuse(self):
        import json
        from llm_away.config import AppConfig
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);data={'config':{},'model':''}
            (path/'session.json').write_text(json.dumps(data))
            args=Namespace(session=10,helper=False,detach=False,model=None,agent_location='local',cli=None,rag=None,agent_args=[],mtp=None)
            with patch.object(resources,'path_for',return_value=path), patch.object(resources,'rpc',return_value=data), patch.object(resources,'config',return_value=AppConfig()), patch.object(resources,'load_config',return_value=AppConfig()), patch.object(resources.sys.stdin,'isatty',return_value=True), patch.object(resources,'choose_option',return_value=0), patch.object(resources,'discover_models',return_value=[]), patch.object(resources,'choose_model',side_effect=RuntimeError('model selection reached')):
                with self.assertRaisesRegex(RuntimeError,'model selection reached'): resources.run_agent(args)

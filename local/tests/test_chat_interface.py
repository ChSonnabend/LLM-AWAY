import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from llm_away import webapp, shared_sessions

class ChatInterfaceTests(unittest.TestCase):
    def test_switch_reuses_model_and_preserves_rag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'session.json').write_text('{}')
            (path/'agent-selection.json').write_text(json.dumps({'cli':'codex','target_location':'local','cwd':'/project','extra_args':['--flag'],'rag_config':{'paths':['/src','/docs'],'compute':'remote','paths_location':'remote','threads':4,'memory_gb':8,'gpu':True}}))
            with patch.object(webapp,'session_path',return_value=path),patch.object(shared_sessions,'current',return_value={'model_state':'LOADED','generation':'g'}),patch('llm_away.agents.choose_cli') as choose,patch.object(webapp,'load_browser_model') as load:
                webapp.switch_chat_interface(1,'opencode')
                choose.assert_called_once_with('opencode')
                settings=load.call_args.args[1]
                self.assertTrue(settings['attach_existing']);self.assertEqual(settings['expected_generation'],'g')
                self.assertEqual(settings['cli'],'opencode');self.assertEqual(settings['rag'],'/src:/docs')
                self.assertEqual(settings['rag_compute'],'remote');self.assertEqual(settings['agent_workdir'],'/project')
                self.assertEqual(settings['rag_gpu'],'yes');self.assertEqual(settings['agent_args'],['--flag'])

    def test_unavailable_interface_does_not_restart_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'session.json').write_text('{}');(path/'agent-selection.json').write_text('{}')
            with patch.object(webapp,'session_path',return_value=path),patch.object(shared_sessions,'current',return_value={'model_state':'LOADED'}),patch('llm_away.agents.choose_cli',side_effect=ValueError('not available')),patch.object(webapp,'load_browser_model') as load:
                with self.assertRaisesRegex(ValueError,'not available'):webapp.switch_chat_interface(1,'opencode')
                load.assert_not_called()

    def test_invalid_cli_rejected_before_access(self):
        with self.assertRaises(ValueError):webapp.switch_chat_interface(1,'invalid')

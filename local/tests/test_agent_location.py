import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
ENGINE=runpy.run_path(str(ROOT/'remote/bin/resource-agent'))

class AgentLocationTests(unittest.TestCase):
    def test_cli_commands_enable_tools_and_resume_correct_session(self):
        with patch('shutil.which',return_value='/fake/agent'):
            with tempfile.TemporaryDirectory() as temp:
                command,env=ENGINE['agent_command']({'cli':'codex','model':'model','state':temp,'codex_catalog_content':'{"models":[]}'},'http://localhost:8765')
                self.assertTrue(any('model_catalog_json="'+str(Path(temp)/'codex-model-catalog.json')+'"' in arg for arg in command))
            for cli in ('codex','claude'):
                command,env=ENGINE['agent_command']({'cli':cli,'model':'model'},'http://localhost:8765','saved-id')
                self.assertEqual(command[0],cli)
                self.assertIn('saved-id',command)
                self.assertTrue(any('dangerously' in arg for arg in command))
            command,env=ENGINE['agent_command']({'cli':'claude','model':'model'},'http://localhost:8765')
            self.assertEqual(env['ANTHROPIC_BASE_URL'],'http://localhost:8765')

    def test_agent_uses_selected_workdir_and_stores_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binary=root/'codex';project=root/'project';project.mkdir()
            binary.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\nprompt=sys.stdin.read();Path("edited.txt").write_text(prompt)\nprint(json.dumps({"type":"thread.started","thread_id":"test-id"}))\nprint(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":"done"}}))\n')
            binary.chmod(0o755)
            with patch.dict(os.environ,{'PATH':str(root)+os.pathsep+os.environ['PATH']}):
                output=[]
                ENGINE['run_agent']({'cli':'codex','model':'m','prompt':'write this'},'http://localhost:1',project,root/'history.json',output.append)
            self.assertEqual((project/'edited.txt').read_text(),'write this')
            self.assertEqual(json.loads((root/'history.json').read_text())['id'],'test-id')
            self.assertIn('done\n',output)

    def test_space_submission_routes_to_persistent_terminal(self):
        from llm_away.resources import submit_prompt
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            (path/'session.json').write_text(json.dumps({'model':'m'}))
            (path/'agent-selection.json').write_text(json.dumps({'location':'local','cli':'codex','cwd':temp}))
            with patch('llm_away.resources.path_for',return_value=path), patch('llm_away.terminals.send',return_value={'status':'TERMINAL'}) as send:
                self.assertEqual(submit_prompt(1,'hello')['status'],'TERMINAL')
                self.assertEqual(send.call_args.args[2],'hello')

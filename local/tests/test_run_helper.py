import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch, MagicMock
from llm_away import resources
from llm_away.config import AppConfig


class RunHelperTests(unittest.TestCase):
    def test_registers_without_starting_or_attaching_agent(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                path=Path(temp)
                data={'model':'test','config':{},'provider_pid':123,'provider_identity':'live'}
                (path/'session.json').write_text(json.dumps(data))
                if existing:
                    (path/'agent-selection.json').write_text(json.dumps({'location':'local'}))
                args=Namespace(session=1,helper=True,detach=False,model=None,rag=[temp],
                               agent_location='local',cli=None,agent_args=[],mtp=None)
                for name,value in [('path_for',path),('rpc',data),('config',AppConfig()),
                                   ('load_config',AppConfig()),('identity','live')]:
                    stack.enter_context(patch.object(resources,name,return_value=value))
                response=MagicMock();response.__enter__.return_value.status=200
                stack.enter_context(patch.object(resources,'urlopen',return_value=response))
                stack.enter_context(patch('llm_away.terminals.engine',return_value={'alive':lambda _:True}))
                register=stack.enter_context(patch('llm_away.serve_registration.register'))
                ensure=stack.enter_context(patch('llm_away.terminals.ensure'))
                attach=stack.enter_context(patch('llm_away.terminals.attach'))
                cli=stack.enter_context(patch.object(resources,'choose_cli'))
                resources.run_agent(args)
                register.assert_called_once_with(1,[temp])
                ensure.assert_not_called();attach.assert_not_called();cli.assert_not_called()

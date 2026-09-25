import io
import json
import os
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from llm_away.agents import choose_cli, claude_launch
from llm_away.config import AppConfig, ClaudeConfig, CodexConfig, load_config
from llm_away.server import ProviderHandler
from llm_away.protocol import normalize_anthropic_system


class AgentTests(unittest.TestCase):
    def test_availability_and_selection(self):
        for installed in ([], ['codex'], ['claude'], ['codex', 'claude']):
            with self.subTest(installed=installed), patch('llm_away.agents.shutil.which', side_effect=lambda name: name if name in installed else None), patch('llm_away.agents.sys.stdin.isatty', return_value=True), patch('llm_away.agents.choose_option', return_value=1) as menu:
                if not installed:
                    with self.assertRaises(ValueError): choose_cli()
                else:
                    self.assertEqual(choose_cli(), installed[-1])
                self.assertEqual(menu.call_count, int(len(installed) == 2))

    def test_explicit_and_noninteractive_selection(self):
        with patch('llm_away.agents.shutil.which', return_value='/bin/cli'), patch('llm_away.agents.sys.stdin.isatty', return_value=False), patch('llm_away.agents.choose_option') as menu:
            self.assertEqual(choose_cli('claude'), 'claude')
            with self.assertRaisesRegex(ValueError, 'Multiple CLIs'): choose_cli()
            menu.assert_not_called()
        with patch('llm_away.agents.shutil.which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'not available'): choose_cli('claude')

    def test_claude_launch_instructions_rag_and_isolated_environment(self):
        cfg=replace(AppConfig(), codex=CodexConfig(instructions='Be concise.'))
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY':'private', 'CLAUDE_CODE_USE_BEDROCK':'1'}):
            command,env=claude_launch(cfg,'http://127.0.0.1:8765','local-model',['search','--stdio'])
            self.assertEqual(os.environ['ANTHROPIC_API_KEY'],'private')
        self.assertNotIn('ANTHROPIC_API_KEY',env)
        self.assertNotIn('CLAUDE_CODE_USE_BEDROCK',env)
        self.assertEqual(env['ANTHROPIC_DEFAULT_HAIKU_MODEL'],'local-model')
        self.assertEqual(command[:3],['claude','--model','local-model'])
        self.assertIn('Be concise.',command)
        self.assertEqual(json.loads(command[-1])['mcpServers']['project_search']['args'],['--stdio'])
        command,_=claude_launch(replace(cfg,claude=ClaudeConfig(instructions='Claude instructions')), 'http://localhost', 'm')
        self.assertIn('Claude instructions',command)
        self.assertEqual(load_config('config/model.toml').agent.cli,'auto')

    def test_system_blocks_and_late_instructions_merge_without_mutation(self):
        payload = {"system": [{"type": "text", "text": "Base", "cache_control": {"type": "ephemeral"}},
                              {"type": "text", "text": "Extra"}],
                   "messages": [{"role": "user", "content": "Hello"},
                                {"role": "developer", "content": "Update"},
                                {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {"command": "pwd"}}]},
                                {"role": "system", "content": [{"type": "text", "text": "Last"}]},
                                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}]}
        original = json.dumps(payload)
        result = normalize_anthropic_system(payload)
        self.assertEqual(result["system"], "Base\n\nExtra\n\nUpdate\n\nLast")
        self.assertEqual(result["messages"], [payload["messages"][i] for i in (0, 2, 4)])
        self.assertEqual(json.dumps(payload), original)
        self.assertEqual(normalize_anthropic_system(result), result)

    def test_native_anthropic_forwarding_and_upstream_errors(self):
        for route in ('/v1/messages?beta=true','/v1/messages/count_tokens'):
            for status in (200,404):
                with self.subTest(route=route,status=status):
                    handler=object.__new__(ProviderHandler)
                    handler.config=AppConfig()
                    handler.backend=Mock()
                    handler.backend.api_key=''
                    handler.backend.auth_headers.return_value={}
                    handler.backend.local_url.side_effect=lambda path:'http://localhost:8080'+path
                    handler.path=route
                    handler.headers={'anthropic-version':'2023-06-01','Authorization':'secret'}
                    handler.wfile=io.BytesIO()
                    handler.send_response=Mock();handler.send_header=Mock();handler.end_headers=Mock()
                    body=b'event: message_stop\ndata: {}\n\n' if status==200 else b'{"error":"not supported"}'
                    upstream=HTTPError('http://localhost',status,'test',{'Content-Type':'text/event-stream' if status==200 else 'application/json'},io.BytesIO(body))
                    payload={'system':[{'type':'text','text':'Base'},{'type':'text','text':'Extra'}],'model':'local-model','messages':[{'role':'user','content':[{'type':'tool_result','tool_use_id':'tool-1','content':'ok'}]}]}
                    handler.read_json=lambda:payload
                    with patch('llm_away.server.urlopen',side_effect=upstream if status==404 else None,return_value=upstream) as opening:
                        handler.do_POST()
                    handler.send_response.assert_called_once_with(status)
                    self.assertEqual(handler.wfile.getvalue(),body)
                    request=opening.call_args.args[0]
                    self.assertEqual(request.full_url,'http://localhost:8080'+route)
                    self.assertEqual(json.loads(request.data),{**payload, "system": "Base\n\nExtra"})
                    self.assertNotIn('Authorization',request.headers)
                    self.assertTrue(handler.close_connection)

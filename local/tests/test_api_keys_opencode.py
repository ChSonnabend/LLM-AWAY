import io
import json
import os
from pathlib import Path
import runpy
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch, Mock
from llm_away.server import ProviderHandler
from llm_away.config import AppConfig
from llm_away.security import write_key
from llm_away.agents import choose_cli

ROOT=Path(__file__).resolve().parents[2]
AGENT=runpy.run_path(str(ROOT/'remote/bin/resource-agent'))

class AuthenticationTests(unittest.TestCase):
    def test_http_authentication_blocks_inference_before_parsing(self):
        class Handler(ProviderHandler):
            backend=SimpleNamespace(api_key='test-secret')
            config=AppConfig()
            def _do_POST(self):self.write_json({'ok':True})
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            for headers in ({},{'Authorization':'Bearer wrong'}):
                with self.assertRaises(HTTPError) as error:urlopen(Request(base+'/v1/chat/completions',data=b'invalid',headers=headers))
                self.assertEqual(error.exception.code,401);error.exception.close()
            with urlopen(Request(base+'/v1/chat/completions',data=b'{}',headers={'Authorization':'Bearer test-secret'})) as result:
                self.assertEqual(json.load(result),{'ok':True})
            with urlopen(base+'/health') as result:self.assertEqual(result.status,200)
            with self.assertRaises(HTTPError) as error:urlopen(base+'/v1/models')
            self.assertEqual(error.exception.code,401);error.exception.close()
        finally:server.shutdown();server.server_close();thread.join()

    def test_credentials_are_private_and_passed_to_all_clients_without_command_line_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'api-key';write_key(path,'test-secret')
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            for cli in ('codex','claude','opencode'):
                with patch('shutil.which',return_value='/bin/'+cli):
                    cmd,env=AGENT['agent_command']({'cli':cli,'model':'glm','state':tmp},'http://localhost:1')
                self.assertNotIn('test-secret',' '.join(cmd))
                self.assertEqual(env['LLM_AWAY_API_KEY'],'test-secret')
                if cli=='claude':self.assertEqual(env['ANTHROPIC_AUTH_TOKEN'],'test-secret')
                if cli=='codex':self.assertIn('model_providers.away_agent.env_key="LLM_AWAY_API_KEY"',cmd)

    def test_opencode_config_rag_and_selection(self):
        self.assertEqual(choose_cli('opencode',['opencode']),'opencode')
        spec={'cli':'opencode','model':'glm','context_window':128000,'rag_command':['python','rag.py'],'instructions':'Be concise'}
        with patch('shutil.which',return_value='/bin/opencode'):
            cmd,env=AGENT['agent_command'](spec,'http://localhost:1234','conversation')
        self.assertEqual(cmd,['opencode','run','--format','json','--model','away/glm','--session','conversation'])
        cfg=json.loads(env['OPENCODE_CONFIG_CONTENT'])
        self.assertEqual(cfg['enabled_providers'],['away'])
        self.assertEqual(cfg['provider']['away']['options']['baseURL'],'http://localhost:1234/v1')
        self.assertEqual(cfg['mcp']['project_search']['command'],['python','rag.py'])
        self.assertEqual(cfg['share'],'disabled')

    def test_chat_tool_calls_and_stream_are_forwarded_without_conversion(self):
        handler=object.__new__(ProviderHandler)
        handler.backend=SimpleNamespace(native_tools=True,ensure_ready=lambda model:None,local_url=lambda path:'http://localhost'+path,auth_headers=lambda:{'Authorization':'Bearer upstream'})
        handler.config=AppConfig();handler.path='/v1/chat/completions';handler.headers={}
        handler.wfile=io.BytesIO();handler.send_response=Mock();handler.send_header=Mock();handler.end_headers=Mock()
        payload={'model':'glm','stream':True,'tools':[{'type':'function','function':{'name':'read'}}], 'messages':[{'role':'tool','tool_call_id':'call1','content':'result'}]}
        body=b'data: {"choices":[{"delta":{"tool_calls":[{"id":"call2"}]}}]}\n\ndata: [DONE]\n\n'
        response=HTTPError('url',200,'OK',{'Content-Type':'text/event-stream'},io.BytesIO(body))
        with patch('llm_away.server.urlopen',return_value=response) as opening:handler.handle_chat(payload)
        request=opening.call_args.args[0]
        self.assertEqual(json.loads(request.data),payload)
        self.assertEqual(request.headers['Authorization'],'Bearer upstream')
        self.assertEqual(handler.wfile.getvalue(),body)

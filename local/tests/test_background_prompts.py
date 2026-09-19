import json
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from llm_away.config import AppConfig
from llm_away.monitor_ui import show
ROOT=Path(__file__).resolve().parents[2]

class BackgroundPromptTests(unittest.TestCase):
    def test_monitor_returns_selected_session(self):
        with patch('llm_away.monitor_ui._terminal_screen',return_value=5):
            self.assertEqual(show(Path('/unused'),None,None),5)

    def test_prompt_survives_submitter_and_retains_history(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            cli=base/'codex';cli.write_text('#!/usr/bin/env python3\nimport json,sys\nsys.stdin.read()\nprint(json.dumps({"type":"thread.started","thread_id":"test-id"}))\nprint(json.dumps({"type":"item.completed","item":{"text":"Answer"}}))\n');cli.chmod(0o755)
            import os
            env=dict(os.environ,PATH=str(base)+os.pathsep+os.environ['PATH'])
            token='a'*32;state=base/'resources'/token;state.mkdir(parents=True)
            (state/'allocation.json').write_text(json.dumps({'job_id':'fixture','created':time.time()}))
            (state/'heartbeat').touch();(state/'worker.state').write_text('generation LOADING host');(state/'desired').write_text('generation')
            runner=subprocess.Popen([sys.executable,str(ROOT/'remote/bin/resource-prompts'),str(state)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,start_new_session=True,env=env)
            try:
                for _ in range(100):
                    if (state/'prompt-worker.ready').exists():break
                    time.sleep(.02)
                cfg=AppConfig();cfg=replace(cfg,backend_type='direct',remote=replace(cfg.remote,workdir=str(ROOT/'remote'),resource_state_dir=str(base)))
                for i in range(2):
                    payload={'config':asdict(cfg),'token':token,'port':1,'cli':'codex','prompt':'Question '+str(i)}
                    submitted=subprocess.run([sys.executable,str(ROOT/'remote/bin/resource-control'),'prompt'],input=json.dumps(payload),capture_output=True,text=True)
                    self.assertEqual(submitted.returncode,0,submitted.stderr)
                    result=state/'prompts'/(json.loads(submitted.stdout)['id']+'.result')
                    for _ in range(100):
                        if result.exists() and json.loads(result.read_text())['status'] in ('DONE','FAILED'):break
                        time.sleep(.03)
                    self.assertEqual(json.loads(result.read_text())['text'],'Answer\n')
                    self.assertEqual(json.loads(result.read_text())['status'],'DONE')
                self.assertEqual(json.loads((state/'remote-agent-history.json').read_text())['id'],'test-id')
            finally:runner.terminate();runner.communicate(timeout=5)

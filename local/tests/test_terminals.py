import json
import os
from pathlib import Path
import runpy
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
E=runpy.run_path(str(ROOT/'remote/bin/resource-terminal'))

class TerminalTests(unittest.TestCase):
    def test_detached_agent_accepts_space_prompt_and_remains_attachable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cli=root/'codex'
            cli.write_text('#!/usr/bin/env python3\nimport sys\nprint("READY",flush=True)\nfor line in sys.stdin: print("RECEIVED:"+line.strip(),flush=True)\n');cli.chmod(0o755)
            import uuid
            spec={'token':uuid.uuid4().hex,'cli':'codex','model':'test','cwd':str(root),'base_url':'http://localhost:1'}
            with patch.dict(os.environ,{'PATH':str(root)+os.pathsep+os.environ['PATH']}):
                E['start'](root,spec)
                try:
                    for _ in range(100):
                        if 'READY' in E['capture'](root)['text']:break
                        time.sleep(.05)
                    E['send'](root,'hello from Space')
                    for _ in range(100):
                        output=E['capture'](root)['text']
                        if 'RECEIVED:hello from Space' in output:break
                        time.sleep(.05)
                    self.assertIn('RECEIVED:hello from Space',output)
                    self.assertTrue(E['alive'](root))
                    # Starting again attaches to the same existing conversation.
                    E['start'](root,spec)
                    self.assertIn('RECEIVED:hello from Space',E['capture'](root)['text'])
                finally:E['stop'](root)
                self.assertFalse(E['alive'](root))

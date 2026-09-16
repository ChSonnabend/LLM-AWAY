import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[1]

class ShutdownTests(unittest.TestCase):
    def test_repeated_terminal_interrupts_during_submission_and_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ssh = root / 'ssh'
            ssh.write_text('''#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
root = Path(os.environ['SHUTDOWN_TEST_DIR'])
payload = json.load(sys.stdin)
command = sys.argv[-1]
if ' ensure ' in command:
    (root / 'submitting').touch()
    time.sleep(0.6)
    (root / 'allocated').touch()
    print(json.dumps({'active': True, 'job_id': '123'}))
elif ' cancel ' in command:
    (root / 'canceling').touch()
    time.sleep(0.6)
    (root / 'canceled').touch()
    print(json.dumps({'active': False}))
else:
    print(json.dumps({'active': (root / 'allocated').exists(), 'job_id': '123'}))
''')
            ssh.chmod(0o755)
            script = '''from llm_epn.config import AppConfig, ServerConfig
from llm_epn.backends import SlurmServerBackend
from llm_epn.server import serve
cfg=AppConfig(server=ServerConfig(port=0))
serve(cfg, SlurmServerBackend(cfg), warm=True)
'''
            env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                       PYTHONPATH=str(REPO/'src'), SHUTDOWN_TEST_DIR=str(root))
            process = subprocess.Popen([sys.executable, '-c', script], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       start_new_session=True)
            def wait_for(name):
                deadline = time.monotonic()+8
                while not (root/name).exists():
                    if time.monotonic()>deadline or process.poll() is not None:
                        self.fail('Did not reach '+name)
                    time.sleep(0.01)
            try:
                wait_for('submitting')
                for _ in range(5):
                    os.killpg(process.pid, signal.SIGINT)
                    time.sleep(0.03)
                wait_for('canceling')
                for _ in range(5):
                    os.killpg(process.pid, signal.SIGINT)
                    time.sleep(0.03)
                _, error = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 0, error)
                self.assertTrue((root/'canceled').exists(), error)
                self.assertEqual(error.count('canceling Slurm job'), 1, error)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()

    def test_symlink_launcher_with_spaces_and_no_gnu_readlink(self):
        with tempfile.TemporaryDirectory(prefix='epn portable ') as tmp:
            root=Path(tmp)
            link=root/'llm epn'
            link.symlink_to(REPO/'bin/llm-epn')
            readlink=root/'readlink'
            readlink.write_text('#!/bin/sh\nexit 99\n')
            readlink.chmod(0o755)
            result=subprocess.run(['bash', str(link), '--help'], text=True, capture_output=True,
                                  env=dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH']))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from llm_away.backends import SlurmServerBackend
from llm_away.config import AppConfig, GatewayConfig

REPO = Path(__file__).resolve().parents[1]

class StartupLogTests(unittest.TestCase):
    def test_new_pending_job_does_not_replay_legacy_log_or_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands = root / 'commands'
            commands.mkdir()
            for name, body in {'squeue': 'printf "%s|%s\\n" "$PHASE" "$REASON"',
                               'sbatch': 'cp "$2" "$CAPTURE"; echo "$JOB_ID"'}.items():
                path = commands / name
                path.write_text('#!/usr/bin/env bash\n' + body + '\n')
                path.chmod(0o755)
            state = root / 'state'
            state.mkdir()
            legacy = state / 'preset.log'
            legacy.write_text('OLD JINJA ERROR\n')
            env = dict(os.environ, PATH=str(commands) + ':' + os.environ['PATH'], PHASE='PENDING',
                       REASON='Resources', JOB_ID='123', CAPTURE=str(root / 'submitted.sh'))
            payload = {'model': 'preset', 'state_dir': str(state), 'remote_workdir': str(root),
                       'server_port': 8080, 'llamacpp': {}}
            def call(command):
                result = subprocess.run(['bash', str(REPO / 'scripts/remote/llm-away-serverctl'), command, '--json'],
                                        input=json.dumps(payload), text=True, capture_output=True, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)
            first = call('ensure')
            self.assertEqual(first['slurm_state'], 'PENDING')
            self.assertEqual(first['reason'], 'Resources')
            self.assertEqual(first['log_path'], str(state / 'preset-123.log'))
            self.assertIn('preset-%j.log', (root / 'submitted.sh').read_text())
            self.assertEqual(call('log')['data'], '')
            (state / 'preset.host').write_text('stale-host')
            self.assertNotIn('host', call('status'))
            fresh = Path(first['log_path'])
            fresh.write_text('NEW JOB\n')
            self.assertEqual(call('log')['data'], 'NEW JOB\n')
            env['PHASE'] = ''
            env['JOB_ID'] = '124'
            second = call('ensure')
            self.assertNotEqual(first['log_path'], second['log_path'])
            self.assertEqual(call('log')['data'], '')
            self.assertEqual(legacy.read_text(), 'OLD JINJA ERROR\n')

    def test_gateway_reports_pending_reason_and_streams_only_when_running(self):
        class Backend(SlurmServerBackend):
            def __init__(self):
                super().__init__(AppConfig(gateway=GatewayConfig(local_port=9999, poll_interval_seconds=0, cancel_on_exit=False)))
                self.states = iter([{'active': False},
                                    {'active': True, 'job_id': '123'},
                                    {'active': True, 'job_id': '123', 'slurm_state': 'PENDING', 'reason': 'Resources'},
                                    {'active': True, 'job_id': '123', 'slurm_state': 'RUNNING', 'host': 'node'}])
                self.streamed = 0
            def serverctl(self, *args, **kwargs):
                return next(self.states)
            def stream_log(self, model):
                self.streamed += 1
            def ensure_tunnel(self, host):
                pass
            def http_ready(self, model):
                return True
        backend = Backend()
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            backend.ensure_ready('preset')
        self.assertIn('PENDING (Resources)', output.getvalue())
        self.assertEqual(backend.streamed, 1)

if __name__ == '__main__':
    unittest.main()

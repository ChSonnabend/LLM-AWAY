import unittest

import json

from llm_epn.backends import InferenceRequest, SlurmServerBackend, SlurmSshBackend
from llm_epn.config import AppConfig


class BackendTests(unittest.TestCase):
    def test_build_command_targets_ssh_host(self):
        backend = SlurmSshBackend(AppConfig())
        cmd, stdin = backend.build_command(InferenceRequest(prompt="Hello", model="epn-llamacpp"))
        self.assertEqual(cmd[:3], ["ssh", "-o", "ConnectTimeout=60"])
        self.assertIn("epnh", cmd)
        self.assertIn("bash -lc", cmd[-1])
        self.assertIn("$HOME/.local/bin/llm-epn-slurm-run --json", cmd[-1])
        self.assertIn('"prompt": "Hello"', stdin)

    def test_server_backend_completion_payload_can_bound_warmup_tokens(self):
        backend = SlurmServerBackend(AppConfig())
        payload = json.loads(backend.completion_payload("Reply with OK.", max_tokens=1))

        self.assertEqual(payload["messages"][0]["content"], "Reply with OK.")
        self.assertEqual(payload["max_tokens"], 1)


if __name__ == "__main__":
    unittest.main()

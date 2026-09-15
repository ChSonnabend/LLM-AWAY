import unittest

from llm_epn.backends import InferenceRequest, SlurmSshBackend
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


if __name__ == "__main__":
    unittest.main()

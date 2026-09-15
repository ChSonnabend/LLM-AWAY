from __future__ import annotations

import unittest

import json
from pathlib import Path

from llm_epn.backends import InferenceRequest, SlurmServerBackend, SlurmSshBackend
from llm_epn.config import AppConfig, ModelConfig


class RecordingSlurmServerBackend(SlurmServerBackend):
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.ready_models = []

    def ensure_ready(self, model: str) -> None:
        self.ready_models.append(model)

    def completion(self, prompt: str, max_tokens: int | None = None) -> str:
        return "OK"


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

    def test_server_backend_readiness_uses_configured_model(self):
        config = AppConfig(model=ModelConfig(name="qwen3-coder-next-f16-1m"))
        backend = RecordingSlurmServerBackend(config)

        self.assertEqual(backend.infer(InferenceRequest(prompt="Hi", model="client-alias")), "OK")
        self.assertEqual(backend.ready_models, ["qwen3-coder-next-f16-1m"])

    def test_remote_serverctl_locks_ensure_submission(self):
        script = Path("scripts/remote/llm-epn-serverctl").read_text(encoding="utf-8")

        self.assertIn("import fcntl", script)
        self.assertIn("lock_path = os.path.join(state_dir, model_name + \".lock\")", script)
        self.assertIn("fcntl.flock(lock, fcntl.LOCK_EX)", script)


if __name__ == "__main__":
    unittest.main()

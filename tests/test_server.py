import os
import signal
import subprocess
import sys
import textwrap
import unittest

from llm_epn.config import AppConfig, GatewayConfig
from llm_epn.server import ProviderHandler


class CompactPromptTests(unittest.TestCase):
    def test_compacted_prompt_stays_within_limit(self):
        class Handler(ProviderHandler):
            pass

        Handler.config = AppConfig(
            gateway=GatewayConfig(max_prompt_chars=120, prompt_keep_head_chars=20, prompt_keep_tail_chars=80)
        )

        prompt = Handler.compact_prompt(Handler, "a" * 200)

        self.assertLessEqual(len(prompt), 120)
        self.assertIn("compacted", prompt)


class ServerTests(unittest.TestCase):
    def test_server_exits_on_sigint(self):
        script = textwrap.dedent(
            """
            from llm_epn.backends import Backend
            from llm_epn.config import AppConfig, ServerConfig
            from llm_epn.server import serve

            class FakeBackend(Backend):
                def infer(self, request):
                    return "ok"

            serve(AppConfig(server=ServerConfig(port=0)), FakeBackend())
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            line = process.stdout.readline()
            self.assertIn("llm-epn provider listening", line)
            process.send_signal(signal.SIGINT)
            _, stderr = process.communicate(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)

        self.assertEqual(process.returncode, 0, stderr)
        self.assertIn("llm-epn: shutting down", stderr)


if __name__ == "__main__":
    unittest.main()

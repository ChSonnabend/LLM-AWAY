import os
import signal
import subprocess
import sys
import textwrap
import unittest


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

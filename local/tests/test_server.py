import os
import signal
import subprocess
import sys
import textwrap
import unittest

from llm_away.config import AppConfig, GatewayConfig
from llm_away.server import ProviderHandler


class CompactPromptTests(unittest.TestCase):
    def test_default_does_not_truncate_large_input(self):
        handler = object.__new__(ProviderHandler)
        handler.config = AppConfig()
        messages = [{"role": "system", "content": "x" * 25000},
                    {"role": "user", "content": "Which model are you?"}]
        self.assertEqual(handler.compact_messages(messages), messages)
        self.assertEqual(handler.compact_prompt("x" * 25000), "x" * 25000)

    def test_compaction_preserves_latest_question_and_roles(self):
        handler = object.__new__(ProviderHandler)
        handler.config = AppConfig(gateway=GatewayConfig(max_prompt_chars=200, prompt_keep_head_chars=4000))
        messages = [{"role": "system", "content": "x" * 500},
                    {"role": "user", "content": "Which model are you?"}]
        result = handler.compact_messages(messages)
        self.assertEqual(result[-1], messages[-1])
        self.assertEqual(result[0]["role"], "system")
        self.assertLessEqual(sum(len(m["content"]) for m in result), 200)

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
    def test_query_log_records_input_and_timing(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        handler=object.__new__(ProviderHandler)
        handler.path='/v1/responses';handler.config=AppConfig()
        handler.read_json=lambda:{'model':'test','input':'hello\nworld'}
        handler.handle_responses=lambda payload:None
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'LLM_SESSION_DIR':tmp}):
            handler._do_POST()
            rows=[json.loads(line.split(' | ',1)[1]) for line in (Path(tmp)/'model-queries.log').read_text().splitlines()]
            self.assertEqual(rows[0]['input'],'hello\nworld')
            self.assertEqual(rows[0]['id'],rows[1]['id'])
            self.assertEqual(rows[1]['event'],'completed')
            self.assertGreaterEqual(rows[1]['elapsed_seconds'],0)

    def test_server_exits_on_sigint(self):
        script = textwrap.dedent(
            """
            from llm_away.backends import Backend
            from llm_away.config import AppConfig, ServerConfig
            from llm_away.server import serve

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
            self.assertIn("llm-away provider listening", line)
            process.send_signal(signal.SIGINT)
            _, stderr = process.communicate(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)

        self.assertEqual(process.returncode, 0, stderr)
        self.assertIn("llm-away: shutting down", stderr)


if __name__ == "__main__":
    unittest.main()

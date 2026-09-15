import os
from pathlib import Path
import tempfile
import unittest

from llm_epn.cli import install_codex_config, remove_toml_table


class CliTests(unittest.TestCase):
    def test_remove_toml_table_removes_nested_provider_tables(self):
        text = """
model = "gpt-5.5"

[model_providers.epn]
name = "old"

[model_providers.epn.auth]
command = "token"

[desktop]
followUpQueueMode = "steer"
""".lstrip()

        result = remove_toml_table(text, "model_providers.epn")

        self.assertNotIn("[model_providers.epn]", result)
        self.assertNotIn("[model_providers.epn.auth]", result)
        self.assertIn("[desktop]", result)

    def test_install_codex_config_writes_default_remote_host_setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            codex_home = tmp / "codex"
            config = tmp / "epn.toml"
            config.write_text(
                """
[server]
host = "127.0.0.1"
port = 8765

[model]
name = "qwen3-coder-next-f16-1m"
""".lstrip(),
                encoding="utf-8",
            )

            previous = os.environ.get("CODEX_HOME")
            os.environ["CODEX_HOME"] = str(codex_home)
            try:
                install_codex_config(str(config))
            finally:
                if previous is None:
                    os.environ.pop("CODEX_HOME", None)
                else:
                    os.environ["CODEX_HOME"] = previous

            active_config = (codex_home / "config.toml").read_text(encoding="utf-8")
            profile = (codex_home / "epn.config.toml").read_text(encoding="utf-8")

            self.assertIn('model = "qwen3-coder-next-f16-1m"', active_config)
            self.assertIn('model_provider = "epn"', active_config)
            self.assertIn("[model_providers.epn]", active_config)
            self.assertIn('base_url = "http://127.0.0.1:8765/v1"', active_config)
            self.assertIn('model_reasoning_effort = "high"', profile)
            self.assertTrue((codex_home / "model-catalogs" / "epn.json").exists())
            self.assertTrue((codex_home / "config-epn.toml").exists())

            combined_catalog = (codex_home / "model-catalogs" / "combined-with-epn.json").read_text(
                encoding="utf-8"
            )
            self.assertIn('"truncation_policy"', combined_catalog)
            self.assertIn('"limit": 10000', combined_catalog)
            self.assertIn('"model_messages"', combined_catalog)
            self.assertIn("You are Qwen3, a coding agent.", combined_catalog)


if __name__ == "__main__":
    unittest.main()

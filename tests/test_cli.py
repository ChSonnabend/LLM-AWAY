import os
from pathlib import Path
import tempfile
import unittest

from llm_epn.cli import codex_provider_display_name, epn_model_catalog, install_codex_config, remove_toml_table


class CliTests(unittest.TestCase):
    def test_codex_provider_display_name_defaults_to_user(self):
        previous_name = os.environ.get("LLM_EPN_CODEX_PROVIDER_NAME")
        previous_user = os.environ.get("USER")
        previous_username = os.environ.get("USERNAME")
        os.environ.pop("LLM_EPN_CODEX_PROVIDER_NAME", None)
        os.environ["USER"] = "alice"
        os.environ.pop("USERNAME", None)
        try:
            self.assertEqual(codex_provider_display_name(), "alice")
        finally:
            if previous_name is None:
                os.environ.pop("LLM_EPN_CODEX_PROVIDER_NAME", None)
            else:
                os.environ["LLM_EPN_CODEX_PROVIDER_NAME"] = previous_name
            if previous_user is None:
                os.environ.pop("USER", None)
            else:
                os.environ["USER"] = previous_user
            if previous_username is None:
                os.environ.pop("USERNAME", None)
            else:
                os.environ["USERNAME"] = previous_username

    def test_epn_model_catalog_advertises_coding_agent_mode(self):
        catalog = epn_model_catalog("qwen3-coder-next-f16-1m")
        model = catalog["models"][0]

        self.assertEqual(model["tool_mode"], "code_mode_only")
        self.assertTrue(model["use_responses_lite"])
        self.assertTrue(model["include_apps_usage_instructions"])
        self.assertTrue(model["include_plugin_usage_instructions"])
        self.assertFalse(model["include_skills_usage_instructions"])
        self.assertEqual(model["truncation_policy"], {"mode": "tokens", "limit": 10000})
        self.assertIn("You are Qwen3, a coding agent.", model["model_messages"]["instructions_template"])

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
            self.assertIn('name = "', active_config)
            self.assertNotIn('name = "EPN Slurm llama.cpp"', active_config)
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

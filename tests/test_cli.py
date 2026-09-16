import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from llm_away.cli import codex_provider_display_name, remote_model_catalog, install_codex_config, main, remove_toml_table


class CliTests(unittest.TestCase):
    def test_codex_provider_display_name_defaults_to_user(self):
        previous_name = os.environ.get("LLM_REMOTE_CODEX_PROVIDER_NAME")
        previous_user = os.environ.get("USER")
        previous_username = os.environ.get("USERNAME")
        os.environ.pop("LLM_REMOTE_CODEX_PROVIDER_NAME", None)
        os.environ["USER"] = "alice"
        os.environ.pop("USERNAME", None)
        try:
            self.assertEqual(codex_provider_display_name(), "alice")
        finally:
            if previous_name is None:
                os.environ.pop("LLM_REMOTE_CODEX_PROVIDER_NAME", None)
            else:
                os.environ["LLM_REMOTE_CODEX_PROVIDER_NAME"] = previous_name
            if previous_user is None:
                os.environ.pop("USER", None)
            else:
                os.environ["USER"] = previous_user
            if previous_username is None:
                os.environ.pop("USERNAME", None)
            else:
                os.environ["USERNAME"] = previous_username

    def test_codex_provider_display_name_uses_config_before_user(self):
        previous_name = os.environ.get("LLM_REMOTE_CODEX_PROVIDER_NAME")
        previous_user = os.environ.get("USER")
        os.environ.pop("LLM_REMOTE_CODEX_PROVIDER_NAME", None)
        os.environ["USER"] = "alice"
        try:
            self.assertEqual(codex_provider_display_name("Christian Sonnabend"), "Christian Sonnabend")
        finally:
            if previous_name is None:
                os.environ.pop("LLM_REMOTE_CODEX_PROVIDER_NAME", None)
            else:
                os.environ["LLM_REMOTE_CODEX_PROVIDER_NAME"] = previous_name
            if previous_user is None:
                os.environ.pop("USER", None)
            else:
                os.environ["USER"] = previous_user

    def test_away_model_catalog_advertises_coding_agent_mode(self):
        catalog = remote_model_catalog("qwen3.8-flash-next-125b-ultralite-37g")
        model = catalog["models"][0]

        self.assertEqual(model["slug"], "away")
        self.assertEqual(model["display_name"], "AWAY")
        self.assertNotIn("tool_mode", model)
        self.assertFalse(model["use_responses_lite"])
        self.assertFalse(model["include_apps_usage_instructions"])
        self.assertFalse(model["include_plugin_usage_instructions"])
        self.assertFalse(model["include_skills_usage_instructions"])
        self.assertEqual(model["truncation_policy"], {"mode": "tokens", "limit": 2000})
        self.assertEqual(model["context_window"], 262000)
        self.assertEqual(model["max_context_window"], 262000)
        self.assertIn("You are a coding agent.", model["model_messages"]["instructions_template"])
        self.assertIn("use read-only commands and then answer", model["model_messages"]["instructions_template"])
        self.assertIn("use apply_patch instead of shell heredocs", model["model_messages"]["instructions_template"])

    def test_remove_toml_table_removes_nested_provider_tables(self):
        text = """
model = "gpt-5.5"

[model_providers.away]
name = "old"

[model_providers.away.auth]
command = "token"

[desktop]
followUpQueueMode = "steer"
""".lstrip()

        result = remove_toml_table(text, "model_providers.away")

        self.assertNotIn("[model_providers.away]", result)
        self.assertNotIn("[model_providers.away.auth]", result)
        self.assertIn("[desktop]", result)

    def test_install_codex_config_writes_default_remote_host_setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            codex_home = tmp / "codex"
            config = tmp / "away.toml"
            config.write_text(
                """
[server]
host = "127.0.0.1"
port = 8765

[model]
name = "qwen3.8-flash-next-125b-ultralite-37g"

[llamacpp]
context_size = 16384

[codex]
provider_display_name = "Christian Sonnabend"
account_email = "sonnabendch@gmail.com"
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
            profile = (codex_home / "away.config.toml").read_text(encoding="utf-8")

            self.assertIn('model = "away"', profile)
            self.assertIn('model_provider = "away"', profile)
            self.assertNotIn('model_provider =', active_config)
            self.assertNotIn('model_catalog_json =', active_config)
            self.assertIn("[model_providers.away]", active_config)
            self.assertIn('name = "Christian Sonnabend"', active_config)
            self.assertNotIn('name = "AWAY Slurm llama.cpp"', active_config)
            self.assertIn('base_url = "http://127.0.0.1:8765/v1"', active_config)
            self.assertIn('model_reasoning_effort = "low"', profile)
            self.assertTrue((codex_home / "model-catalogs" / "away.json").exists())
            self.assertFalse((codex_home / "config-away.toml").exists())

            combined_catalog = (codex_home / "model-catalogs" / "combined-with-away.json").read_text(
                encoding="utf-8"
            )
            self.assertIn('"truncation_policy"', combined_catalog)
            self.assertIn('"limit": 2000', combined_catalog)
            self.assertIn('"context_window": 16384', combined_catalog)
            self.assertIn('"model_messages"', combined_catalog)
            self.assertIn("You are a coding agent.", combined_catalog)
            self.assertIn("use read-only commands and then answer", combined_catalog)
            self.assertIn("use apply_patch instead of shell heredocs", combined_catalog)

    def test_repeated_install_preserves_global_settings(self):
        repo = Path(__file__).resolve().parents[1]
        for catalog_line in ("", 'model_catalog_json = "/custom/catalog.json"\n'):
            with self.subTest(catalog=catalog_line), tempfile.TemporaryDirectory() as tmpdir:
                home = Path(tmpdir)
                codex_home = home / ".codex"
                codex_home.mkdir()
                config_file = codex_home / "config.toml"
                original = (
                    '# User settings\nmodel = "my-openai-model"\n'
                    'model_provider = "openai"\nmodel_reasoning_effort = "medium"\n'
                    + catalog_line
                    + '\n[desktop]\nfollowUpQueueMode = "steer"\n'
                    '\n[model_providers.other]\nname = "Other"\n'
                )
                config_file.write_text(original, encoding="utf-8")
                with patch.dict(os.environ, {"HOME": str(home), "CODEX_HOME": str(codex_home)}):
                    # Exercise both the bare CLI default and the actual init script.
                    self.assertEqual(main(["install-codex-config"]), 0)
                    installed = config_file.read_text(encoding="utf-8")
                    self.assertEqual(remove_toml_table(installed, "model_providers.away"), original)
                    backups = list(codex_home.glob("config.toml.bak-*"))
                    self.assertEqual(len(backups), 1)
                    self.assertEqual(backups[0].read_text(encoding="utf-8"), original)
                    for _ in range(2):
                        subprocess.run(["bash", str(repo / "scripts/init-local.sh"), "--skip-model-selection"], check=True, capture_output=True, text=True)
                        self.assertEqual(config_file.read_text(encoding="utf-8"), installed)
                    self.assertEqual(list(codex_home.glob("config.toml.bak-*")), backups)

    def test_global_activation_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"CODEX_HOME": tmpdir}):
            self.assertEqual(main(["install-codex-config", "--activate"]), 0)
            home = Path(tmpdir)
            config = (home / "config.toml").read_text(encoding="utf-8")
            self.assertTrue(config.startswith('model = '))
            self.assertIn('model_provider = "away"', config)
            self.assertIn('model_reasoning_effort = "low"', config)
            self.assertIn('model_catalog_json = ', config)
            self.assertEqual((home / "config-away.toml").read_text(encoding="utf-8"), config)


if __name__ == "__main__":
    unittest.main()

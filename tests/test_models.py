import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from llm_away.cli import main
from llm_away.config import AppConfig, _loads_toml, load_config
from llm_away.models import choose_model, discover_models, save_model
from llm_away.remote_models import installed_models


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "away.toml"
        self.config.write_text('# Keep this comment\n[model]\nname = "old"\n'
                               '[llamacpp]\nmodel_name = "old"\ncontext_size = 12345\n'
                               '[ssh]\nhost = "example"\n', encoding="utf-8")
        self.model = {"name": "preset", "alias": "served-alias", "path": "/remote/model.gguf", "size_bytes": 42}

    def test_save_changes_only_model_names_and_is_idempotent(self):
        original = self.config.read_text()
        save_model(str(self.config), self.model)
        data = _loads_toml(self.config.read_text())
        expected = _loads_toml(original)
        expected["model"]["name"] = "served-alias"
        expected["llamacpp"]["model_name"] = "preset"
        self.assertEqual(data, expected)
        self.assertIn("# Keep this comment", self.config.read_text())
        saved = self.config.read_text()
        save_model(str(self.config), self.model)
        self.assertEqual(saved, self.config.read_text())
        backups = list(self.root.glob("away.toml.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), original)

    def test_custom_command_is_preserved_and_rejected(self):
        self.config.write_text('[llamacpp]\nserver_command = ["custom-server"]\n')
        original = self.config.read_text()
        with self.assertRaisesRegex(ValueError, "server_command"):
            save_model(str(self.config), self.model)
        self.assertEqual(self.config.read_text(), original)

    def test_prompt_selection_default_retry_and_cancel(self):
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", side_effect=["bad", "2", "1"]):
                self.assertEqual(choose_model([self.model], "old"), self.model)
            with patch("builtins.input", return_value=""):
                self.assertEqual(choose_model([self.model], "preset"), self.model)
            with patch("builtins.input", return_value="q"):
                with self.assertRaisesRegex(ValueError, "canceled"):
                    choose_model([self.model], "old")
        with patch("sys.stdin.isatty", return_value=False):
            with self.assertRaisesRegex(ValueError, "terminal"):
                choose_model([self.model], "old")
        self.assertEqual(choose_model([self.model], "old", "preset"), self.model)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            choose_model([self.model], "old", "missing")
        with self.assertRaisesRegex(ValueError, "No installed"):
            choose_model([], "old")

    def test_cli_selection_updates_profile_and_preserves_global_settings(self):
        home = self.root / "codex"
        home.mkdir()
        global_config = home / "config.toml"
        global_config.write_text('model_provider = "openai"\nmodel = "user-model"\n')
        with patch.dict(os.environ, {"CODEX_HOME": str(home)}), patch("llm_away.cli.discover_models", return_value=[self.model]):
            self.assertEqual(main(["select-model", "--config", str(self.config), "--model", "preset"]), 0)
        self.assertEqual(load_config(self.config).model.name, "served-alias")
        self.assertTrue(global_config.read_text().startswith('model_provider = "openai"\nmodel = "user-model"\n'))
        self.assertNotIn("model_catalog_json", global_config.read_text())
        self.assertIn('model = "away"', (home / "away.config.toml").read_text())
        catalog = json.loads((home / "model-catalogs/away.json").read_text())
        self.assertEqual(catalog["models"][0]["display_name"], "AWAY")

    def test_discovery_failure_does_not_write(self):
        original = self.config.read_text()
        with patch("llm_away.cli.discover_models", side_effect=ValueError("SSH unavailable")):
            self.assertEqual(main(["select-model", "--config", str(self.config), "--model", "preset"]), 2)
        self.assertEqual(self.config.read_text(), original)
        self.assertFalse(list(self.root.glob("*.backup-*")))

    def test_init_selects_before_install_and_unavailable_model_stops_setup(self):
        repo = Path(__file__).resolve().parents[1]
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        ssh = fake_bin / "ssh"
        ssh.write_text("#!/usr/bin/env python3\nimport sys\nsys.stdin.read()\nprint(" + repr(json.dumps([self.model])) + ")\n")
        ssh.chmod(0o755)
        home = self.root / "home"
        env = dict(os.environ, HOME=str(home), CODEX_HOME=str(home / ".codex"),
                   PATH=str(fake_bin) + os.pathsep + os.environ["PATH"])
        command = ["bash", str(repo / "scripts/init-local.sh"), "--config", str(self.config), "--model"]
        original = self.config.read_text()
        result = subprocess.run(command + ["unavailable"], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.config.read_text(), original)
        self.assertFalse(home.exists())
        result = subprocess.run(command + ["preset"], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(load_config(self.config).llamacpp.model_name, "preset")
        self.assertTrue((home / ".local/bin/codex-away").is_symlink())
        self.assertIn('model = "away"', (home / ".codex/away.config.toml").read_text())
        self.assertNotIn('model_provider =', (home / ".codex/config.toml").read_text())

    def test_discovery_ssh_transport_and_invalid_output(self):
        with patch("llm_away.models.subprocess.run", return_value=subprocess.CompletedProcess([], 0, json.dumps([self.model]), "")) as run:
            self.assertEqual(discover_models(AppConfig()), [self.model])
            self.assertIn("epnh", run.call_args.args[0])
            self.assertIn("def installed_models", run.call_args.kwargs["input"])
        with patch("llm_away.models.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "{}", "")):
            with self.assertRaisesRegex(ValueError, "invalid model list"):
                discover_models(AppConfig())

    def test_remote_discovery_filters_missing_and_incomplete_models(self):
        library = self.root / "scripts/lib/llamacpp-env.sh"
        library.parent.mkdir(parents=True)
        library.write_text('llamacpp_load_model_config() { MODEL_NAME="$1"; source "models/$1/model.env"; MODEL_ALIAS=${MODEL_ALIAS:-$1}; }\n'
                           'llamacpp_resolve_model() { test -f "$MODEL_GGUF" || return 1; printf "%s" "$PWD/$MODEL_GGUF"; }\n')
        for name in ("complete", "partial", "missing"):
            preset = self.root / "models" / name
            preset.mkdir(parents=True)
            (preset / "model.env").write_text(f'MODEL_GGUF="models/{name}/weights-00001-of-00002.gguf"\n')
            if name != "missing":
                (preset / "weights-00001-of-00002.gguf").write_bytes(b"test")
            if name == "complete":
                (preset / "weights-00002-of-00002.gguf").write_bytes(b"data")
        models = installed_models(str(self.root))
        self.assertEqual([model["name"] for model in models], ["complete"])
        self.assertEqual(models[0]["size_bytes"], 8)
        self.assertEqual(models[0]["alias"], "complete")


if __name__ == "__main__":
    unittest.main()

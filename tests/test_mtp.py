from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from llm_away.backends import SlurmServerBackend, SlurmSshBackend, InferenceRequest
from llm_away.cli import main
from llm_away.config import AppConfig, GatewayConfig, LlamaCppConfig, load_config
from llm_away.models import choose_mtp, mtp_label, save_model
from llm_away.remote_models import installed_models


REPO = Path(__file__).resolve().parents[1]


class MtpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = {"name": "preset", "alias": "preset", "path": "/model.gguf", "size_bytes": 10,
                      "mtp": {"configured": True, "available": True, "toggle_supported": True,
                              "draft_path": "/draft.gguf", "draft_size_bytes": 2,
                              "draft_n_max": "3", "spec_type": "draft-mtp"}}

    def test_prompt_default_preserves_enabled_preset_and_explicit_off(self):
        with patch("builtins.input", return_value="") as prompt:
            self.assertEqual(choose_mtp(self.model, interactive=True), "on")
            self.assertIn("[Y/n]", prompt.call_args.args[0])
            self.assertEqual(choose_mtp(self.model, "off", interactive=True), "off")
            self.assertIn("[y/N]", prompt.call_args.args[0])
        with patch("builtins.input", side_effect=["invalid", "n"]):
            self.assertEqual(choose_mtp(self.model, interactive=True), "off")
        with patch("builtins.input", return_value="q"):
            with self.assertRaisesRegex(ValueError, "canceled"):
                choose_mtp(self.model, interactive=True)

    def test_noninteractive_selection_never_prompts(self):
        with patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
            self.assertEqual(choose_mtp(self.model), "auto")
            self.assertEqual(choose_mtp(self.model, requested="off", interactive=True), "off")

    def test_unavailable_draft_and_old_wrapper_reject_explicit_enabling(self):
        self.model["mtp"]["available"] = False
        self.assertIn("missing/incomplete", mtp_label(self.model))
        for mode in ("auto", "on"):
            with self.assertRaises(ValueError):
                choose_mtp(self.model, requested=mode)
        self.assertEqual(choose_mtp(self.model, requested="off"), "off")
        self.model["mtp"]["toggle_supported"] = False
        with self.assertRaisesRegex(ValueError, "wrapper"):
            choose_mtp(self.model, requested="off")

    def test_invalid_or_conflicting_settings_rejected(self):
        for values in ({"mtp": True}, {"mtp": "invalid"}, {"mtp": "on", "model_path": "custom.gguf"},
                       {"mtp": "off", "server_command": ["custom"]},
                       {"mtp": "on", "server_extra_args": ["--spec-type=draft-mtp"]}):
            with self.assertRaises(ValueError):
                LlamaCppConfig(**values)
        self.assertEqual(LlamaCppConfig().mtp, "auto")

    def test_selection_persists_mtp_without_changing_other_arguments(self):
        config = self.root / "away.toml"
        config.write_text('[llamacpp]\nmodel_name = "preset"\nserver_extra_args = ["--reasoning", "off"]\n')
        save_model(str(config), self.model, mtp="off")
        parsed = load_config(config)
        self.assertEqual(parsed.llamacpp.mtp, "off")
        self.assertEqual(parsed.llamacpp.server_extra_args, ["--reasoning", "off"])

    def test_switching_to_non_mtp_model_resets_sticky_on_and_cancel_writes_nothing(self):
        config = self.root / "away.toml"
        config.write_text('[llamacpp]\nmodel_name = "previous"\nmtp = "on"\n')
        plain = {key: value for key, value in self.model.items() if key != "mtp"}
        with patch("llm_away.cli.discover_models", return_value=[plain]), patch("llm_away.cli.install_codex_config"):
            self.assertEqual(main(["select-model", "--config", str(config), "--model", "preset"]), 0)
        self.assertEqual(load_config(config).llamacpp.mtp, "auto")
        original = config.read_text()
        with patch("llm_away.cli.discover_models", return_value=[self.model]), patch("sys.stdin.isatty", return_value=True), \
             patch("builtins.input", side_effect=["1", "q"]), patch("llm_away.cli.install_codex_config") as install:
            self.assertEqual(main(["select-model", "--config", str(config)]), 2)
            install.assert_not_called()
        self.assertEqual(config.read_text(), original)

    def test_backends_send_mtp_without_duplicate_speculative_cli_flags(self):
        for mode in ("auto", "on", "off"):
            config = AppConfig(llamacpp=LlamaCppConfig(mtp=mode), gateway=GatewayConfig(local_port=9999))
            _, data = SlurmSshBackend(config).build_command(InferenceRequest("hello", "preset"))
            self.assertEqual(json.loads(data)["llamacpp"]["mtp"], mode)
            backend = SlurmServerBackend(config)
            with patch("llm_away.backends.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "{}", "")) as run:
                backend.serverctl("status", "preset")
                payload = json.loads(run.call_args.kwargs["input"])
            self.assertEqual(payload["llamacpp"]["mtp"], mode)
            self.assertEqual(payload["llamacpp"]["server_extra_args"], [])

    def test_discovery_checks_draft_shards_and_isolates_inherited_settings(self):
        library = self.root / "scripts/lib/llamacpp-env.sh"
        library.parent.mkdir(parents=True)
        library.write_text('llamacpp_apply_mtp_mode() { :; }\n'
                           'llamacpp_load_model_config() { source "models/$1/model.env"; MODEL_ALIAS="$1"; }\n'
                           'llamacpp_resolve_model() { printf "%s" "$PWD/$MODEL_GGUF"; }\n')
        for name in ("mtp", "partial", "plain", "other"):
            preset = self.root / "models" / name
            preset.mkdir(parents=True)
            (preset / "main.gguf").write_bytes(b"main")
            text = f'MODEL_GGUF=models/{name}/main.gguf\n'
            if name != "plain":
                text += f'MODEL_DRAFT_GGUF=models/{name}/draft-00001-of-00002.gguf\n'
                (preset / "draft-00001-of-00002.gguf").write_bytes(b"draft")
                if name != "partial":
                    (preset / "draft-00002-of-00002.gguf").write_bytes(b"draft")
            if name == "other":
                text += 'LLAMACPP_SPEC_TYPE=draft-eagle3\n'
            (preset / "model.env").write_text(text)
        with patch.dict(os.environ, {"MODEL_DRAFT_GGUF": "wrong.gguf", "LLAMACPP_SPEC_TYPE": "none", "LLAMACPP_MTP": "off"}):
            models = {model["name"]: model for model in installed_models(str(self.root))}
        self.assertTrue(models["mtp"]["mtp"]["available"])
        self.assertEqual(models["mtp"]["mtp"]["draft_size_bytes"], 10)
        self.assertTrue(models["partial"]["mtp"]["configured"])
        self.assertFalse(models["partial"]["mtp"]["available"])
        self.assertFalse(models["plain"]["mtp"]["configured"])
        self.assertFalse(models["other"]["mtp"]["configured"])

    def test_controller_records_mode_and_rejects_reusing_different_mode(self):
        commands = self.root / "commands"
        commands.mkdir()
        for name, body in {"squeue": "echo RUNNING\n", "sbatch": 'cp "$2" "$CAPTURE"\necho 123\n'}.items():
            path = commands / name
            path.write_text("#!/usr/bin/env bash\n" + body)
            path.chmod(0o755)
        capture = self.root / "job.sh"
        env = dict(os.environ, PATH=str(commands) + ":" + os.environ["PATH"], CAPTURE=str(capture))
        payload = {"model": "preset", "state_dir": str(self.root / "state"), "remote_workdir": str(self.root),
                   "server_port": 8080, "llamacpp": {"mtp": "off"}}
        command = ["bash", str(REPO / "scripts/remote/llm-away-serverctl"), "ensure", "--json"]
        first = subprocess.run(command, input=json.dumps(payload), env=env, text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(json.loads(first.stdout)["mtp"], "off")
        self.assertIn("export LLAMACPP_MTP=off", capture.read_text())
        subprocess.run(["bash", "-n", str(capture)], check=True)
        timestamp = capture.stat().st_mtime_ns
        payload["llamacpp"]["mtp"] = "on"
        second = subprocess.run(command, input=json.dumps(payload), env=env, text=True, capture_output=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("server-cancel", second.stderr)
        self.assertEqual(capture.stat().st_mtime_ns, timestamp)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from datetime import datetime
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from .config import AppConfig, _loads_toml, load_config
from .prompt import choose_option, interactive_available


def mtp_label(model: dict) -> str:
    mtp = model.get("mtp", {})
    if not mtp.get("configured"):
        return "MTP not configured"
    if not mtp.get("available"):
        return "MTP configured; draft files missing/incomplete"
    if mtp.get('embedded'):
        return f"MTP embedded; up to {mtp['draft_n_max']} draft tokens"
    return (f"MTP enabled by remote preset; +{mtp['draft_size_bytes'] / 1024**3:.1f} GiB draft, "
            f"up to {mtp['draft_n_max']} draft tokens")


def choose_mtp(model: dict, current: str = "auto", requested: str | None = None,
               interactive: bool | None = None) -> str:
    mtp = model.get("mtp", {})
    mode = requested if requested is not None else current
    if interactive is None:
        interactive = interactive_available()
    if interactive and requested is None and mtp.get("available") and mtp.get("toggle_supported"):
        default = mode != "off"
        if interactive_available():
            index = choose_option(["MTP on", "MTP off"], "MTP", 0 if default else 1)
            mode = "on" if index == 0 else "off"
        else:
            while True:
                answer = input("Use MTP? " + ("[Y/n]" if default else "[y/N]") + ", q to cancel: ").strip().lower()
                if answer == "q":
                    raise ValueError("Model selection canceled; settings unchanged")
                if answer in ("", "y", "yes", "n", "no"):
                    mode = "on" if (default if not answer else answer in ("y", "yes")) else "off"
                    break
                print("Choose y or n, or q to cancel.")
    if requested is not None:
        mode = requested
    if mode not in ("auto", "on", "off"):
        raise ValueError("MTP mode must be auto, on, or off")
    if mode != "auto" and not mtp.get("toggle_supported"):
        raise ValueError("Remote wrapper needs LLAMACPP_MTP support for an explicit MTP choice; use --mtp auto or update it")
    # A preset with no MTP draft can only run without MTP.
    if not mtp.get("configured"):
        mode = "auto"
    if mode == "auto" and mtp.get("configured") and not mtp.get("available"):
        raise ValueError("Remote preset enables MTP but draft files are missing/incomplete; install them or use --mtp off")
    return mode


def discover_models(config: AppConfig) -> list[dict]:
    script = Path(__file__).with_name("remote_models.py").read_text(encoding="utf-8")
    command = ["ssh", "-o", "BatchMode=yes", "-o",
               f"ConnectTimeout={config.ssh.connect_timeout_seconds}",
               config.ssh.destination, "python3 - " + shlex.quote(config.remote.workdir)]
    if config.ssh.connection == "local":
        command = [sys.executable, '-', config.remote.workdir]
    for attempt in range(max(1, config.ssh.retries)):
        result = subprocess.run(command, input=script, text=True, capture_output=True,
                                timeout=max(60, config.ssh.connect_timeout_seconds + 30))
        if result.returncode == 0:
            models = json.loads(result.stdout)
            if not isinstance(models, list) or any(
                not isinstance(model, dict)
                or not all(isinstance(model.get(key), str) and model[key] for key in ("name", "alias", "path"))
                or not isinstance(model.get("size_bytes"), int)
                for model in models
            ):
                raise ValueError("Remote returned an invalid model list")
            for model in models:
                if "mtp" in model:
                    mtp = model["mtp"]
                    if (not isinstance(mtp, dict)
                        or any(type(mtp.get(key)) is not bool for key in ("configured", "available", "toggle_supported"))
                        or not isinstance(mtp.get("draft_size_bytes"), int)
                        or any(not isinstance(mtp.get(key), str) for key in ("draft_path", "draft_n_max", "spec_type"))):
                        raise ValueError("Remote returned invalid MTP metadata")
            return models
        if result.returncode != 255 or attempt + 1 >= max(1, config.ssh.retries):
            raise ValueError("Remote model discovery failed: " + (result.stderr.strip() or result.stdout.strip()))
        time.sleep(config.ssh.retry_delay_seconds)
    raise ValueError("Remote model discovery failed")


def choose_model(models: list[dict], current: str, requested: str | None = None) -> dict:
    if not models:
        raise ValueError("No installed managed models found on the remote host")
    if requested is not None:
        for model in models:
            if model["name"] == requested:
                return model
        raise ValueError(f"Model {requested!r} is unavailable; run list-models to see installed models")
    if interactive_available():
        default_index = next((i for i, model in enumerate(models) if model["name"] == current), 0)
        labels = [f"{m['name']} — {m['size_bytes'] / 1024**3:.1f} GiB; {mtp_label(m)}" for m in models]
        return models[choose_option(labels, "model", default_index)]
    for index, model in enumerate(models, 1):
        marker = " (current)" if model["name"] == current else ""
        print(f"  {index}. {model['name']} — {model['size_bytes'] / 1024**3:.1f} GiB{marker}; {mtp_label(model)}")
    if not sys.stdin.isatty():
        raise ValueError("Model selection needs a terminal; pass --model NAME to run")
    default = next((model for model in models if model["name"] == current), None)
    while True:
        answer = input("Choose model number or name" + (" [Enter keeps current]" if default else "") + ", q to cancel: ").strip()
        if answer.lower() == "q":
            raise ValueError("Model selection canceled; settings unchanged")
        if not answer and default:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(models):
            return models[int(answer) - 1]
        for model in models:
            if answer == model["name"]:
                return model
        print("Choose one of the listed models.")


def save_model(config_path: str, model: dict, mtp: str | None = None) -> None:
    path = Path(config_path).resolve()
    original = path.read_text(encoding="utf-8")
    before = _loads_toml(original)
    cfg = load_config(path)
    if cfg.llamacpp.model_path or cfg.llamacpp.server_command:
        raise ValueError("Managed model selection requires empty llamacpp.model_path and server_command")
    updates = {"model": {"name": model["alias"]}, "llamacpp": {"model_name": model["name"]}}
    if mtp is not None:
        from dataclasses import replace
        # Validate the full proposed settings before writing a backup or config.
        replace(cfg.llamacpp, mtp=mtp)
        updates["llamacpp"]["mtp"] = mtp
    if cfg.active_host:
        from .host_store import read_store, write_store
        store = read_store(config_path)
        profile = store['hosts'][cfg.active_host]
        for section, values in updates.items():
            profile.setdefault(section, {}).update(values)
        write_store(config_path, store)
        return
    save_settings(config_path, updates, label="model choice")


def save_settings(config_path: str, updates: dict, label: str = "settings") -> None:
    path = Path(config_path).resolve()
    original = path.read_text(encoding="utf-8")
    before = _loads_toml(original)
    updated = original
    for section, values in updates.items():
        header = re.search(r"(?m)^\s*\[" + re.escape(section) + r"\][ \t]*(?:#[^\n]*)?$", updated)
        if header is None:
            updated = updated.rstrip() + f"\n\n[{section}]\n"
            header = re.search(r"(?m)^\[" + re.escape(section) + r"\]$", updated)
        start = header.end()
        next_table = re.search(r"(?m)^[ \t]*\[", updated[start:])
        end = start + next_table.start() if next_table else len(updated)
        body = updated[start:end]
        for key, value in values.items():
            pattern = r"(?m)^([ \t]*" + re.escape(key) + r"[ \t]*=[ \t]*)[^\n]*"
            body, count = re.subn(pattern, lambda match: match[1] + json.dumps(value), body)
            if not count:
                body = "\n" + key + " = " + json.dumps(value) + "\n" + body.lstrip("\n")
        updated = updated[:start] + body + updated[end:]
    expected = deepcopy(before)
    for section, values in updates.items():
        expected.setdefault(section, {}).update(values)
    if _loads_toml(updated) != expected:
        raise ValueError("Unexpected config edit; nothing written")
    if updated == original:
        return
    backup = path.with_name(path.name + ".backup-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    shutil.copy2(path, backup)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(updated)
        shutil.copymode(path, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    print(f"Saved {label} in {path} (backup: {backup})")

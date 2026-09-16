"""Per-user SSH settings for the shared EPN installation."""
import re
import shlex
import subprocess
import sys

from .config import load_config
from .models import save_settings

SHARED_WORKDIR = "/scratch/csonnabe/cern-fellowship/misc/lamacpp-llm"


def configure_local(config_path, alias=None, workdir=SHARED_WORKDIR):
    cfg = load_config(config_path)
    if alias is None:
        if not sys.stdin.isatty():
            raise ValueError("SSH alias selection needs a terminal; pass --ssh-alias NAME from ~/.ssh/config")
        alias = input("EPN login-node Host alias from ~/.ssh/config [" + cfg.ssh.host + "]: ").strip() or cfg.ssh.host
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", alias):
        raise ValueError("Use a single SSH Host alias, not a username, command, or URL")
    if not workdir.startswith("/") or "\n" in workdir:
        raise ValueError("Shared remote workdir must be an absolute path")
    workdir = workdir.rstrip("/")
    # Never use the previous user's explicit username; let the alias select it.
    command = " && ".join("test -r " + shlex.quote(workdir + "/" + name) for name in (
        "bin/run-server", "scripts/lib/llamacpp-env.sh", "scripts/epn/llm-epn-serverctl", "scripts/epn/llm-epn-slurm-run"))
    command += " && test -r " + shlex.quote(workdir + "/models")
    for name in ("bin/run-server", "scripts/epn/llm-epn-serverctl", "scripts/epn/llm-epn-slurm-run"):
        command += " && test -x " + shlex.quote(workdir + "/" + name)
    result = subprocess.run(["ssh", "-o", "ConnectTimeout=" + str(cfg.ssh.connect_timeout_seconds),
                             alias, "bash -lc " + shlex.quote(command)], capture_output=True, text=True)
    if result.returncode:
        raise ValueError("Cannot access the shared EPN installation using SSH alias " + alias +
                         ". Check SSH access and shared-directory read permissions. " + result.stderr.strip())
    save_settings(config_path, {
        "ssh": {"host": alias, "user": ""},
        "remote": {"workdir": workdir, "runner": workdir + "/scripts/epn/llm-epn-slurm-run",
                   "serverctl": workdir + "/scripts/epn/llm-epn-serverctl", "state_dir": "$HOME/.cache/llm-epn"},
        "llamacpp": {"build_before_run": False},
    }, label="SSH alias and shared installation")
    print("Shared model directory: " + workdir + "/models (no downloads)")
    print("Jobs and logs belong to the remote account selected by your SSH alias.")

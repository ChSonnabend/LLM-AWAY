"""Create and validate the project environment without changing base Python."""
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def ensure():
    if sys.version_info < (3, 10):
        raise SystemExit("LLM-AWAY requires Python 3.10+; install it with venv support.")
    requirements = ROOT / "requirements.txt"
    signature = hashlib.sha256(requirements.read_bytes() + str(sys.version_info[:2]).encode()).hexdigest()
    directory = ROOT / ".venv"
    python = directory / "bin/python"
    marker = directory / ".requirements"
    with (ROOT / ".venv.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if python.exists() and marker.exists() and marker.read_text() == signature:
            check = subprocess.run([str(python), "-I", "-c", "import pick; import tomllib" if sys.version_info >= (3,11) else "import pick, tomli"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if check.returncode == 0:
                return python
        print("Preparing LLM-AWAY project environment…", file=sys.stderr)
        env = dict(os.environ)
        for key in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(key, None)
        subprocess.run([sys.executable, "-I", "-m", "venv", "--clear", str(directory)], check=True, env=env, stdout=sys.stderr)
        subprocess.run([str(python), "-I", "-m", "pip", "install", "-r", str(requirements)], check=True, env=env, stdout=sys.stderr)
        marker.write_text(signature)
    return python

if __name__ == "__main__":
    print(ensure())

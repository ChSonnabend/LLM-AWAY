import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SETUP = Path(__file__).resolve().parents[1] / "scripts/remote/setup.sh"
HELPERS = ("llm-away-slurm-run", "llm-away-serverctl")


class RemoteSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="away-remote-setup-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workdir = self.root / "runner 'with spaces'"
        for name in ("bin/run-server", "bin/run-cli", "scripts/lib/llamacpp-env.sh"):
            path = self.workdir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# test wrapper\n")
        self.mockbin = self.root / "commands"
        self.mockbin.mkdir()
        # Setup may check for these commands, but must never execute a Slurm job.
        for name in ("sbatch", "squeue", "scancel", "sinfo", "srun", "timeout"):
            self.command(name, "exit 99\n")
        self.dest = self.root / "installed 'helpers'"
        self.env = dict(os.environ, PATH=str(self.mockbin) + ":" + os.environ["PATH"])

    def command(self, name, body):
        path = self.mockbin / name
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(0o755)

    def run_setup(self, *args, success=True):
        result = subprocess.run(
            ["bash", str(SETUP), "--workdir", str(self.workdir),
             "--bin-dir", str(self.dest), *args],
            env=self.env, text=True, capture_output=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_check_is_read_only_and_install_is_idempotent(self):
        self.run_setup("--check", success=False)
        self.assertFalse(self.dest.exists())
        self.run_setup()
        before = {}
        for name in HELPERS:
            path = self.dest / name
            self.assertEqual(path.read_bytes(), (SETUP.parent / name).read_bytes())
            self.assertEqual(path.stat().st_mode & 0o777, 0o755)
            before[name] = path.stat().st_mtime_ns
        self.run_setup("--check")
        self.run_setup()
        self.assertEqual(before, {name: (self.dest / name).stat().st_mtime_ns for name in HELPERS})
        self.assertEqual(sorted(path.name for path in self.dest.iterdir()), sorted(HELPERS))

    def test_replacement_preserves_original_bytes_permissions_and_timestamp(self):
        self.run_setup()
        path = self.dest / HELPERS[0]
        path.write_text("previous helper\n")
        path.chmod(0o700)
        previous_mtime = path.stat().st_mtime_ns
        self.run_setup("--check", success=False)
        self.assertEqual(path.read_text(), "previous helper\n")
        self.run_setup()
        backup, = self.dest.glob(HELPERS[0] + ".backup.*/original")
        self.assertEqual(backup.read_text(), "previous helper\n")
        self.assertEqual(backup.stat().st_mode & 0o777, 0o700)
        self.assertEqual(backup.stat().st_mtime_ns, previous_mtime)
        self.run_setup("--check")

    def test_missing_wrapper_or_unsafe_destination_prevents_all_installation(self):
        wrapper = self.workdir / "bin/run-server"
        wrapper.unlink()
        self.run_setup(success=False)
        self.assertFalse(self.dest.exists())
        wrapper.write_text("# restored\n")
        self.dest.mkdir()
        target = self.root / "unrelated-file"
        target.write_text("keep me\n")
        (self.dest / HELPERS[1]).symlink_to(target)
        self.run_setup(success=False)
        self.assertFalse((self.dest / HELPERS[0]).exists())
        self.assertEqual(target.read_text(), "keep me\n")

    def test_backup_failure_does_not_overwrite_original(self):
        self.run_setup()
        path = self.dest / HELPERS[0]
        path.write_text("previous helper\n")
        self.command("cp", 'if [[ "$1" == -p ]]; then exit 9; fi\nexec /bin/cp "$@"\n')
        self.run_setup(success=False)
        self.assertEqual(path.read_text(), "previous helper\n")
        self.assertFalse(list(self.dest.glob(".llm-away-*")))


if __name__ == "__main__":
    unittest.main()

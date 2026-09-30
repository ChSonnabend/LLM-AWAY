"""Run installer branches with fake tools; never modify real user installations."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


class InstallScriptsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='llm away install ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.local = self.root / 'local'
        self.scripts = self.local / 'scripts'
        self.scripts.mkdir(parents=True)
        (self.local / 'bin').mkdir()
        (self.root / 'vscode-inline').mkdir()
        (self.root / 'vscode-inline/package.json').write_text('{"version":"9.8.7"}')
        for name in ('install.sh', 'install-resource-tools.sh', 'install-vscode-inline.sh', 'init', 'test.sh'):
            shutil.copy2(SCRIPTS / name, self.scripts / name)
        self.log = self.root / 'calls.jsonl'
        self.tools = self.root / 'tools'
        self.tools.mkdir()
        self.python = self.local / 'fixture-python/bin'
        self.python.mkdir(parents=True)
        (self.scripts / 'env.sh').write_text(
            'export VIRTUAL_ENV="${AWAY_LOCAL_ROOT:-${ROOT:-}}/fixture-python"\n'
            'printf \'["bootstrap"]\\n\' >> "$AWAY_INSTALL_TEST_LOG"\n')
        code = '#!' + sys.executable + '\n' + '''import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['AWAY_INSTALL_TEST_LOG'], 'a') as log:
    log.write(json.dumps([name, *args]) + '\\n')
if name == 'node':
    if '-p' in args: print('9.8.7')
    sys.exit(int(os.environ.get('FAIL_NODE', '0')))
if name == 'npm':
    if os.environ.get('FAIL_NPM') == args[0]: sys.exit(1)
    if '--out' in args: Path(args[args.index('--out') + 1]).write_text('mock package')
if name in ('code', 'alternate-code'): sys.exit(int(os.environ.get('FAIL_CODE', '0')))
'''
        for name in ('node', 'npm', 'code', 'alternate-code'):
            p = self.tools / name
            p.write_text(code)
            p.chmod(0o755)
        p = self.python / 'python'
        p.write_text(code)
        p.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.tools) + os.pathsep + os.environ['PATH'],
                        AWAY_INSTALL_TEST_LOG=str(self.log))
        self.env.pop('SSH_CONNECTION', None)
        self.env.pop('SSH_TTY', None)
        self.links = self.root / 'command links'

    def run_script(self, script='install.sh', *args, env=None):
        return subprocess.run(['bash', str(self.scripts / script), *args], env=env or self.env,
                              text=True, capture_output=True, timeout=10)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_default_and_legacy_entry_install_core_and_manifest_extension(self):
        for entry in ('install.sh', 'install-resource-tools.sh'):
            result = self.run_script(entry, '--bin-dir', str(self.links))
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.links / 'res-alloc').readlink(), self.local / 'bin/resource-allocator')
        self.assertEqual(self.calls().count(['bootstrap']), 2)
        self.assertIn(['npm', 'ci', '--include=dev', '--ignore-scripts'], self.calls())
        package = str(self.local / 'run/artifacts/llm-away-inline-9.8.7.vsix')
        self.assertIn(['code', '--install-extension', package, '--force'], self.calls())

    def test_cli_only_preserves_user_files_and_unrelated_symlinks(self):
        self.links.mkdir()
        (self.links / 'run').write_text('user command')
        (self.links / 'res-mon').symlink_to(self.root / 'unrelated-missing-file')
        (self.links / 'add-serve').symlink_to(self.local / 'bin/add-serve')
        result = self.run_script('install.sh', '--without-vscode', '--bin-dir', str(self.links))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.links / 'run').read_text(), 'user command')
        self.assertEqual((self.links / 'res-mon').readlink(), self.root / 'unrelated-missing-file')
        self.assertFalse((self.links / 'add-serve').is_symlink())
        self.assertEqual(self.calls(), [['bootstrap']])

    def test_extension_only_wrapper_skips_python(self):
        result = self.run_script('install-vscode-inline.sh', '--code-command', str(self.tools / 'alternate-code'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(['bootstrap'], self.calls())
        self.assertTrue(any(call[0] == 'alternate-code' for call in self.calls()))

    def test_optional_prerequisite_failure_is_clear_but_required_fails(self):
        env = dict(self.env, FAIL_NODE='1')
        auto = self.run_script('install.sh', '--bin-dir', str(self.links), env=env)
        self.assertEqual(auto.returncode, 0)
        self.assertIn('not installed', auto.stdout)
        required = self.run_script('install-vscode-inline.sh', env=env)
        self.assertEqual(required.returncode, 1)
        self.assertFalse(any(c[0] == 'npm' for c in self.calls()))

    def test_failed_build_never_installs_stale_package(self):
        env = dict(self.env, FAIL_NPM='run')
        auto = self.run_script('install.sh', '--bin-dir', str(self.links), env=env)
        self.assertEqual(auto.returncode, 0)
        self.assertIn('incomplete', auto.stderr)
        required = self.run_script('install-vscode-inline.sh', env=env)
        self.assertEqual(required.returncode, 1)
        self.assertFalse(any(c[0] == 'code' for c in self.calls()))

    def test_auto_skips_ssh_with_explicit_override(self):
        env = dict(self.env, SSH_CONNECTION='remote shell')
        result = self.run_script('install.sh', '--bin-dir', str(self.links), env=env)
        self.assertEqual(result.returncode, 0)
        self.assertIn('SSH shell', result.stdout)
        self.assertEqual(self.calls(), [['bootstrap']])
        override = self.run_script('install-vscode-inline.sh', env=env)
        self.assertEqual(override.returncode, 0, override.stderr)
        self.assertTrue(any(c[0] == 'code' for c in self.calls()))

    def test_auto_skips_remote_cli_even_without_ssh_environment(self):
        remote = self.root / 'remote-cli'
        remote.mkdir()
        shutil.copy2(self.tools / 'code', remote / 'code')
        result = self.run_script('install.sh', '--bin-dir', str(self.links), '--code-command', str(remote / 'code'))
        self.assertEqual(result.returncode, 0)
        self.assertIn('Remote CLI', result.stdout)
        self.assertEqual(self.calls(), [['bootstrap']])

    def test_init_retains_document_options_without_command_install(self):
        result = self.run_script('init', '--check-documents')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [['bootstrap'], ['python', '-m', 'llm_away.document_tools', '--check']])

    def test_help_invalid_flags_and_conflicts_do_not_bootstrap(self):
        for args, status in [(('--help',), 0), (('--unknown',), 2), (('--code-command',), 2),
                             (('--vscode-only', '--documents'), 2), (('--env-only', '--vscode-only'), 2),
                             (('--bin-dir', '--without-vscode'), 2), (('--env-only', '--with-vscode'), 2)]:
            with self.subTest(args=args):
                result = self.run_script('install.sh', *args)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertEqual(self.calls(), [])

    def test_test_script_uses_project_interpreter(self):
        result = self.run_script('test.sh', '-k', 'inline')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [['bootstrap'], ['python', '-m', 'unittest', 'discover', '-s', 'tests', '-v', '-k', 'inline']])

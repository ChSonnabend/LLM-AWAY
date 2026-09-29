import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from llm_away.helper_system import list_directory, read_file, request_system_command, register_system_tools


class HelperSystemTests(unittest.TestCase):
    def test_listing_includes_hidden_binary_and_symlink_with_pagination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '.hidden').touch()
            (root / 'model.pt').write_bytes(b'\0')
            (root / 'link').symlink_to(root / 'missing')
            first = json.loads(list_directory(tmp, limit=2))
            second = json.loads(list_directory(tmp, offset=first['next_offset'], limit=2))
            self.assertEqual(first['total'], 3)
            self.assertEqual([x['name'] for x in first['entries'] + second['entries']], ['.hidden', 'link', 'model.pt'])
            self.assertEqual(first['entries'][1]['type'], 'symlink')
            self.assertIsNone(second['next_offset'])

    def test_reads_without_rag_and_bounds_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'file.txt'
            path.write_text('abcdef')
            first = json.loads(read_file(str(path), max_bytes=3))
            self.assertEqual(first['text'], 'abc')
            self.assertEqual(first['next_offset'], 3)
            second = json.loads(read_file(str(path), offset=3, max_bytes=3))
            self.assertEqual(second['text'], 'def')
            self.assertIsNone(second['next_offset'])

    def test_rejects_binary_and_fifo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'binary'
            path.write_bytes(b'a\0b')
            with self.assertRaises(ValueError): read_file(str(path))
            fifo = Path(tmp) / 'fifo'
            os.mkfifo(fifo)
            with self.assertRaises(ValueError): read_file(str(fifo))

    def test_command_handoff_never_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'must-not-exist'
            result = json.loads(request_system_command(f'touch {target}', tmp, 'test write'))
            self.assertEqual(result['status'], 'approval_required')
            self.assertFalse(result['executed'])
            self.assertFalse(target.exists())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): list_directory('.')
        with self.assertRaises(ValueError): list_directory('/tmp', limit=0)
        with self.assertRaises(ValueError): read_file('/tmp/file', max_bytes=60001)
        with self.assertRaises(ValueError): request_system_command('', '/tmp', 'reason')

    def test_scoped_tools_deny_outside_roots_and_symlink_escape(self):
        registered = {}
        class Server:
            def tool(self, name=None):
                def decorate(fn):
                    registered[name or fn.__name__] = fn
                    return fn
                return decorate
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            register_system_tools(Server(), [tmp])
            self.assertEqual(json.loads(registered['list_directory'](tmp))['total'], 0)
            with self.assertRaises(ValueError): registered['list_directory'](other)
            (Path(tmp) / 'escape').symlink_to(other)
            with self.assertRaises(ValueError): registered['list_directory'](str(Path(tmp) / 'escape'))
            register_system_tools(Server())
            with self.assertRaises(ValueError): registered['list_directory'](tmp)

    def test_access_file_updates_live_and_empty_revokes(self):
        registered = {}
        class Server:
            def tool(self, name=None):
                def decorate(fn):
                    registered[name or fn.__name__] = fn
                    return fn
                return decorate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root/'nested'; child.mkdir()
            file = child/'hello.txt'; file.write_text('hello')
            settings = root/'helper-access.json'
            register_system_tools(Server(), [], access_file=settings)
            with self.assertRaises(ValueError): registered['read_file'](str(file))
            settings.write_text(json.dumps({'read_roots': [tmp]}))
            self.assertEqual(json.loads(registered['read_file'](str(file)))['text'], 'hello')
            settings.write_text(json.dumps({'read_roots': []}))
            with self.assertRaises(ValueError): registered['read_file'](str(file))

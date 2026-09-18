import io
import os
import pty
import select
import subprocess
import sys
import termios
import time
import unittest
from unittest.mock import patch

from llm_away.server import ProviderHandler
from llm_away.models import choose_model, choose_mtp


class DisconnectTests(unittest.TestCase):
    def test_disconnect_does_not_write_an_error_to_closed_connection(self):
        for error in (BrokenPipeError(), ConnectionResetError(), ConnectionAbortedError()):
            handler = object.__new__(ProviderHandler)
            handler.path = '/v1/responses'
            handler.read_json = lambda: {}
            handler.handle_responses = lambda payload: (_ for _ in ()).throw(error)
            handler.write_json = lambda *a, **kw: self.fail('Attempted second write')
            handler.do_POST()
            self.assertTrue(handler.close_connection)

    def test_disconnected_error_response_is_also_quiet(self):
        handler = object.__new__(ProviderHandler)
        handler.read_json = lambda: (_ for _ in ()).throw(ValueError('bad request'))
        handler.write_json = lambda *a, **kw: (_ for _ in ()).throw(BrokenPipeError())
        handler.do_POST()
        self.assertTrue(handler.close_connection)

    def test_unavailable_tool_completes_instead_of_retrying_transport(self):
        handler = object.__new__(ProviderHandler)
        handler.begin_sse = lambda: None
        events = []
        handler.write_sse = lambda event, payload: events.append((event, payload))
        class Backend:
            def infer(self, request):
                return '<tool_call>{"name":"apply_patch","arguments":{"input":"patch"}}</tool_call>'
        handler.backend = Backend()
        handler.handle_responses_stream('away', 'test', tools=[{'type':'function','name':'exec_command'}])
        self.assertEqual(events[-1][0], 'response.completed')
        self.assertIn('No tool from these attempts was executed', events[-1][1]['response']['output_text'])


class TerminalTests(unittest.TestCase):
    def run_menu(self, keys):
        master, slave = pty.openpty()
        original = termios.tcgetattr(slave)
        code = '''from llm_away.prompt import choose_option, PromptCanceled
import termios, sys
before = termios.tcgetattr(sys.stdin.fileno())
try:
 print("RESULT", choose_option(["one", "two", "three"], "test"), flush=True)
except PromptCanceled:
 print("CANCELED", flush=True)
after = termios.tcgetattr(sys.stdin.fileno())
# macOS sets the transient PENDIN status when canonical mode is restored.
after[3] &= ~getattr(termios, "PENDIN", 0)
before[3] &= ~getattr(termios, "PENDIN", 0)
assert after == before
'''
        process = subprocess.Popen([sys.executable, '-c', code], stdin=slave, stdout=slave, stderr=slave,
                                   env=dict(os.environ, PYTHONPATH='src', TERM='xterm'))
        output = b''
        try:
            deadline = time.monotonic() + 5
            while b'q/Esc cancels' not in output:
                if time.monotonic() > deadline:
                    self.fail('Menu did not render: '+repr(output))
                if select.select([master], [], [], .1)[0]:
                    output += os.read(master, 4096)
            os.write(master, keys)
            deadline = time.monotonic() + 5
            while process.poll() is None:
                if time.monotonic() > deadline:
                    self.fail('Menu did not exit: ' + repr(output))
                if select.select([master], [], [], .1)[0]:
                    output += os.read(master, 4096)
            while select.select([master], [], [], .1)[0]:
                output += os.read(master, 4096)
            self.assertEqual(process.returncode, 0, output)
            return output
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
            os.close(master); os.close(slave)

    def test_real_terminal_navigation_and_restoration(self):
        for keys, expected in [(b'\x1b[B\r', 1), (b'\x1bOA\r', 2),
                               (b'\x1b[F\x1b[H\r', 0), (b'\x1b[6~\r', 0),
                               (b'jj\r', 2), (b'jk\r', 0)]:
            with self.subTest(keys=keys):
                self.assertIn(f'RESULT {expected}'.encode(), self.run_menu(keys))

    def test_real_terminal_cancellation_restores_terminal(self):
        for key in (b'q', b'\x03', b'\x04', b'\x1b'):
            with self.subTest(key=key):
                self.assertIn(b'CANCELED', self.run_menu(key))

    def test_model_and_mtp_use_menu_and_preserve_default(self):
        models = [{'name':name, 'size_bytes':1} for name in ('one','two')]
        with patch('llm_away.models.interactive_available',return_value=True), patch('llm_away.models.choose_option',return_value=0) as menu:
            self.assertEqual(choose_model(models,'two'),models[0])
            self.assertEqual(menu.call_args.args[2],1)
            mtp={'mtp':{'available':True,'toggle_supported':True}}
            self.assertEqual(choose_mtp(mtp,'off',interactive=True),'on')
            self.assertEqual(menu.call_args.args[2],1)

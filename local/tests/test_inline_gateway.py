import threading
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm_away.config import AppConfig
from llm_away.server import ProviderHandler


class InlineGatewayTests(unittest.TestCase):
    def setUp(self):
        self.server = SimpleNamespace(inline_lock=threading.Lock(), inline_fault=False)
        self.backend = Mock(native_tools=True)
        self.backend.local_url.return_value = 'http://127.0.0.1:1234/v1/chat/completions'
        self.backend.auth_headers.return_value = {}

    def handler(self):
        h = object.__new__(ProviderHandler)
        h.server = self.server
        h.config = AppConfig()
        h.backend = self.backend
        h.write_json = Mock()
        return h

    def test_disconnected_editor_does_not_release_gate_before_upstream_finishes(self):
        entered = threading.Event()
        finish = threading.Event()
        errors = []
        upstream = Mock(status=200)
        upstream.__enter__ = Mock(return_value=upstream)
        upstream.__exit__ = Mock(return_value=False)
        def read(_):
            entered.set()
            if not finish.wait(2): raise RuntimeError('test timed out')
            return b'{"choices": []}'
        upstream.read.side_effect = read
        first = self.handler()
        first.write_json.side_effect = BrokenPipeError('editor closed')
        def run():
            try: first.handle_inline({'max_tokens': 128})
            except Exception as exc: errors.append(exc)
        with patch('llm_away.server.local_urlopen', return_value=upstream) as request:
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                second = self.handler()
                second.handle_inline({'max_tokens': 128})
                self.assertEqual(second.write_json.call_args.kwargs['status'], 429)
                self.assertEqual(request.call_count, 1)
                first.write_json.assert_not_called()
            finally:
                finish.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(errors[0], BrokenPipeError)
        self.assertFalse(self.server.inline_lock.locked())
        self.assertFalse(self.server.inline_fault)

    def test_ambiguous_upstream_timeout_latches_gate(self):
        with patch('llm_away.server.local_urlopen', side_effect=TimeoutError) as request:
            with self.assertRaises(TimeoutError): self.handler().handle_inline({})
            later = self.handler()
            later.handle_inline({})
            self.assertEqual(later.write_json.call_args.kwargs['status'], 503)
            self.assertEqual(request.call_count, 1)
        self.assertTrue(self.server.inline_fault)

    def test_rejects_unbounded_streaming_or_tool_requests_without_inference(self):
        for payload in ({'max_tokens': 99999}, {'stream': True}, {'tools': [{}]}, {'max_tokens': True}):
            h = self.handler()
            h.handle_inline(payload)
            self.assertEqual(h.write_json.call_args.kwargs['status'], 400)
            self.assertFalse(self.server.inline_lock.locked())
        self.backend.ensure_ready.assert_not_called()

    def test_prefix_suffix_routes_to_raw_fim_completion(self):
        upstream = Mock(status=200)
        upstream.__enter__ = Mock(return_value=upstream)
        upstream.__exit__ = Mock(return_value=False)
        upstream.read.return_value = b'{"choices": [{"text": "x"}]}'
        h = self.handler()
        self.backend.local_url.side_effect = lambda path: 'http://127.0.0.1:1234' + path
        with patch('llm_away.server.local_urlopen', return_value=upstream) as request:
            h.handle_inline({'prefix': 'def f():\n    ', 'suffix': '\n', 'max_tokens': 64})
        sent = json.loads(request.call_args.args[0].data)
        self.assertTrue(request.call_args.args[0].full_url.endswith('/v1/completions'))
        self.assertEqual(sent['prompt'], '<|fim_prefix|>def f():\n    <|fim_suffix|>\n<|fim_middle|>')
        self.assertNotIn('messages', sent)
        self.assertIn('<|fim_middle|>', sent['stop'])

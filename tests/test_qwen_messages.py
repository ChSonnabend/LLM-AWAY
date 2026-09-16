import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from llm_epn.backends import BackendError, SlurmServerBackend
from llm_epn.config import AppConfig, GatewayConfig
from llm_epn.protocol import normalize_chat_messages, responses_request_to_messages
from llm_epn.server import ProviderHandler


class QwenMessagesTests(unittest.TestCase):
    def test_interleaved_instructions_are_merged_without_changing_history(self):
        history = [{"role": "system", "content": "First instruction"},
                   {"role": "user", "content": "Old question"},
                   {"role": "assistant", "content": "Old answer"},
                   {"role": "developer", "content": [{"type": "input_text", "text": "Later instruction"}]},
                   {"role": "system", "content": "Last instruction"},
                   {"role": "user", "content": "New question"}]
        original = json.dumps(history)
        messages = normalize_chat_messages(history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[0]["content"], "First instruction\n\nLater instruction\n\nLast instruction")
        self.assertEqual(messages[1:], [history[1], history[2], history[5]])
        self.assertEqual(json.dumps(history), original)
        self.assertEqual(normalize_chat_messages(messages), messages)

    def test_responses_history_is_valid_for_single_system_template(self):
        messages = responses_request_to_messages({"instructions": "Base", "input": [
            {"role": "user", "content": "Question"},
            {"role": "developer", "content": "Environment update"},
            {"type": "function_call", "name": "exec_command", "arguments": '{}'},
            {"type": "function_call_output", "call_id": "x", "output": "Result"},
            {"role": "user", "content": "Next question"}]})
        backend = SlurmServerBackend(AppConfig(gateway=GatewayConfig(local_port=9999)))
        payload = json.loads(backend.completion_payload("fallback", messages=messages))
        self.assertEqual([i for i, m in enumerate(payload["messages"]) if m["role"] == "system"], [0])
        self.assertIn("Environment update", messages[0]["content"])
        self.assertIn("tool_call", messages[2]["content"])
        self.assertEqual(messages[-1]["content"], "Next question")

    def test_chat_endpoint_preserves_roles_and_epn_alias(self):
        handler = object.__new__(ProviderHandler)
        handler.config = AppConfig()
        seen = []
        class Backend:
            def infer(self, request):
                seen.append(request)
                return "OK"
        handler.backend = Backend()
        handler.write_json = lambda result: None
        handler.handle_chat({"model": "epn", "messages": [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": "Hello"},
            {"role": "developer", "content": "Update"}]})
        self.assertEqual(seen[0].model, "epn")
        self.assertEqual(seen[0].messages, [{"role": "system", "content": "Base\n\nUpdate"}, {"role": "user", "content": "Hello"}])

    def test_upstream_template_error_is_reported_in_failed_stream(self):
        backend = SlurmServerBackend(AppConfig(gateway=GatewayConfig(local_port=9999)))
        detail = "System message must be at the beginning."
        body = json.dumps({"error": {"message": detail}}).encode()
        error = HTTPError('http://localhost', 500, 'Internal Server Error', {}, io.BytesIO(body))
        with patch('llm_epn.backends.urlopen', side_effect=error):
            with self.assertRaisesRegex(BackendError, detail):
                backend.completion("Hello")
        handler = object.__new__(ProviderHandler)
        handler.begin_sse = lambda: None
        events = []
        handler.write_sse = lambda event, value: events.append((event, value))
        class FailingBackend:
            def infer(self, request):
                raise BackendError(detail)
        handler.backend = FailingBackend()
        handler.handle_responses_stream('epn', 'Hello', messages=[])
        self.assertEqual([event for event, _ in events], ['response.created', 'response.failed'])
        self.assertIn(detail, events[-1][1]['response']['error']['message'])
        self.assertEqual(events[-1][1]['response']['error']['code'], "server_error")

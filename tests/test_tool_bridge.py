import json
import unittest

from llm_epn.protocol import response_object, responses_request_to_messages, parse_tool_call
from llm_epn.server import ProviderHandler


class ToolBridgeTests(unittest.TestCase):
    def test_shell_history_round_trip_preserves_quotes_without_json_wrapping(self):
        args = {'cmd': "python3 - <<'PY'\nprint('hello')\nPY", 'max_output_tokens': 1000}
        history = responses_request_to_messages({'input': [
            {'type': 'function_call', 'name': 'exec_command', 'arguments': json.dumps(args)}]})
        markup = history[-1]['content']
        self.assertNotIn('\\\\', markup)
        tools = [{'type': 'function', 'name': 'exec_command', 'parameters': {'properties': {
            'max_output_tokens': {'type': 'integer'}}}}]
        result = response_object('epn', markup, tools=tools)
        self.assertEqual(json.loads(result['output'][0]['arguments']), args)
        self.assertEqual(result['output_text'], '')

    def test_patch_round_trip_and_streaming(self):
        patch = '*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch'
        messages = responses_request_to_messages({'input': [
            {'type': 'custom_tool_call', 'name': 'apply_patch', 'input': patch},
            {'type': 'custom_tool_call_output', 'call_id': 'x', 'output': 'Success'}]})
        self.assertIn('Success', messages[-1]['content'])
        result = response_object('epn', messages[-2]['content'], tools=[{'type': 'custom', 'name': 'apply_patch'}])
        item = result['output'][0]
        self.assertEqual(item['type'], 'custom_tool_call')
        self.assertEqual(item['input'], patch)
        self.assertNotIn('arguments', item)
        handler = object.__new__(ProviderHandler)
        handler.begin_sse = lambda: None
        events = []
        handler.write_sse = lambda event, payload: events.append((event, payload))
        handler.write_response_stream(result)
        done = next(v for e, v in events if e == 'response.custom_tool_call_input.done')
        self.assertEqual(done['input'], patch)

    def test_invalid_or_unavailable_calls_never_become_commands_or_visible_markup(self):
        for text in ['<tool_call>{"name":"exec_command","arguments":"{"cmd":"pwd"}"}</tool_call>',
                     '<tool_call>{"name":"exec_command"}',
                     '<tool_call>{"name":"unknown","arguments":{}}</tool_call>']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                response_object('epn', text, tools=[{'type': 'function', 'name': 'exec_command'}])

    def test_malformed_call_completes_with_error_without_transport_retry(self):
        handler = object.__new__(ProviderHandler)
        handler.begin_sse = lambda: None
        events = []
        handler.write_sse = lambda event, payload: events.append((event, payload))
        class Backend:
            def infer(self, request):
                return '<tool_call>{bad JSON}</tool_call>'
        handler.backend = Backend()
        handler.handle_responses_stream('epn', 'test')
        self.assertEqual(events[-1][0], 'response.completed')
        self.assertIn('No tool from these attempts was executed', events[-1][1]['response']['output_text'])

    def test_one_format_retry_recovers_without_changing_shell_quotes(self):
        handler = object.__new__(ProviderHandler)
        requests = []
        command = 'ls -d "/Applications/Visual Studio Code.app"'
        good = '<tool_call><function=exec_command><parameter=cmd>' + command + '</parameter></function></tool_call>'
        class Backend:
            def infer(self, request):
                requests.append(request)
                return '<tool_call>{bad JSON}</tool_call>' if len(requests) == 1 else good
        handler.backend = Backend()
        original = [{'role': 'user', 'content': 'Inspect VS Code'}]
        result = handler.infer_response('epn', 'Inspect VS Code', messages=original,
                                      tools=[{'type':'function','name':'exec_command'}])
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(original), 1)
        self.assertEqual(json.loads(result['output'][0]['arguments'])['cmd'], command)

    def test_repeated_invalid_calls_stop_after_one_retry(self):
        handler = object.__new__(ProviderHandler)
        requests = []
        class Backend:
            def infer(self, request):
                requests.append(request)
                return '<tool_call>{bad JSON}</tool_call>'
        handler.backend = Backend()
        result = handler.infer_response('epn', 'test')
        self.assertEqual(len(requests), 2)
        self.assertEqual(result['status'], 'completed')
        self.assertIn('invalid tool call twice', result['output_text'])
        self.assertEqual([i['type'] for i in result['output']], ['message'])

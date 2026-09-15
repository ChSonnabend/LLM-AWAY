import unittest

import json

from llm_epn.protocol import messages_to_prompt, models_list, parse_tool_call, response_object, responses_input_to_prompt


class ProtocolTests(unittest.TestCase):
    def test_messages_to_prompt(self):
        prompt = messages_to_prompt([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ])
        self.assertIn("system: Be brief.", prompt)
        self.assertTrue(prompt.endswith("assistant:"))

    def test_responses_string_input(self):
        self.assertEqual(responses_input_to_prompt("Hello"), "Hello")

    def test_responses_message_input(self):
        prompt = responses_input_to_prompt([
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
        ])
        self.assertIn("user: Hi", prompt)

    def test_models_list(self):
        data = models_list("qwen3-coder-next-f16-1m")
        self.assertEqual(data["object"], "list")
        self.assertEqual(data["data"][0]["id"], "qwen3-coder-next-f16-1m")
        self.assertEqual(data["data"][0]["owned_by"], "llm-epn")

    def test_parse_qwen_tool_call(self):
        call = parse_tool_call(
            "I'll inspect it.\n"
            "<tool_call> function=exec <parameter=command> "
            "ls -la /home/chris/alice/misc/LLM_EPN "
            "</parameter> </function>"
        )

        self.assertEqual(call["name"], "exec")
        self.assertEqual(call["arguments"]["command"], "ls -la /home/chris/alice/misc/LLM_EPN")

    def test_response_object_emits_function_call_for_qwen_markup(self):
        response = response_object(
            "qwen3-coder-next-f16-1m",
            "<tool_call> function=exec <parameter=command> pwd </parameter> </function>",
        )
        output = response["output"][0]

        self.assertEqual(response["output_text"], "")
        self.assertEqual(output["type"], "function_call")
        self.assertEqual(output["name"], "exec")
        self.assertEqual(json.loads(output["arguments"]), {"command": "pwd"})

    def test_responses_prompt_includes_function_outputs(self):
        prompt = responses_input_to_prompt(
            [
                {"role": "user", "content": "Inspect files"},
                {
                    "type": "function_call",
                    "name": "exec",
                    "call_id": "call_1",
                    "arguments": "{\"command\":\"ls\"}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "README.md\nsrc"},
            ]
        )

        self.assertIn("assistant tool_call exec:", prompt)
        self.assertIn("tool call_1: README.md\nsrc", prompt)
        self.assertTrue(prompt.endswith("assistant:"))


if __name__ == "__main__":
    unittest.main()

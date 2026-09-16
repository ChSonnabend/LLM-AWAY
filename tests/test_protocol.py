import unittest

import json

from llm_epn.protocol import (
    messages_to_prompt,
    models_list,
    parse_tool_call,
    parse_tool_calls,
    response_object,
    responses_input_to_prompt,
    responses_request_to_prompt,
    responses_request_to_messages,
)


class ProtocolTests(unittest.TestCase):
    def test_responses_messages_keep_question_separate_from_instructions(self):
        messages = responses_request_to_messages({
            "instructions": "Be concise.",
            "input": [
                {"role": "developer", "content": "File access requires permission."},
                {"role": "user", "content": [{"type": "input_text", "text": "Hi, which model are you?"}]},
            ],
        })
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], "Hi, which model are you?")
        self.assertIn("Be concise.", messages[0]["content"])

    def test_responses_messages_preserve_tool_round_trip(self):
        messages = responses_request_to_messages({"input": [
            {"type": "function_call", "name": "exec_command", "arguments": '{"cmd":"pwd"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "/workspace"},
        ]})
        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertIn("exec_command", messages[-2]["content"])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("/workspace", messages[-1]["content"])

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

    def test_responses_request_prompt_includes_tools(self):
        prompt = responses_request_to_prompt(
            {
                "instructions": "Be careful.",
                "tools": [
                    {
                        "type": "function",
                        "name": "exec",
                        "description": "Run a shell command.",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    }
                ],
                "input": [{"role": "user", "content": "Inspect files"}],
            }
        )

        self.assertIn("instructions: Be careful.", prompt)
        self.assertIn("- exec: Run a shell command.", prompt)
        self.assertIn("Do not invent tools such as read_file", prompt)
        self.assertIn("use read-only commands and then answer", prompt)
        self.assertIn("use apply_patch instead of shell heredocs", prompt)
        self.assertIn("user: Inspect files", prompt)

    def test_responses_request_prompt_defaults_to_codex_exec_command(self):
        prompt = responses_request_to_prompt({"input": [{"role": "user", "content": "Inspect files"}]})

        self.assertIn("- exec_command: run a shell command. Parameters: cmd", prompt)
        self.assertIn("prefer the shell tool", prompt)
        self.assertIn("use read-only commands and then answer", prompt)

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

    def test_parse_nested_qwen_tool_calls(self):
        calls = parse_tool_calls(
            "I'll inspect it.\n"
            "<tool_call>\n"
            "<function=read_file>\n"
            "<parameter=path>\n"
            "/home/chris/alice/misc/LLM_EPN/README.md\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=read_file>\n"
            "<parameter=path>\n"
            "/home/chris/alice/misc/LLM_EPN/src/llm_epn/server.py\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        self.assertEqual([call["name"] for call in calls], ["read_file", "read_file"])
        self.assertEqual(calls[0]["arguments"]["path"], "/home/chris/alice/misc/LLM_EPN/README.md")
        self.assertEqual(calls[1]["arguments"]["path"], "/home/chris/alice/misc/LLM_EPN/src/llm_epn/server.py")

    def test_response_object_emits_function_call_for_qwen_markup(self):
        response = response_object(
            "qwen3-coder-next-f16-1m",
            "<tool_call> function=exec <parameter=command> pwd </parameter> </function>",
        )
        output = response["output"][0]

        self.assertEqual(response["output_text"], "")
        self.assertEqual(response["output"][0]["type"], "function_call")
        self.assertEqual(output["type"], "function_call")
        self.assertEqual(output["name"], "exec_command")
        self.assertEqual(json.loads(output["arguments"]), {"cmd": "pwd"})

    def test_response_object_keeps_explicit_exec_tool(self):
        response = response_object(
            "qwen3-coder-next-f16-1m",
            "<tool_call> function=exec <parameter=command> pwd </parameter> </function>",
            tools=[{"type": "function", "name": "exec"}],
        )
        output = response["output"][0]

        self.assertEqual(response["output_text"], "")
        self.assertEqual(output["name"], "exec")
        self.assertEqual(json.loads(output["arguments"]), {"command": "pwd"})

    def test_response_object_preserves_visible_text_before_tool_calls(self):
        response = response_object(
            "qwen3-coder-next-f16-1m",
            "I'll inspect it.\n"
            "<tool_call><function=read_file><parameter=path>README.md</parameter></function></tool_call>",
        )

        self.assertEqual(response["output_text"], "I'll inspect it.")
        self.assertEqual(response["output"][0]["type"], "message")
        self.assertEqual(response["output"][0]["content"][0]["text"], "I'll inspect it.")
        self.assertEqual(response["output"][1]["type"], "function_call")

    def test_response_object_emits_multiple_function_calls(self):
        response = response_object(
            "qwen3-coder-next-f16-1m",
            "<tool_call><function=read_file><parameter=path>README.md</parameter></function></tool_call>"
            "<tool_call><function=read_file><parameter=path>src/llm_epn/server.py</parameter></function></tool_call>",
        )

        self.assertEqual(response["output_text"], "")
        self.assertEqual([item["type"] for item in response["output"]], ["function_call", "function_call"])
        self.assertEqual([item["name"] for item in response["output"]], ["exec_command", "exec_command"])
        self.assertEqual(
            json.loads(response["output"][1]["arguments"]),
            {"cmd": "sed -n '1,240p' -- src/llm_epn/server.py"},
        )

    def test_response_object_rewrites_unsupported_read_file_to_exec(self):
        response = response_object(
            "qwen3-coder-next-f16-1m",
            "<tool_call><function=read_file><parameter=path>README.md</parameter></function></tool_call>",
            tools=[{"type": "function", "name": "exec"}],
        )
        output = response["output"][0]

        self.assertEqual(output["name"], "exec")
        self.assertEqual(json.loads(output["arguments"]), {"command": "sed -n '1,240p' -- README.md"})

    def test_response_object_rewrites_unsupported_read_file_to_exec_command_by_default(self):
        response = response_object(
            "qwen3-coder-next-f16-1m",
            "<tool_call><function=read_file><parameter=path>README.md</parameter></function></tool_call>",
        )
        output = response["output"][0]

        self.assertEqual(output["name"], "exec_command")
        self.assertEqual(json.loads(output["arguments"]), {"cmd": "sed -n '1,240p' -- README.md"})

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

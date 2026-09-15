import unittest

from llm_epn.protocol import messages_to_prompt, models_list, responses_input_to_prompt


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


if __name__ == "__main__":
    unittest.main()

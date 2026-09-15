import unittest

from llm_epn.protocol import messages_to_prompt, responses_input_to_prompt


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


if __name__ == "__main__":
    unittest.main()

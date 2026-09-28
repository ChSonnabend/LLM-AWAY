import unittest
from pathlib import Path
from unittest.mock import patch
from llm_away import resources


class RpcHealthTests(unittest.TestCase):
    def test_drains_entire_status_reply(self):
        with patch.object(resources.socket, 'socket') as factory:
            client = factory.return_value.__enter__.return_value
            client.recv.side_effect = [b'{"allocation":', b'"large status"}', b'\n']
            self.assertTrue(resources.rpc_alive(Path('/tmp/session')))
            self.assertEqual(client.recv.call_count, 3)

    def test_incomplete_reply_is_not_alive(self):
        with patch.object(resources.socket, 'socket') as factory:
            client = factory.return_value.__enter__.return_value
            client.recv.side_effect = [b'{"allocation":', b'']
            self.assertFalse(resources.rpc_alive(Path('/tmp/session')))

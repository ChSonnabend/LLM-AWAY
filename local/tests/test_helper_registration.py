import unittest
from unittest.mock import patch
from llm_away import serve_registration, session_tool


class HelperRegistrationTests(unittest.TestCase):
    def test_helper_checks_accept_live_provider_across_locales(self):
        record = dict(provider_pid=123, provider_identity='Tue Sep 29 14:52:56 2026')
        with patch('llm_away.resources.identity', return_value='Di Sep 29 14:52:56 2026'):
            self.assertTrue(serve_registration.alive(record))
            self.assertTrue(session_tool.provider_alive(record))

    def test_helper_checks_reject_recycled_and_exited_providers(self):
        record = dict(provider_pid=123, provider_identity='Tue Sep 29 14:52:56 2026')
        for observed in ('', 'Di Sep 29 14:52:57 2026'):
            with patch('llm_away.resources.identity', return_value=observed):
                self.assertFalse(serve_registration.alive(record))
                self.assertFalse(session_tool.provider_alive(record))
        with patch('llm_away.resources.identity', return_value=record['provider_identity']):
            self.assertFalse(serve_registration.alive(dict(record, provider_exit=0)))

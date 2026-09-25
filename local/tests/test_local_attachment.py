import unittest
from unittest.mock import patch
from llm_away import resources


class LocalAttachmentTests(unittest.TestCase):
    def test_remote_model_alone_does_not_attach_this_machine(self):
        self.assertFalse(resources.locally_attached({'model':'glm','allocation':{'model_state':'LOADED'}}))

    def test_provider_must_be_live_and_match_generation(self):
        row={'model':'glm','provider_pid':123,'provider_identity':'original',
             'attachment_generation':'a','allocation':{'generation':'a'}}
        with patch.object(resources,'identity',return_value='original'):
            self.assertTrue(resources.locally_attached(row))
            self.assertFalse(resources.locally_attached(dict(row,provider_exit=1)))
            self.assertFalse(resources.locally_attached(dict(row,allocation={'generation':'b'})))
        with patch.object(resources,'identity',return_value='reused PID'):
            self.assertFalse(resources.locally_attached(row))

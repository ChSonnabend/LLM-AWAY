import json
import unittest
from unittest.mock import MagicMock
from urllib.error import URLError
from llm_away.session_tool import helper_endpoint

class HelperEndpointTests(unittest.TestCase):
    def response(self):
        r=MagicMock();r.__enter__.return_value.status=200;return r

    def test_live_tunnel_needs_no_repair(self):
        opener=MagicMock();opener.open.return_value=self.response()
        data={'model':'test','config':{'gateway':{'local_port':1234},'server':{'port':1235}}}
        self.assertEqual(helper_endpoint(data,opener),'http://127.0.0.1:1234')
        opener.open.assert_called_once_with('http://127.0.0.1:1234/health',timeout=5)

    def test_dead_tunnel_repaired_by_owner_without_completion(self):
        opener=MagicMock();opener.open.side_effect=[URLError('Connection refused'),self.response(),self.response()]
        data={'model':'test','config':{'gateway':{'local_port':1234},'server':{'port':1235}}}
        self.assertEqual(helper_endpoint(data,opener),'http://127.0.0.1:1234')
        request=opener.open.call_args_list[1].args[0]
        self.assertEqual(request.full_url,'http://127.0.0.1:1235/v1/messages/count_tokens')
        self.assertEqual(json.loads(request.data)['model'],'test')

    def test_failed_repair_is_not_reported_as_ready(self):
        opener=MagicMock();opener.open.side_effect=[URLError('dead'),URLError('provider dead')]
        data={'model':'test','config':{'gateway':{'local_port':1234},'server':{'port':1235}}}
        with self.assertRaises(URLError):helper_endpoint(data,opener)

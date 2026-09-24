import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from llm_away import gpu_history

class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)

    def sample(self, stamp):
        return {'timestamp':stamp,'lines':['index, uuid, name, used, total, utilization','0, uuid, GPU, 100, 200, 50']}

    def test_reopen_and_pagination_keep_entire_session(self):
        for stamp in range(405):gpu_history.append(self.path,self.sample(stamp))
        first=gpu_history.read(self.path,limit=400)
        self.assertTrue(first['more']);self.assertEqual(len(first['samples']),400)
        rest=gpu_history.read(self.path,first['cursor'])
        self.assertEqual(len(rest['samples']),5);self.assertFalse(rest['more'])
        self.assertEqual(gpu_history.read(self.path)['samples'][0]['gpu_timestamp'],0)
        self.assertEqual(gpu_history.read(self.path,rest['cursor'])['samples'],[])

    def test_concurrent_duplicate_samples_and_isolation(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _:gpu_history.append(self.path,self.sample(7)),range(12)))
        self.assertEqual(len(gpu_history.read(self.path)['samples']),1)
        other=self.path/'other';other.mkdir()
        self.assertEqual(gpu_history.read(other)['samples'],[])

    def test_invalid_samples_do_not_create_history(self):
        for value in [None,{},dict(timestamp=float('nan'),lines=['x']),dict(timestamp=2,lines=[])]:
            gpu_history.append(self.path,value)
        self.assertFalse((self.path/'gpu-history.sqlite3').exists())

    def test_legacy_collector_saves_final_sample_and_exits(self):
        (self.path/'session.json').write_text(json.dumps(dict(phase='RELEASED',allocation={'gpu_telemetry':self.sample(8)})))
        gpu_history.follow([self.path])
        self.assertEqual(gpu_history.read(self.path)['samples'][0]['gpu_timestamp'],8)

    def test_endpoint_restores_persisted_samples_and_cursor(self):
        from unittest.mock import patch
        from llm_away import webapp
        gpu_history.append(self.path,self.sample(10))
        (self.path/'session.json').write_text(json.dumps({'allocation':{'gpu_telemetry':self.sample(11)}}))
        handler=object.__new__(webapp.DashboardHandler)
        handler.path='/api/gpu-history?session=7&after=1'
        with patch.object(webapp,'session_path',return_value=self.path),patch.object(handler,'_json') as send:
            handler.do_GET()
        payload=send.call_args.args[0]
        self.assertEqual([s['gpu_timestamp'] for s in payload['samples']],[11])
        self.assertEqual(len(gpu_history.read(self.path)['samples']),2)

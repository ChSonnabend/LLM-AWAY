import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from llm_away.helper_batch import investigate_batch,load_snapshot,bounded_summary
from llm_away.helper_system import access_checker


def reply(messages,timeout):
    return {'choices':[{'message':{'content':'README.md describes a test project.'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110}}

class BatchTests(unittest.TestCase):
    def test_single_call_compact_private_cache_and_followup(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cache:
            root=Path(tmp);(root/'README.md').write_text('PRIVATE SOURCE\nTest project')
            check=access_checker([tmp]);calls=[]
            def complete(messages,timeout):calls.append(messages);return reply(messages,timeout)
            result=investigate_batch(tmp,'Summarize',check,complete,cache)
            self.assertEqual(len(calls),1)
            self.assertIn('PRIVATE SOURCE',json.dumps(calls))
            self.assertNotIn('PRIVATE SOURCE',json.dumps(result))
            self.assertNotIn('file_reads',result['coverage'])
            detail=load_snapshot(cache,result['investigation_id'],check)['report']
            self.assertEqual(len(detail['coverage']['file_reads']),1)
            with patch('llm_away.helper_batch.read_file',side_effect=AssertionError('must reuse cache')):
                follow=investigate_batch('','What is its purpose?',check,complete,cache,investigation_id=result['investigation_id'])
            self.assertTrue(follow['reused_evidence']);self.assertEqual(len(calls),2)
            self.assertEqual(len(calls[-1]),2) # no growing message history

    def test_revoked_cache_rejected_and_sensitive_dirs_skipped(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cache:
            root=Path(tmp);(root/'token').mkdir();(root/'token'/'README.md').write_text('SECRET')
            (root/'README.md').write_text('safe')
            (root/'node_modules').mkdir()
            result=investigate_batch(tmp,'Summarize',access_checker([tmp]),reply,cache)
            self.assertEqual(result['coverage']['files_read'],1)
            self.assertEqual(result['coverage']['skipped'],2)
            with self.assertRaises(ValueError):load_snapshot(cache,result['investigation_id'],access_checker([]))
            with self.assertRaises(ValueError):load_snapshot(cache,'../escape',access_checker([tmp]))

    def test_budget_and_unknown_usage(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cache:
            (Path(tmp)/'README.md').write_text('x'*10000)
            def complete(messages,timeout):
                self.assertLessEqual(len(json.dumps(messages)),1200)
                r=reply(messages,timeout);r.pop('usage');return r
            result=investigate_batch(tmp,'Summarize',access_checker([tmp]),complete,cache,prompt_budget=1200,max_bytes=1024)
            self.assertLessEqual(result['coverage']['bytes_read'],1024)
            self.assertIsNone(result['usage']['tokens'])
            self.assertGreater(result['usage']['evidence_items_omitted'],0)

    def test_expired_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cache:
            result=investigate_batch(tmp,'Summarize',access_checker([tmp]),reply,cache)
            with patch('llm_away.helper_batch.time.time',return_value=result['snapshot_time']+3601):
                with self.assertRaisesRegex(ValueError,'expired'):load_snapshot(cache,result['investigation_id'],access_checker([tmp]))

    def test_named_followup_excludes_other_projects(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cache:
            root=Path(tmp)
            for name in ('Alpha','Beta'):
                (root/name).mkdir();(root/name/'README.md').write_text(name+' PRIVATE DOCUMENT')
            check=access_checker([tmp])
            first=investigate_batch(tmp,'Overview',check,reply,cache)
            def complete(messages,timeout):
                prompt=json.dumps(messages)
                self.assertIn('Alpha PRIVATE DOCUMENT',prompt)
                self.assertNotIn('Beta PRIVATE DOCUMENT',prompt)
                return reply(messages,timeout)
            result=investigate_batch('','Explain Alpha',check,complete,cache,investigation_id=first['investigation_id'])
            self.assertEqual(result['usage']['new_files_read'],0)
            self.assertEqual(result['usage']['evidence_files_supplied'],1)

    def test_overlong_summary_ends_at_complete_bullet(self):
        answer='- First complete bullet.\n- Second complete bullet.\n- Third bullet is unfinished beyond limit.'
        trimmed=bounded_summary(answer,60)
        self.assertEqual(trimmed,'- First complete bullet.\n- Second complete bullet.')
        self.assertLessEqual(len(trimmed),60)

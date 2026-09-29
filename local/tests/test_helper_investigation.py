import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from llm_away.helper_investigation import investigate
from llm_away.helper_system import access_checker


def reply(action, usage=True):
    result={'choices':[{'message':{'content':json.dumps(action)},'finish_reason':'stop'}]}
    if usage:result['usage']={'prompt_tokens':100,'completion_tokens':10,'total_tokens':110}
    return result


class InvestigationTests(unittest.TestCase):
    def test_private_reads_counts_and_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'notes.txt';path.write_text('PRIVATE RAW CONTENT')
            calls=[]
            def complete(messages,timeout):
                calls.append(json.loads(json.dumps(messages)))
                return reply({'action':'read_file','path':str(path)} if len(calls)==1 else {'action':'finish','summary':'A notes file.'})
            result=investigate(tmp,'Summarize',access_checker([tmp]),complete)
            self.assertIn('PRIVATE RAW CONTENT',json.dumps(calls[-1]))
            self.assertNotIn('PRIVATE RAW CONTENT',json.dumps(result))
            self.assertEqual(result['coverage']['files_read'],1)
            self.assertEqual(result['usage']['tokens']['total_tokens'],220)
            self.assertEqual(result['coverage']['directory_pages'][0]['page_counts'],{'.txt':1})

    def test_outside_scope_symlink_and_shell_denied(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            (Path(tmp)/'escape').symlink_to(outside)
            actions=iter([{'action':'list_directory','path':str(Path(tmp)/'escape')},
                          {'action':'shell','path':tmp,'command':'touch BAD'},
                          {'action':'finish','summary':'Restricted scope.'}])
            result=investigate(tmp,'Summarize',access_checker([tmp,outside]),lambda *a:reply(next(actions)))
            self.assertEqual(len(result['coverage']['errors']),2)
            self.assertEqual(result['coverage']['files_read'],0)

    def test_revocation_applies_to_each_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy=Path(tmp)/'access.json';policy.write_text(json.dumps({'read_roots':[tmp]}))
            target=Path(tmp)/'note';target.write_text('secret')
            count=0
            def complete(*args):
                nonlocal count
                count+=1
                policy.write_text(json.dumps({'read_roots':[]}))
                return reply({'action':'read_file','path':str(target)} if count==1 else {'action':'finish','summary':'Access revoked.'})
            result=investigate(tmp,'Summarize',access_checker([],policy),complete)
            self.assertEqual(result['coverage']['files_read'],0)
            self.assertIn('outside configured',result['coverage']['errors'][0])

    def test_limits_and_missing_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            count=0
            def complete(*args):
                nonlocal count
                count+=1
                return reply({'action':'list_directory','path':tmp},usage=False)
            result=investigate(tmp,'Summarize',access_checker([tmp]),complete,max_turns=2)
            self.assertEqual(count,2)
            self.assertEqual(result['stop_reason'],'turn_limit')
            self.assertIsNone(result['usage']['tokens'])
            with self.assertRaises(ValueError):investigate(tmp,'q',access_checker([tmp]),complete,max_turns=100)

    def test_binary_and_malformed_model_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'binary';target.write_bytes(b'a\0b')
            replies=iter([reply({'action':'read_file','path':str(target)}),
                          {'choices':[{'message':{'content':'not json'}}]},
                          reply({'action':'finish','summary':'Binary could not be inspected.'})])
            result=investigate(tmp,'Summarize',access_checker([tmp]),lambda *a:next(replies))
            self.assertEqual(len(result['coverage']['errors']),2)
            self.assertEqual(result['coverage']['files_read'],0)

    def test_file_byte_budget_and_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'large.txt';target.write_text('x'*5000)
            count=0
            def complete(*args):
                nonlocal count
                count+=1
                return reply({'action':'read_file','path':str(target),'offset':0} if count<3 else {'action':'finish','summary':'Partial read.'})
            result=investigate(tmp,'Summarize',access_checker([tmp]),complete,max_bytes=1024)
            self.assertEqual(result['coverage']['bytes_read'],1024)
            self.assertIn('Read-byte limit',result['coverage']['errors'][0])
            with patch('llm_away.helper_investigation.time.monotonic',side_effect=[0,200]):
                result=investigate(tmp,'Summarize',access_checker([tmp]),complete,timeout_seconds=10)
            self.assertEqual(result['stop_reason'],'time_limit')
            self.assertEqual(result['usage']['model_calls'],0)

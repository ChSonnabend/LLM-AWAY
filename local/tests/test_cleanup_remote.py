import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from contextlib import ExitStack
from dataclasses import asdict

from llm_away import cleanup
from llm_away.config import AppConfig


class RemoteCleanupTests(unittest.TestCase):
    def test_preview_recovery_with_remote_controller(self):
        import subprocess,sys
        script=Path(__file__).resolve().parents[2]/'remote/bin/resource-control'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);session=root/'1';session.mkdir()
            token='a'*32;state=root/'remote/resources'/token;state.mkdir(parents=True)
            (state/'allocation.json').write_text('{}')
            (state/'worker.state').write_text('STOPPED host')
            cfg=asdict(AppConfig());cfg['remote']['resource_state_dir']=str(root/'remote');cfg['backend_type']='direct'
            (session/'session.json').write_text(json.dumps({'phase':'RELEASED','config':cfg,'token':token,'remote_port':8080}))
            def call(config,token,port,action,**options):
                result=subprocess.run([sys.executable,str(script),action],input=json.dumps(dict(config=asdict(config),token=token,port=port,**options)),text=True,capture_output=True)
                if result.returncode:raise RuntimeError(result.stderr)
                return json.loads(result.stdout)
            items=[('remote',session,'remove remote state')]
            with patch.object(cleanup,'remote',side_effect=call):
                plan=cleanup.preview(items)
                self.assertTrue(state.exists())
                self.assertIn(str((state/'release').resolve()),plan['remote_paths'][str(session)])
                self.assertEqual(cleanup.preview(items),plan)
                cleanup.execute(items,[0],expected_remote=plan['remote_paths'])
            self.assertFalse(state.exists())

    def test_preview_repairs_release_before_capturing_and_deleting_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            session=Path(temp)/'1';session.mkdir()
            (session/'session.json').write_text(json.dumps({'phase':'RELEASED','config':asdict(AppConfig()),'token':'a'*32,'remote_port':8080}))
            paths=['/remote/state','/remote/state/release','/remote/state/release-audit.jsonl']
            calls=[]
            def call(_cfg,_token,_port,action,**options):
                calls.append(action)
                if action=='cleanup-preview':
                    if 'release' not in calls:raise RuntimeError('allocation is not explicitly released')
                    return {'paths':paths}
                if action=='status':return {'active':False}
                if action=='release':
                    self.assertEqual(options,{'explicit_release':True,'only_if_inactive':True})
                    return {'released':True}
                self.assertEqual(options,{'expected_paths':paths})
                return {'cleaned':True}
            items=[('remote',session,'remove remote state')]
            with patch.object(cleanup,'busy',return_value=False),patch.object(cleanup,'remote',side_effect=call):
                plan=cleanup.preview(items)
                self.assertNotIn('cleanup',calls)
                self.assertEqual(cleanup.preview(items),plan)
                cleanup.execute(items,[0],expected_remote=plan['remote_paths'])
            self.assertEqual(calls,['cleanup-preview','status','release','cleanup-preview','cleanup-preview','cleanup'])
            self.assertTrue(json.loads((session/'session.json').read_text())['remote_cleanup_done'])

    def test_preview_preserves_active_uncertain_and_unreleased_sessions(self):
        for phase,busy,status in [('RELEASED',False,{'active':True}),('RELEASED',False,{}),('RUNNING',False,{'active':False}),('RELEASED',True,{'active':False})]:
            with self.subTest(phase=phase,busy=busy,status=status),tempfile.TemporaryDirectory() as temp:
                session=Path(temp)
                (session/'session.json').write_text(json.dumps({'phase':phase,'config':asdict(AppConfig()),'token':'a'*32,'remote_port':8080}))
                def call(_cfg,_token,_port,action,**options):
                    if action=='cleanup-preview':raise RuntimeError('not explicitly released')
                    self.assertEqual(action,'status')
                    return status
                with patch.object(cleanup,'busy',return_value=busy),patch.object(cleanup,'remote',side_effect=call) as remote:
                    with self.assertRaises((RuntimeError,ValueError)):
                        cleanup.preview([('remote',session,'remove remote state')])
                self.assertNotIn('release',[c.args[3] for c in remote.call_args_list])

    def test_preview_does_not_repair_unrelated_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            session=Path(temp)
            (session/'session.json').write_text(json.dumps({'phase':'RELEASED','config':asdict(AppConfig()),'token':'a'*32,'remote_port':8080}))
            with patch.object(cleanup,'remote',side_effect=RuntimeError('offline')) as remote:
                with self.assertRaisesRegex(RuntimeError,'offline'):
                    cleanup.preview([('remote',session,'remove remote state')])
            remote.assert_called_once()

    def test_remote_failure_keeps_local_record_and_success_allows_removal(self):
        for fails in (True,False):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root=Path(temp);session=root/'1';session.mkdir()
                (session/'session.json').write_text(json.dumps({'phase':'RELEASED','config':asdict(AppConfig()),'token':'a'*32,'remote_port':8080}))
                stack.enter_context(patch.object(cleanup,'STORE',root))
                stack.enter_context(patch.object(cleanup,'released_ids',return_value=[1]))
                stack.enter_context(patch.object(cleanup,'busy',return_value=False))
                stack.enter_context(patch('sys.argv',['res-clean','--session','1']))
                stack.enter_context(patch('builtins.input',side_effect=['all','y']))
                remote=stack.enter_context(patch.object(cleanup,'remote',side_effect=RuntimeError('offline') if fails else None,return_value={'cleaned':True}))
                cleanup.main()
                self.assertEqual(session.exists(),fails)
                self.assertEqual(remote.call_args.args[3],'cleanup')

    def test_remote_cleanup_handles_empty_retry_and_lock_outside_state(self):
        import subprocess,sys
        script=Path(__file__).resolve().parents[2]/'remote/bin/resource-control'
        for empty in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);token='a'*32;state=root/'resources'/token;state.mkdir(parents=True)
                if not empty:
                    (state/'allocation.json').write_text('{}')
                    (state/'release').touch()
                    (state/'worker.state').write_text('STOPPED host')
                cfg=asdict(AppConfig());cfg['remote']['resource_state_dir']=tmp;cfg['backend_type']='direct'
                payload=json.dumps({'config':cfg,'token':token,'port':1})
                for _ in range(2):
                    result=subprocess.run([sys.executable,str(script),'cleanup'],input=payload,text=True,capture_output=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertTrue(json.loads(result.stdout)['cleaned'])
                    self.assertFalse(state.exists())
                self.assertTrue((root/'resources/.locks'/ (token+'.lock')).exists())

    def test_stale_tunnel_record_is_removed_even_when_its_process_is_gone(self):
        with tempfile.TemporaryDirectory() as temp:
            session=Path(temp)/'1';session.mkdir()
            (session/'session.json').write_text(json.dumps({'phase':'RELEASED'}))
            tunnel=session/'tunnel.json'
            tunnel.write_text(json.dumps({'pid':12345,'identity':'old'}))
            with patch.object(cleanup,'busy',return_value=False),patch.object(cleanup,'identity',return_value=''):
                cleanup.execute([('ssh',session,'remove tunnel')],[0])
            self.assertFalse(tunnel.exists())

    def test_dead_control_socket_is_unlinked_even_when_ssh_refuses_it(self):
        with tempfile.TemporaryDirectory() as temp:
            session=Path(temp)/'1';session.mkdir()
            (session/'session.json').write_text(json.dumps({'phase':'RELEASED'}))
            socket=session/'ssh.sock';socket.touch()
            failed=MagicMock(returncode=255,stderr=b'connection refused')
            with patch.object(cleanup,'busy',return_value=False),patch.object(cleanup.subprocess,'run',return_value=failed):
                cleanup.execute([('socket',socket,'remove socket')],[0])
            self.assertFalse(socket.exists())

    def test_released_timed_out_job_repairs_missing_remote_release_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            session=Path(temp)/'1';session.mkdir()
            (session/'session.json').write_text(json.dumps({'phase':'RELEASED','config':asdict(AppConfig()),'token':'a'*32,'remote_port':8080}))
            calls=[]
            def call(_cfg,_token,_port,action,**options):
                calls.append(action)
                if action=='release':self.assertEqual(options,{'explicit_release':True,'only_if_inactive':True})
                if action=='cleanup' and calls.count('cleanup') < 4:
                    raise RuntimeError('not explicitly released')
                if action=='status':return {'active':False}
                return {'cleaned':True} if action=='cleanup' else {'released':True}
            with patch.object(cleanup,'busy',return_value=False),patch.object(cleanup,'remote',side_effect=call),patch.object(cleanup.time,'sleep'):
                cleanup.execute([('remote',session,'remove remote state')],[0])
            self.assertEqual(calls,['cleanup','cleanup','cleanup','status','release','cleanup'])
            self.assertTrue(json.loads((session/'session.json').read_text())['remote_cleanup_done'])

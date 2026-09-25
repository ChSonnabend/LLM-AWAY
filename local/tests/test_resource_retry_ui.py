import curses
import json
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import MagicMock, patch
from llm_away import monitor_ui, resources, loading
from llm_away.config import AppConfig

ROOT=Path(__file__).resolve().parents[2]

class RetryAndMenus(unittest.TestCase):
    def test_pending_allocation_does_not_require_model_stop_acknowledgement(self):
        self.assertTrue(resources.allocation_pending({'slurm_state':'PENDING','host':'','model_state':'STARTING'}))
        self.assertTrue(resources.allocation_pending({'slurm_state':'CONFIGURING','host':'node'}))
        self.assertFalse(resources.allocation_pending({'slurm_state':'RUNNING','host':'node','model_state':'LOADED'}))

    def test_release_falls_back_when_daemon_rejects_model_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);data={'id':6,'token':'token','remote_port':1234,'phase':'PENDING','model':'test',
                'config':asdict(AppConfig()),'allocation':{'active':True,'slurm_state':'PENDING','model_state':'STOPPING'}}
            (path/'session.json').write_text(json.dumps(data))
            with patch.object(resources,'path_for',return_value=path),patch.object(resources,'rpc') as rpc,patch.object(resources,'remote',return_value={'released':True}) as remote:
                result=resources.release_session(6)
            self.assertTrue(result['fallback']);rpc.assert_not_called();remote.assert_called_once()
            saved=json.loads((path/'session.json').read_text())
            self.assertEqual(saved['phase'],'RELEASED');self.assertFalse(saved['allocation']['active'])

    def test_new_generation_masks_previous_failure_and_start_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=AppConfig();cfg=replace(cfg,backend_type='direct',remote=replace(cfg.remote,workdir=tmp),llamacpp=replace(cfg.llamacpp,container=''))
            state=Path(tmp)/'.state/resources'/('a'*32);state.mkdir(parents=True)
            (state/'allocation.json').write_text(json.dumps({'job_id':'123','created':time.time()}))
            (state/'heartbeat').touch();(state/'worker.state').write_text('old EXITED host')
            payload=json.dumps({'config':asdict(cfg),'token':'a'*32,'port':12345})
            def call(action):
                p=subprocess.run([sys.executable,str(ROOT/'remote/bin/resource-control'),action],input=payload,text=True,capture_output=True)
                self.assertEqual(p.returncode,0,p.stderr);return json.loads(p.stdout)
            first=call('start');self.assertEqual(first['model_state'],'STARTING')
            self.assertEqual(call('start')['generation'],first['generation'])
            (state/'worker.state').write_text(first['generation']+' EXITED host')
            self.assertEqual(call('status')['model_state'],'EXITED')
            self.assertNotEqual(call('start')['generation'],first['generation'])

    def test_stale_snapshot_does_not_abort_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec={'loading':{'config':asdict(AppConfig()),'model':{'alias':'test'},'reuse':False,'mtp':'off'}}
            response=MagicMock();response.__enter__.return_value.status=200
            with patch.object(loading,'urlopen',side_effect=[OSError(),response]),patch.object(loading.time,'sleep'),patch.object(resources,'identity',return_value='live'),patch.object(resources,'rpc',return_value={'allocation':{'active':True,'model_state':'EXITED'},'provider_exit':None}):
                self.assertFalse(loading.prepare(Path(tmp),spec))

    def screen(self,keys,**kwargs):
        win=MagicMock();win.getmaxyx.return_value=(30,160);win.getch.side_effect=keys
        row={'id':5,'model':'failed','provider_exit':1,'_busy':False,'allocation':{'model_state':'EXITED'}}
        with patch.object(monitor_ui,'_terminal_screen',side_effect=lambda f:f(win)),patch.object(monitor_ui,'snapshots',return_value=[row]),patch.object(monitor_ui.curses,'curs_set'),patch.object(monitor_ui.curses,'has_colors',return_value=False),patch.object(monitor_ui.curses,'ACS_HLINE',45,create=True):
            return monitor_ui.show(Path('/unused'),kwargs.pop('release',None),None,**kwargs),win

    def test_failed_attach_uses_model_screen(self):
        allocate=MagicMock(return_value=5)
        result,_=self.screen([curses.KEY_F2],allocate=allocate)
        self.assertEqual(result,5);allocate.assert_called_once_with('model',5)

    def test_release_menu_can_cancel_without_mutation(self):
        release=MagicMock();unload=MagicMock()
        self.screen([curses.KEY_F3,27,ord('q')],release=release,unload=unload)
        release.assert_not_called();unload.assert_not_called()

    def test_reload_unloads_then_selects_model(self):
        calls=[]
        self.screen([curses.KEY_F3,ord('2'),ord('q')],unload=lambda n:calls.append(('stop',n)),allocate=lambda m,n:calls.append((m,n)))
        self.assertEqual(calls,[('stop',5),('model',5)])

    def test_f6_selects_resource_monitor(self):
        _,win=self.screen([curses.KEY_F6,ord('4'),ord('q')])
        self.assertTrue(any('GPU telemetry unavailable' in str(c) for c in win.addnstr.call_args_list))

    def test_explicit_batch_options_override_preset(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'bin').mkdir();(root/'scripts/lib').mkdir(parents=True)
            shutil.copy2(ROOT/'remote/bin/run-server',root/'bin/run-server')
            fake=root/'bin/fake'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');fake.chmod(0o755)
            (root/'scripts/lib/llamacpp-env.sh').write_text('''LLAMACPP_DEFAULT_MODEL=test
MODEL_ALIAS=test
LLAMACPP_SERVER_EXTRA_ARGS='--batch-size 8192 --ubatch-size 8192'
llamacpp_load_model_config() { :; }
llamacpp_detect_backend() { echo cpu; }
llamacpp_export_visible_devices() { :; }
llamacpp_gpu_count() { echo 0; }
llamacpp_resolve_model() { echo fake.gguf; }
llamacpp_binary() { echo "$ROOT_DIR/bin/fake"; }
llamacpp_common_run_args() { :; }
''')
            (root/'scripts/lib/container-env.sh').write_text('llamacpp_enter_container() { :; }\n')
            p=subprocess.run(['bash',str(root/'bin/run-server'),'--batch-size','1024','--ubatch-size','1024'],text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr)
            args=p.stdout.splitlines()
            for flag in ('--batch-size','--ubatch-size'):
                self.assertEqual([args[i+1] for i,x in enumerate(args[:-1]) if x==flag][-1],'1024')

    def test_model_batch_defaults_replace_both_flag_forms(self):
        cfg=AppConfig();cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,server_extra_args=['--batch-size=8192','-ub','8192','--flash-attn','on']))
        for name,ub in [('glm-5.3-flash-q4','1024'),('glm-5.3-flash-q8','512')]:
            expected=['--flash-attn','on']
            if name.endswith('-q4'):expected+=['--cache-type-k','q4_0','--cache-type-v','q4_0']
            expected+=['--batch-size','2048','--ubatch-size',ub]
            self.assertEqual(resources.model_server_options(cfg,name),expected)

    def test_monitor_returns_after_attached_agent_exits(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as tmp,patch.object(resources.sys.stdin,'isatty',return_value=True),patch.object(resources.sys.stdout,'isatty',return_value=True),patch.object(monitor_ui,'show',side_effect=[5,None]) as show,patch.object(resources,'path_for',return_value=Path(tmp)),patch.object(resources.subprocess,'call',return_value=0) as launch:
            resources.monitor(Namespace(logs=None,list=False,kill=None,release=False))
            self.assertEqual(show.call_count,2);launch.assert_called_once()

    def test_refresh_keeps_running_and_uncertain_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for n in range(1,4):
                p=root/str(n);p.mkdir();(p/'session.json').write_text(json.dumps({'id':n,'config':asdict(AppConfig()),'token':'a','remote_port':1}))
            with patch.object(resources,'STORE',root),patch.object(resources,'remote',side_effect=[{'active':True,'slurm_state':'RUNNING'},{'active':False,'slurm_state':'CANCELLED'},RuntimeError('SSH down')]),patch.object(resources,'release_session') as release:
                message=resources.refresh_monitor()
                release.assert_not_called();self.assertTrue((root/'2'/'discovery-retired').exists());self.assertIn('SSH down',message)

    def test_refresh_retires_ended_direct_and_kubernetes_allocations(self):
        from dataclasses import replace
        for backend in ('direct','kubernetes'):
            with self.subTest(backend=backend),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);path=root/'1';path.mkdir()
                (path/'session.json').write_text(json.dumps({'id':1,'config':asdict(replace(AppConfig(),backend_type=backend)),'token':'a','remote_port':1}))
                with patch.object(resources,'STORE',root),patch.object(resources,'remote',return_value={'active':False,'slurm_state':'STOPPED'}),patch.object(resources,'release_session') as release:
                    resources.refresh_monitor()
                    self.assertTrue((path/'discovery-retired').exists())
                    release.assert_not_called()

    def test_tools_maintenance_actions(self):
        for key,kind in [(curses.KEY_F4,'refresh'),(curses.KEY_F5,'cleanup')]:
            callback=MagicMock(return_value='Done')
            with patch.object(monitor_ui,'dropdown_win'):
                self.screen([curses.KEY_F1,key,ord('q')],**{kind+'_monitor':callback})
            callback.assert_called_once()

    def test_tools_refresh_is_available_without_a_selected_session(self):
        callback=MagicMock(return_value='No ended sessions.')
        win=MagicMock();win.getmaxyx.return_value=(30,100)
        win.getch.side_effect=[curses.KEY_F1,curses.KEY_F4,ord('q')]
        with patch.object(monitor_ui,'_terminal_screen',side_effect=lambda f:f(win)),patch.object(monitor_ui,'snapshots',return_value=[]),patch.object(monitor_ui,'dropdown_win'),patch.object(monitor_ui.curses,'curs_set'),patch.object(monitor_ui.curses,'has_colors',return_value=False),patch.object(monitor_ui.curses,'ACS_HLINE',45,create=True):
            monitor_ui.show(Path('/unused'),None,None,refresh_monitor=callback)
        callback.assert_called_once()

    def test_tools_reconnects_selected_session(self):
        reconnect=MagicMock(return_value='Reconnected')
        with patch.object(monitor_ui,'dropdown_win'):
            self.screen([curses.KEY_F1,curses.KEY_F6,ord('q')],reconnect=reconnect)
        reconnect.assert_called_once_with(5)

    def test_log_view_shows_its_full_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Path(tmp);session=store/'5';session.mkdir()
            long_line='x'*100
            log=session/'session.log';log.write_text(long_line+'\n')
            win=MagicMock();win.getmaxyx.return_value=(30,60)
            win.getch.side_effect=[curses.KEY_F5,10,ord('q'),ord('q')]
            row={'id':5,'model':'test','_busy':False,'allocation':{}}
            with patch.object(monitor_ui,'_terminal_screen',side_effect=lambda f:f(win)),patch.object(monitor_ui,'snapshots',return_value=[row]),patch.object(monitor_ui.curses,'curs_set'),patch.object(monitor_ui.curses,'has_colors',return_value=False),patch.object(monitor_ui.curses,'ACS_HLINE',45,create=True):
                monitor_ui.show(store,None,None)
            self.assertTrue(any(str(log) in str(call) for call in win.addnstr.call_args_list))
            self.assertTrue(any('x'*59 in str(call) for call in win.addnstr.call_args_list))
            self.assertTrue(any('x'*41 in str(call) for call in win.addnstr.call_args_list))

    def test_missing_helper_preview_is_explained_without_an_os_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Path(tmp);(store/'5').mkdir()
            win=MagicMock();win.getmaxyx.return_value=(30,60)
            win.getch.side_effect=[curses.KEY_F5,ord('2'),ord('q'),ord('q')]
            row={'id':5,'model':'test','_busy':False,'allocation':{}}
            with patch.object(monitor_ui,'_terminal_screen',side_effect=lambda f:f(win)),patch.object(monitor_ui,'snapshots',return_value=[row]),patch.object(monitor_ui.curses,'curs_set'),patch.object(monitor_ui.curses,'has_colors',return_value=False),patch.object(monitor_ui.curses,'ACS_HLINE',45,create=True):
                monitor_ui.show(store,None,None)
            self.assertTrue(any('No helper log yet' in str(call) for call in win.addnstr.call_args_list))

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

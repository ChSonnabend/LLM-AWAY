from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from llm_away import model_preferences as prefs, resources, webapp
from llm_away.config import AppConfig

class PreferencesTests(unittest.TestCase):
    def test_model_and_host_isolation_and_last_container(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(prefs,'preferences_path',return_value=Path(tmp)/'settings.json'):
            cfg=AppConfig();other=replace(cfg,ssh=replace(cfg.ssh,host='other'))
            prefs.save(cfg,'q4',{'rag':'/project','rag_paths_location':'remote','build_mode':'container','container_path':'/llama.sif'})
            prefs.save(cfg,'q8',{'rag':'/second','build_mode':'native'})
            self.assertEqual(prefs.read(cfg)['q4']['rag'],'/project')
            self.assertEqual(prefs.read(cfg)['q8']['rag'],'/second')
            self.assertEqual(prefs.read(cfg)['_container_path'],'/llama.sif')
            self.assertEqual(prefs.read(other),{})
            prefs.save(cfg,'q4',{'rag':''})
            self.assertEqual(prefs.read(cfg)['q4']['rag'],'')

    def test_hydra_batch_defaults_are_separate_and_replace_old_flags(self):
        cfg=AppConfig();cfg=replace(cfg,ssh=replace(cfg.ssh,host='hydra'),llamacpp=replace(cfg.llamacpp,server_extra_args=['--ubatch-size','2048']))
        options=resources.model_server_options(cfg,'glm-5.3-flash-q4')
        self.assertEqual(options[options.index('--ubatch-size')+1],'512')
        cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,model_host_batch_defaults={'hydra':{'glm-5.3-flash-q4':[2048,512],'glm-5.3-flash-q8':[2048,256]}}))
        self.assertEqual(resources.model_server_options(cfg,'glm-5.3-flash-q8')[-1],'256')
        other=replace(cfg,ssh=replace(cfg.ssh,host='other'))
        self.assertEqual(resources.model_server_options(other,'glm-5.3-flash-q4')[-1],'1024')

    def test_empty_container_is_rejected_before_model_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'session.json').write_text('{}')
            with patch.object(webapp,'session_path',return_value=path),patch.object(resources,'rpc') as rpc:
                with self.assertRaisesRegex(ValueError,'container path'):webapp.load_browser_model(1,{'build_mode':'container','container_path':' '})
                rpc.assert_not_called()

    def test_runtime_selection_is_passed_to_launch_and_saved(self):
        from llm_away import shared_sessions
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'session.json').write_text(json.dumps({'config':asdict(AppConfig())}))
            with patch.object(webapp,'session_path',return_value=path),patch.object(shared_sessions,'ensure_daemon'),patch.object(shared_sessions,'current',return_value={}),patch.object(resources,'run_agent') as run,patch.object(prefs,'save') as save:
                for mode,container in [('container','/new.sif'),('native','')]:
                    webapp.load_browser_model(1,{'model':'q4','build_mode':mode,'container_path':'/new.sif','rag':'/source'})
                    self.assertEqual(run.call_args.args[0].container,container)
                    self.assertEqual(save.call_args.args[2]['rag'],'/source')

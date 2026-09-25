from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from llm_away import models, resources, shared_sessions, webapp
from llm_away.config import AppConfig

class DiscoveryResilience(unittest.TestCase):
    def test_cached_catalogue_avoids_ssh_and_stale_cache_survives_timeout(self):
        cfg=AppConfig();catalogue=[{'name':'glm','alias':'glm','path':'/models/glm.gguf','size_bytes':10}]
        with tempfile.TemporaryDirectory() as tmp,patch.object(models,'catalogue_path',return_value=Path(tmp)/'models.json'):
            models.save_catalogue(cfg,catalogue)
            with patch.object(models.subprocess,'run',side_effect=AssertionError('Cache must avoid SSH')):
                self.assertEqual(models.discover_models(cfg,allow_cached=True),catalogue)
            record=models.cached_models(cfg);record['time']=0;models.catalogue_path(cfg).write_text(json.dumps(record))
            with patch.object(models.subprocess,'run',side_effect=subprocess.TimeoutExpired('ssh',10)) as run:
                self.assertEqual(models.discover_models(cfg,allow_cached=True),catalogue)
                self.assertEqual(run.call_args.kwargs['timeout'],10)
                run.assert_called_once()

    def test_uncached_timeout_has_actionable_error_and_no_raw_traceback(self):
        with patch.object(models,'cached_models',return_value={}),patch.object(models.subprocess,'run',side_effect=subprocess.TimeoutExpired('ssh',10)):
            with self.assertRaisesRegex(ValueError,'allocation is retained'):
                models.discover_models(AppConfig(),allow_cached=True)

    def test_attach_options_survive_catalogue_timeout(self):
        cfg=AppConfig();cfg=replace(cfg,llamacpp=replace(cfg.llamacpp,model_name='glm'),model=replace(cfg.model,name='glm'))
        allocation={'model_state':'LOADED','generation':'g','session':asdict(cfg)}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'session.json').write_text(json.dumps({'config':asdict(cfg)}))
            with patch.object(webapp,'session_path',return_value=path),patch.object(resources,'load_config',return_value=cfg),patch.object(resources,'discover_models',side_effect=ValueError('Discovery timed out')),patch.object(shared_sessions,'current',return_value=allocation),patch('llm_away.model_preferences.read',return_value={}):
                options=webapp.model_options(1)
            self.assertTrue(options['can_attach'])
            self.assertEqual(options['models'][0]['name'],'glm')
            self.assertIn('still be attached',options['discovery_warning'])

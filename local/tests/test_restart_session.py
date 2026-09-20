import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import asdict,replace
from pathlib import Path
from unittest.mock import patch
from llm_away import resources
from llm_away.config import AppConfig

class RestartSessionTests(unittest.TestCase):
    def test_restart_repairs_host_paths_preserves_history_and_session(self):
        for ended in (False,True):
            with self.subTest(ended=ended),tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                path=Path(tmp);cfg=AppConfig()
                cfg=replace(cfg,ssh=replace(cfg.ssh,host='hydra'),remote=replace(cfg.remote,resource_state_dir='/bad'))
                data={'id':25,'token':'old','gpus':2,'phase':'TIMEOUT' if ended else 'ERROR','config':asdict(cfg),'model':'', 'allocation_cleaned':ended}
                (path/'session.json').write_text(json.dumps(data));(path/'session.log').write_text('previous log\n')
                stack.enter_context(patch.object(resources,'path_for',return_value=path))
                stack.enter_context(patch.object(resources,'load_config',return_value=cfg))
                stack.enter_context(patch('llm_away.host_store.read_store',return_value={'hosts':{'hydra':{'backend_type':'slurm_server','remote':{'workdir':'/lustre/project'},'llamacpp':{}}}}))
                spawn=stack.enter_context(patch.object(resources.subprocess,'Popen'))
                self.assertEqual(resources.restart_session(25),25)
                result=json.loads((path/'session.json').read_text())
                self.assertEqual(result['config']['remote']['resource_state_dir'],'')
                self.assertEqual(result['config']['remote']['workdir'],'/lustre/project')
                self.assertEqual(result['token']=='old',not ended)
                self.assertEqual(result['restart_history'][0]['phase'],data['phase'])
                self.assertTrue((path/'session.log').read_text().startswith('previous log'))
                spawn.assert_called_once()

    def test_active_allocation_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'session.json').write_text(json.dumps({'allocation':{'active':True}}))
            with patch.object(resources,'path_for',return_value=path),patch.object(resources.subprocess,'Popen') as spawn:
                with self.assertRaisesRegex(ValueError,'still be active'):resources.restart_session(25)
                spawn.assert_not_called()

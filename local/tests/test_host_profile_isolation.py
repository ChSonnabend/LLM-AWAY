from dataclasses import replace
import unittest
from llm_away.config import AppConfig
from llm_away.host_store import apply_profile

class HostProfileIsolation(unittest.TestCase):
    def test_older_host_does_not_inherit_other_host_paths(self):
        cfg=AppConfig()
        cfg=replace(cfg,remote=replace(cfg.remote,resource_state_dir='/scratch/other/cache'),
                    llamacpp=replace(cfg.llamacpp,models_dir='/scratch/other/models',installation_dir='/scratch/other'))
        profile={'backend_type':'slurm_server','remote':{'workdir':'/lustre/project'},'llamacpp':{'backend':'cuda'}}
        result=apply_profile(cfg,profile)
        self.assertEqual(result.remote.resource_state_dir,'')
        self.assertEqual(result.llamacpp.models_dir,'')
        self.assertEqual(result.llamacpp.installation_dir,'')
        profile['remote']['resource_state_dir']='/lustre/cache'
        profile['llamacpp']['models_dir']='/lustre/models'
        self.assertEqual(apply_profile(cfg,profile).remote.resource_state_dir,'/lustre/cache')
        self.assertEqual(apply_profile(cfg,profile).llamacpp.models_dir,'/lustre/models')

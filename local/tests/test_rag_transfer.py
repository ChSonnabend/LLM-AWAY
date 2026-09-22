import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from llm_away.config import AppConfig
from llm_away.rag_transfer import snapshot


class RagTransferTests(unittest.TestCase):
    def test_shared_paths_are_not_copied(self):
        cfg=AppConfig()
        self.assertEqual(snapshot(cfg,{},'token',['/shared/project'],'shared','remote','/tmp/session'),
                         ['/shared/project'])

    def test_local_paths_snapshot_to_direct_remote_filesystem(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'project';source.mkdir();(source/'a.txt').write_text('one')
            cfg=AppConfig();cfg=replace(cfg,backend_type='direct',ssh=replace(cfg.ssh,connection='local'),
                                        remote=replace(cfg.remote,resource_state_dir=str(root/'state')))
            paths=snapshot(cfg,{},'token',[str(source)],'local','remote',root/'session')
            self.assertEqual(Path(paths[0],'a.txt').read_text(),'one')
            (source/'a.txt').write_text('two')
            paths=snapshot(cfg,{},'token',[str(source)],'local','remote',root/'session')
            self.assertEqual(Path(paths[0],'a.txt').read_text(),'two')

    def test_remote_paths_snapshot_to_local_when_filesystem_is_directly_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'remote.txt';source.write_text('data')
            cfg=replace(AppConfig(),backend_type='direct',ssh=replace(AppConfig().ssh,connection='local'))
            paths=snapshot(cfg,{},'token',[str(source)],'remote','local',root/'session')
            self.assertEqual(Path(paths[0]).read_text(),'data')


if __name__=='__main__':unittest.main()

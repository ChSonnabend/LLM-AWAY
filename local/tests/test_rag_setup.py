import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

from llm_away import loading, rag, resources
from llm_away.config import AppConfig


class RagSetupTests(unittest.TestCase):
    def test_dependency_failure_is_visible_before_builder_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=root/'rag.log'
            with patch.object(rag,'runtime',side_effect=RuntimeError('wheel download failed')), \
                 patch.object(rag.subprocess,'Popen') as child:
                with self.assertRaisesRegex(RuntimeError,'wheel download failed'):
                    rag.prepare([root],status_log=log)
            self.assertIn('RAG: SETUP',log.read_text())
            self.assertIn('RAG: FAILED | setup | RuntimeError: wheel download failed',log.read_text())
            child.assert_not_called()

    def test_missing_root_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'rag.log'
            with self.assertRaises(ValueError):rag.prepare([Path(tmp)/'missing'],status_log=log)
            self.assertIn('RAG: FAILED',log.read_text())

    def test_worker_signal_is_logged_even_without_python_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'rag.log';build=Path(tmp)/'build.log'
            child=Mock();child.wait.return_value=-9
            rag.watch_build(child,log,build)
            self.assertIn('signal 9',log.read_text())
            self.assertIn(str(build),log.read_text())

    def test_main_logs_validation_errors_before_index_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=root/'rag.log'
            with patch.object(sys,'argv',['rag','--index','--roots',json.dumps([str(root/'missing')]),'--status-log',str(log)]):
                with self.assertRaises(ValueError):rag.main()
            self.assertIn('RAG: FAILED | ValueError',log.read_text())

    def test_cuda_runtime_is_isolated_cached_and_recovers_broken_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            def setup(command,log):
                calls.append(command)
                if 'venv' in command:
                    binary=Path(command[-1])/'bin/python'
                    binary.parent.mkdir(parents=True,exist_ok=True);binary.write_text('python')
            with patch.object(rag,'CACHE',root), \
                 patch.object(rag,'bootstrap_python',return_value=('/usr/bin/python3.14','3.14')), \
                 patch.object(rag,'requested_gpu_provider',return_value='CUDAExecutionProvider'), \
                 patch.object(rag,'gpu_provider_available',return_value=True), \
                 patch.object(rag,'setup_command',side_effect=setup), \
                 patch.object(rag.subprocess,'run',return_value=Mock(returncode=0)):
                binary=rag.runtime(True);initial=len(calls)
                self.assertIn('venv-cuda-py3.14',str(binary))
                self.assertFalse(any('--system-site-packages' in c for c in calls))
                self.assertTrue(any(c[-1].endswith('rag-cuda.txt') for c in calls))
                self.assertEqual(rag.runtime(True),binary);self.assertEqual(len(calls),initial)
                binary.unlink()
                self.assertEqual(rag.runtime(True),binary);self.assertGreater(len(calls),initial)

    def test_bootstrap_skips_deleted_interpreters(self):
        with patch.object(rag.shutil,'which',return_value='/missing/old-venv/python'), \
             patch.object(rag.sys,'_base_executable','/missing/base/python'), \
             patch.object(rag.sys,'executable','/missing/old-venv/python'), \
             patch.object(rag.subprocess,'run',return_value=Mock(returncode=0,stdout='3.14\n')):
            python,version=rag.bootstrap_python()
            self.assertEqual(python,str(Path('/usr/bin/python3').resolve()))
            self.assertEqual(version,'3.14')

    def test_cuda_preloads_environment_libraries(self):
        ort=Mock();ort.get_available_providers.return_value=['CUDAExecutionProvider','CPUExecutionProvider']
        with patch.dict(sys.modules,onnxruntime=ort), \
             patch.object(rag,'requested_gpu_provider',return_value='CUDAExecutionProvider'):
            self.assertEqual(rag.embedding_providers(True),['CUDAExecutionProvider','CPUExecutionProvider'])
        ort.preload_dlls.assert_called_once_with(directory='')

    def test_session_start_failure_is_logged_before_rag_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            spec={'loading':{'config':asdict(AppConfig()),'model':{'alias':'test'},'reuse':False,'mtp':'off'},
                  'rag_config':{'paths':[tmp]}}
            with patch.object(resources,'identity',return_value='live'), \
                 patch.object(resources,'rpc',side_effect=FileNotFoundError('removed interpreter')):
                with self.assertRaises(FileNotFoundError):loading.prepare(root,spec)
            text=(root/'rag.log').read_text()
            self.assertIn('RAG: WAITING',text)
            self.assertIn('RAG: FAILED | session startup | FileNotFoundError: removed interpreter',text)
            self.assertFalse((root/'attachment.json').exists())


if __name__=='__main__':unittest.main()

"""Backend selection and UI operations for shared resource sessions."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from llm_away import resources, webapp
from llm_away.config import AppConfig

ROOT=Path(__file__).resolve().parents[2]

class SharedModelRegressions(unittest.TestCase):
    def test_glm_presets_select_native_rocm_and_keep_cuda_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);native=root/'builds/rocm-gfx906/bin/llama-server'
            native.parent.mkdir(parents=True);native.touch();native.chmod(0o755)
            cuda=root/'cuda-server';cuda.touch();cuda.chmod(0o755)
            for quant in ('q4','q8'):
                script='''source "$1/remote/scripts/lib/llamacpp-env.sh"
source "$1/remote/models/glm-5.3-flash-$2/model.env"
unset LLAMA_SERVER
export LLAMACPP_INSTALLATION_DIR="$3" LLAMACPP_ROCM_ARCH=gfx906
llamacpp_binary server rocm
export LLAMACPP_MODEL_CUDA_SERVER="$3/cuda-server"
llamacpp_binary server cuda
'''
                result=subprocess.run(['bash','-c',script,'test',str(ROOT),quant,tmp],text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout.splitlines(),[str(native),str(cuda)])

    def test_remote_loading_model_display_overrides_empty_or_stale_local_model(self):
        allocation={'model_state':'LOADING','session':{'model':{'name':'glm-5.3-flash-q4'}}}
        for local in ('','old-model'):
            self.assertEqual(webapp.display_model({'model':local},allocation),'glm-5.3-flash-q4')
        self.assertEqual(webapp.display_model({'model':''},{'model_state':'IDLE'}),'')

    def test_release_from_web_thread_does_not_wait_on_stale_attachment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'session.json').write_text(json.dumps({'config':asdict(AppConfig()),'token':'test','remote_port':1,'allocation':{'active':True,'host':'node','slurm_state':'RUNNING','model_state':'EXITED'}}))
            (path/'attachment.json').write_text(json.dumps({'client_pid':os.getpid(),'client_identity':'stale'}))
            with patch.object(resources,'path_for',return_value=path),patch.object(resources,'rpc',return_value={'ok':True}) as rpc,patch.object(resources,'identity',side_effect=AssertionError('Do not wait for the wrapper before release')),patch.object(resources,'remote',side_effect=AssertionError('Unexpected remote operation')):
                self.assertEqual(resources.release_session(1),{'ok':True})
                rpc.assert_called_once_with(path,'release')

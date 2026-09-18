import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from llm_away.onboarding import configure_local
from llm_away.remote_models import installed_models

ROOT = Path(__file__).resolve().parents[2]


class SharedPathsTests(unittest.TestCase):
    def test_discovery_weights_draft_and_binary_use_shared_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'own'
            library = root/'scripts/lib/llamacpp-env.sh'
            library.parent.mkdir(parents=True)
            shutil.copyfile(ROOT/'remote/scripts/lib/llamacpp-env.sh', library)
            shared = Path(temp)/'shared models'
            preset = shared/'test';preset.mkdir(parents=True)
            (preset/'model.env').write_text('MODEL_GGUF=models/test/main.gguf\nMODEL_DRAFT_GGUF=models/test/draft.gguf\n')
            for name in ('main.gguf', 'draft.gguf'):
                (preset/name).write_bytes(b'GGUF')
            install = Path(temp)/'shared llama'
            binary = install/'builds/cpu/bin/llama-server'
            binary.parent.mkdir(parents=True);binary.write_text('#!/bin/sh\n');binary.chmod(0o755)
            models = installed_models(str(root),str(shared))
            self.assertEqual(models[0]['path'],str((preset/'main.gguf').resolve()))
            self.assertTrue(models[0]['mtp']['available'])
            env=dict(os.environ,LLAMACPP_INSTALLATION_DIR=str(install))
            for key in ('LLAMA_SERVER','LLAMACPP_BUILD_DIR','LLAMACPP_IN_CONTAINER'):
                env.pop(key,None)
            bash='/opt/homebrew/bin/bash' if Path('/opt/homebrew/bin/bash').exists() else 'bash'
            result=subprocess.check_output([bash,'-c','source "$1"; llamacpp_binary server cpu','test',str(library)],env=env,text=True)
            self.assertEqual(result.strip(),str(binary))

    def test_shared_defaults_do_not_depend_on_ssh_alias(self):
        for shared in ('/scratch/csonnabe/LLM-AWAY/remote','/lustre/alice/users/csonnab/LLM-AWAY/remote'):
            with self.subTest(shared=shared), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                config=Path(temp)/'model.toml';config.write_text('')
                seen={}
                def ask(label, default='', options=None):
                    seen[label]=default
                    answers={'Full path to LLM-AWAY/remote':'/scratch/other/remote',
                             'Container required? yes/no':'no','Submission system':'direct',
                             'GPU backend':'cpu','GPUs per job (0 for CPU)':'0'}
                    return answers.get(label,default)
                stack.enter_context(patch('llm_away.onboarding.ask',side_effect=ask))
                stack.enter_context(patch('llm_away.onboarding.ssh_hosts',return_value=(['my-cluster'],[])))
                stack.enter_context(patch('llm_away.onboarding.remote_json',return_value={'tools':{},'partitions':[],'shared_root':shared}))
                configure_local(config,alias='my-cluster',connection='ssh',resources_only=True)
                self.assertEqual(seen['Full path to LLM-AWAY/remote'],'')
                self.assertEqual(seen['Full path to models directory (presets and GGUF files)'],shared+'/models')
                self.assertEqual(seen['Full path to llama.cpp installation (contains builds/ or build/)'],shared)

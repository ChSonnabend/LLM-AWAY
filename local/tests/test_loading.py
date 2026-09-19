import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch, MagicMock

from llm_away import resources, loading
from llm_away.config import AppConfig

class LoadingTests(unittest.TestCase):
    def test_run_hands_off_before_waiting_and_remote_allocation_has_no_native_prompt(self):
        for location in ('local','remote'):
            for detached in (False,True):
                with self.subTest(location=location,detached=detached), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                    path=Path(tmp);cfg=AppConfig();cfg=replace(cfg,ssh=replace(cfg.ssh,connection='ssh'))
                    data={'config':asdict(cfg),'model':'','token':'test','remote_port':123}
                    (path/'session.json').write_text(json.dumps(data))
                    args=Namespace(session=1,helper=False,detach=detached,model=None,agent_location=location,cli='codex',agent_workdir=None,agent_args=[],rag=None,mtp=None)
                    for name,value in [('path_for',path),('rpc',data),('config',cfg),('load_config',cfg),('discover_models',[]),('choose_model',{'alias':'test','name':'test','context_size':750000}),('choose_mtp','off'),('choose_cli','codex'),('ask','')]:
                        stack.enter_context(patch.object(resources,name,return_value=value))
                    stack.enter_context(patch.object(resources.sys.stdin,'isatty',return_value=True))
                    menu=stack.enter_context(patch.object(resources,'choose_option',side_effect=AssertionError('Unexpected native CLI prompt')))
                    health=stack.enter_context(patch.object(resources,'urlopen'))
                    stack.enter_context(patch('llm_away.terminals.engine',return_value={'tmux':MagicMock()}))
                    ensure=stack.enter_context(patch('llm_away.terminals.ensure'))
                    attach=stack.enter_context(patch('llm_away.terminals.attach'))
                    resources.run_agent(args)
                    health.assert_not_called();menu.assert_not_called()
                    self.assertEqual(attach.call_count,0 if detached else 1)
                    saved=ensure.call_args.args[2]
                    self.assertEqual(saved['loading']['model']['alias'],'test')
                    self.assertEqual(saved['target_location'],location)
                    self.assertEqual(saved['location'],'local')
                    self.assertEqual(saved['context_window'],cfg.codex.context_window)
                    self.assertEqual(saved['auto_compact_token_limit'],int(cfg.codex.context_window*0.7))

    def test_loading_then_ready_sets_agent_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);cfg=AppConfig();spec={'loading':{'config':asdict(cfg),'model':{'alias':'test','name':'test'},'reuse':False,'mtp':'off'}}
            response=MagicMock();response.__enter__.return_value.status=200
            with patch.object(loading,'urlopen',side_effect=[OSError(),response]),patch.object(loading.time,'sleep'),patch.object(resources,'identity',return_value='live'),patch.object(resources,'rpc',return_value={'allocation':{'active':True,'model_state':'LOADING'}}) as rpc:
                self.assertFalse(loading.prepare(path,spec))
            self.assertEqual(rpc.call_args_list[0].args,(path,'start'))
            self.assertEqual(spec['model'],'test')
            self.assertFalse((path/'attachment.json').exists())

    def test_failed_backend_cleans_loading_attachment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);spec={'loading':{'config':asdict(AppConfig()),'model':{'alias':'test'},'reuse':True,'mtp':'off'}}
            with patch.object(loading,'urlopen',side_effect=OSError()),patch.object(resources,'identity',return_value='live'),patch.object(resources,'rpc',return_value={'provider_exit':1}) as rpc:
                with self.assertRaisesRegex(RuntimeError,'Backend stopped'):loading.prepare(path,spec)
            self.assertFalse((path/'attachment.json').exists())
            self.assertEqual(rpc.call_args.args,(path,'status'))

    def test_remote_cli_opens_only_after_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);cfg=AppConfig();data={'config':asdict(cfg),'token':'test','remote_port':1,'model':'test'}
            (path/'session.json').write_text(json.dumps(data))
            spec={'loading':{'config':asdict(cfg),'model':{'alias':'test'},'reuse':True,'mtp':'off'},'target_location':'remote','cli':'auto','location':'local','cwd':''}
            (path/'agent-selection.json').write_text(json.dumps(spec))
            response=MagicMock();response.__enter__.return_value.status=200
            with patch.object(loading,'urlopen',return_value=response),patch.object(resources,'identity',return_value='live'),patch.object(resources,'remote',return_value={'clis':['codex']}),patch('llm_away.agents.choose_cli',return_value='codex'),patch('llm_away.terminals.ensure') as ensure,patch('llm_away.terminals.attach') as attach:
                self.assertTrue(loading.prepare(path,spec))
            self.assertEqual(ensure.call_args.args[2]['location'],'remote')
            self.assertNotIn('loading',ensure.call_args.args[2])
            attach.assert_called_once()

    def test_reopen_preserves_picker_for_kept_and_changed_models(self):
        for change,detached in ((False,False),(True,False),(False,True)):
            with self.subTest(change=change,detached=detached), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                path=Path(tmp);cfg=AppConfig()
                data={'config':asdict(cfg),'model':'loaded','token':'test','remote_port':123,
                      'provider_pid':123,'provider_identity':'alive'}
                (path/'session.json').write_text(json.dumps(data))
                args=Namespace(session=1,helper=False,detach=detached,resume=not detached,
                               model=None,agent_location='local',cli='codex',agent_workdir=tmp,
                               agent_args=[],rag=[],mtp=None,quiet=detached)
                for name,value in [('path_for',path),('rpc',data),('config',cfg),('load_config',cfg),
                                   ('identity','alive'),('choose_option',int(change)),('discover_models',[]),
                                   ('choose_model',{'name':'new','alias':'new','context_size':32768}),
                                   ('choose_cli','codex'),('choose_mtp','auto'),('ask','')]:
                    stack.enter_context(patch.object(resources,name,return_value=value))
                stack.enter_context(patch('llm_away.terminals.engine',return_value={'tmux':MagicMock()}))
                ensure=stack.enter_context(patch('llm_away.terminals.ensure'))
                stack.enter_context(patch('llm_away.terminals.attach'))
                resources.run_agent(args)
                self.assertEqual(ensure.call_args.args[2]['resume'],not detached)
                self.assertEqual(ensure.call_args.args[2]['loading']['reuse'],not change)

    def test_terminal_launch_uses_conversation_picker_only_when_requested(self):
        import runpy
        engine=runpy.run_path(str(Path(__file__).resolve().parents[2]/'remote/bin/resource-terminal'))
        for cli,option in [('codex','resume'),('claude','--resume')]:
            command=[cli,'--model','test']
            self.assertEqual(engine['resume_picker'](command.copy(),{'cli':cli,'resume':True}),
                             [cli,option,'--model','test'])
            self.assertEqual(engine['resume_picker'](command.copy(),{'cli':cli,'resume':False}),command)

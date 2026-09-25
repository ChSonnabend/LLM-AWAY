import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from dataclasses import replace
from llm_away.config import AppConfig
from llm_away import terminals
ROOT=Path(__file__).resolve().parents[2]
E=runpy.run_path(str(ROOT/'remote/bin/resource-terminal'))

class TerminalTests(unittest.TestCase):
    def setUp(self):
        # Real tmux tests must never change the user's server or key bindings.
        self.socket_dir=tempfile.TemporaryDirectory(prefix='away-tmux-')
        self.env=patch.dict(os.environ,{'TMUX_TMPDIR':self.socket_dir.name})
        self.env.start()
        self.addCleanup(self.socket_dir.cleanup)
        self.addCleanup(self.env.stop)
        self.addCleanup(lambda: subprocess.run(['tmux','-L','default','kill-server'],
                        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))

    def test_legacy_agent_stays_reachable_until_it_exits(self):
        with tempfile.TemporaryDirectory() as temp:
            state=Path(temp)
            (state/'terminal.json').write_text(json.dumps({'token':'legacy-test'}))
            legacy=['tmux','-L','llm-away-legacy-test']
            subprocess.check_call(legacy+['-f','/dev/null','new-session','-d','-s','away-legacy-test','sleep 60'])
            try:
                self.assertTrue(E['alive'](state))
                self.assertEqual(E['tmux'](state)[-1],'llm-away-legacy-test')
                E['stop'](state)
                self.assertFalse(E['alive'](state))
                self.assertEqual(E['tmux'](state)[-1],'default')
            finally:
                subprocess.run(legacy+['kill-server'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    def test_attach_restores_cursor_even_on_failure(self):
        import io
        output=io.StringIO()
        output.isatty=lambda:True
        globals_=E['attach'].__globals__
        with patch.dict(globals_,{'configure':lambda state:None,'tmux':lambda state:['tmux'],
                                 'name':lambda state:'away-test'}), \
             patch.object(E['sys'],'stdout',output), \
             patch.object(E['subprocess'],'call',side_effect=OSError('attach failed')):
            with self.assertRaises(OSError):E['attach'](Path('/unused'))
        self.assertEqual(output.getvalue(),'\x1b[0 q\x1b[?25h'*2)

    def test_remote_agent_rag_client_points_at_local_bridge(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=replace(AppConfig(),ssh=replace(AppConfig().ssh,connection='ssh',host='cluster'))
            data={'token':'token123'}
            with patch.object(terminals.subprocess,'Popen') as popen:
                popen.return_value.pid=1234
                command=terminals.local_rag_for_remote_agent(
                    Path(temp),data,cfg,{'job_id':'42'},['python','rag.py'])
            self.assertEqual(command[1],'rag-socket-client')
            self.assertIn('token123',command[2])
            self.assertEqual(json.loads((Path(temp)/'rag-bridge.json').read_text())['pid'],1234)

    def test_local_agent_rag_command_enters_slurm_allocation(self):
        cfg=replace(AppConfig(),ssh=replace(AppConfig().ssh,connection='ssh',host='cluster'))
        command=terminals.remote_rag_command(cfg,{'job_id':'42'},{'paths':['/remote/project']},'token123')
        self.assertEqual(command[:6],['ssh','-x','-o','BatchMode=yes','-o','ConnectTimeout=60'])
        self.assertEqual(command[6],'cluster')
        self.assertIn('srun --jobid=42 --overlap',command[7])
        self.assertIn('rag-stdio',command[7])
        self.assertIn('token123',command[7])

    def test_remote_rag_is_prepared_on_the_agent_host(self):
        config={'paths':['/remote/project','/remote/docs/readme.md'],'threads':6,
                'memory_gb':10,'gpu':True}
        with patch('llm_away.rag.prepare',return_value=['python','rag.py']) as prepare:
            self.assertEqual(E['prepare_rag']({'rag_config':config}),['python','rag.py'])
        prepare.assert_called_once_with(config['paths'],6,10,True,None)

    def test_remote_rag_refuses_the_slurm_login_node(self):
        spec={'allocation_kind':'slurm_server','rag_config':{'paths':['/remote/project']}}
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'outside its Slurm allocation'):
                E['prepare_rag'](spec)

    def test_detached_agent_accepts_space_prompt_and_remains_attachable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cli=root/'codex'
            cli.write_text('#!/usr/bin/env python3\nimport sys\nprint("READY",flush=True)\nfor line in sys.stdin: print("RECEIVED:"+line.strip(),flush=True)\n');cli.chmod(0o755)
            import uuid
            spec={'token':uuid.uuid4().hex,'cli':'codex','model':'test','cwd':str(root),'base_url':'http://localhost:1'}
            with patch.dict(os.environ,{'PATH':str(root)+os.pathsep+os.environ['PATH']}):
                E['start'](root,spec)
                try:
                    for _ in range(100):
                        if 'READY' in E['capture'](root)['text']:break
                        time.sleep(.05)
                    E['send'](root,'hello from Space')
                    for _ in range(100):
                        output=E['capture'](root)['text']
                        if 'RECEIVED:hello from Space' in output:break
                        time.sleep(.05)
                    self.assertIn('RECEIVED:hello from Space',output)
                    self.assertTrue(E['alive'](root))
                    env=dict(os.environ);env.pop('TMUX',None)
                    sessions=subprocess.check_output(['tmux','ls','-F','#{session_name}'],env=env,text=True)
                    self.assertIn(E['name'](root),sessions.splitlines())
                    style=subprocess.check_output(E['tmux'](root)+['show-options','-wv','-t',E['name'](root),'cursor-style'],text=True)
                    self.assertEqual(style.strip(),'block')
                    overrides=subprocess.check_output(E['tmux'](root)+['show-options','-sv','terminal-overrides'],text=True)
                    E['configure'](root)
                    self.assertEqual(overrides,subprocess.check_output(E['tmux'](root)+['show-options','-sv','terminal-overrides'],text=True))
                    # Starting again attaches to the same existing conversation.
                    E['start'](root,spec)
                    self.assertIn('RECEIVED:hello from Space',E['capture'](root)['text'])
                finally:E['stop'](root)
                self.assertFalse(E['alive'](root))

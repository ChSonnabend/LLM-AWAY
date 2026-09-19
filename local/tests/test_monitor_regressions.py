import json
import os
import pty
import select
import time
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from llm_away import monitor_ui, resources, helper_relations
from llm_away.config import AppConfig

class MonitorRegressions(unittest.TestCase):
    def test_curses_restores_terminal_across_repeated_screens_and_errors(self):
        master,slave=pty.openpty()
        code='''
import curses, termios
from llm_away.monitor_ui import _terminal_screen, _form_screen, mtp_select_win, terminal_operation
before=termios.tcgetattr(0)
for fail in (False,True,False):
 def run(win):
  win.addstr(0,0,'monitor');win.refresh();curses.nonl()
  if fail:raise ValueError('exit')
  curses.ungetch(10);curses.ungetch(curses.KEY_DOWN)
  assert _form_screen('Allocation',[('text','GPUs',None,'1')],['1'])==['1']
  model={'name':'test','size_bytes':1,'mtp':{}}
  curses.ungetch(10)
  assert mtp_select_win([model],1)==(model,'auto')
  def operation():
   assert termios.tcgetattr(0)[1]==before[1]
   print('HIDDEN BACKGROUND OUTPUT')
  terminal_operation(operation)
  assert termios.tcgetattr(0)[3] & termios.ICANON == 0
  win.erase();win.addstr(0,0,'monitor again');win.refresh()
 try:_terminal_screen(run)
 except ValueError:pass
 after=termios.tcgetattr(0)
 after[3] &= ~getattr(termios,'PENDIN',0)
 before[3] &= ~getattr(termios,'PENDIN',0)
 assert after==before,(before,after)
print('FIRST\\nSECOND',flush=True)
'''
        try:
            proc=subprocess.Popen([sys.executable,'-c',code],stdin=slave,stdout=slave,stderr=slave,
                                  env=dict(os.environ,TERM='xterm',PYTHONPATH='src'))
            output=b'';deadline=time.monotonic()+5
            while proc.poll() is None:
                if time.monotonic()>deadline:
                    proc.kill();proc.wait();self.fail(repr(output))
                if select.select([master],[],[],.05)[0]:output+=os.read(master,65536)
            while select.select([master],[],[],.05)[0]:output+=os.read(master,65536)
            self.assertEqual(proc.returncode,0,output)
            self.assertIn(b'FIRST\r\nSECOND\r\n',output)
            self.assertNotIn(b'HIDDEN BACKGROUND OUTPUT',output)
            self.assertEqual(output.count(b'\x1b[?1049l'),3)
        finally:os.close(master);os.close(slave)

    def test_mtp_uses_top_level_screen(self):
        model={'name':'test','mtp':{'available':True,'toggle_supported':True}}
        with patch.object(monitor_ui,'model_select_win',return_value=model), patch.object(monitor_ui,'dropdown_win',return_value='MTP off'), patch.object(monitor_ui,'_sub_screen',side_effect=AssertionError('inactive curses')):
            self.assertEqual(monitor_ui.mtp_select_win([model],1),(model,'off'))

    def test_empty_host_choices(self):
        self.assertIsNone(monitor_ui.dropdown_win('SSH host',[]))

    def test_local_wizard(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(resources,'STORE',Path(tmp)), patch.object(resources,'load_config',return_value=AppConfig()), patch.object(resources,'released_ids',return_value=[]), patch.object(resources,'free_port',side_effect=[12340,12341,12342]), patch.object(resources.subprocess,'Popen'):
            number=resources.allocate_from_wizard({'connection':'local','gpus':'1'})
            data=json.loads((Path(tmp)/str(number)/'session.json').read_text())
            self.assertEqual(data['config']['ssh']['connection'],'local')

    def test_refresh_is_noninteractive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'session.json').write_text('{}')
            (path/'agent-selection.json').write_text(json.dumps({'location':'local','cli':'claude','cwd':tmp}))
            with patch.object(resources,'path_for',return_value=path), patch('llm_away.terminals.engine',return_value={'alive':lambda p:False}), patch.object(resources,'run_agent') as run:
                resources.refresh_session(1)
                args=run.call_args.args[0]
                self.assertTrue(args.quiet)
                self.assertTrue(args.detach)
                self.assertEqual(args.cli,'claude')
                self.assertEqual(args.agent_workdir,tmp)

    def test_helper_rejects_self_accepts_others(self):
        with patch.object(helper_relations,'master_session',return_value={'id':1}):
            with self.assertRaisesRegex(ValueError,'own helper'):
                helper_relations.ensure_other_session(Path('/sessions/1'))
            helper_relations.ensure_other_session(Path('/sessions/2'))

    def test_tools_footer_back_refresh_and_helper_keys(self):
        import curses
        for keys,action in [([ord('1'),ord('q'),ord('q')],None),
                            ([curses.KEY_F1,curses.KEY_F1,ord('q')],'refresh'),
                            ([ord('1'),curses.KEY_F2,ord('q')],'helper')]:
            win=MagicMock();win.getmaxyx.return_value=(24,100);win.getch.side_effect=keys
            refresh=MagicMock();helper=MagicMock()
            with patch.object(monitor_ui,'_terminal_screen',side_effect=lambda f:f(win)), patch.object(monitor_ui,'snapshots',return_value=[{'id':1,'_busy':False}]), patch.object(monitor_ui.curses,'curs_set'), patch.object(monitor_ui.curses,'has_colors',return_value=False), patch.object(monitor_ui.curses,'ACS_HLINE',ord('-'),create=True), patch.object(monitor_ui,'dropdown_win'):
                monitor_ui.show(Path('/unused'),None,None,refresh=refresh,set_helper=helper)
            self.assertEqual(refresh.call_count,int(action=='refresh'))
            self.assertEqual(helper.call_count,int(action=='helper'))
            self.assertTrue(any('2/F2 Set helper' in str(c) for c in win.addnstr.call_args_list))

    def test_agent_command_applies_context_and_compaction_budget(self):
        import runpy
        engine=runpy.run_path(str(Path(__file__).resolve().parents[2]/'remote/bin/resource-agent'))
        with patch('shutil.which',return_value='/bin/codex'):
            command,_=engine['agent_command']({'cli':'codex','model':'glm','context_window':128000,'auto_compact_token_limit':89600},'http://localhost:1')
        self.assertIn('model_context_window=128000',command)
        self.assertIn('model_auto_compact_token_limit=89600',command)

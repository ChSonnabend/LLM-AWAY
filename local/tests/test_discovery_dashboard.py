from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from llm_away import shared_sessions as shared, resources, webapp
from llm_away.config import AppConfig, load_config


class DiscoveryTests(unittest.TestCase):
    def test_ssh_aliases_without_saved_profiles_are_included(self):
        cfg=AppConfig();cfg=replace(cfg,ssh=replace(cfg.ssh,host='saved',user='saved-user'),remote=replace(cfg.remote,resource_state_dir='/private/saved'))
        with patch('llm_away.onboarding.ssh_hosts',return_value=(['saved','fresh','gateway'],['*.example'])):
            profiles=shared.discovery_profiles(cfg)
        self.assertEqual([p.ssh.host for p in profiles],['saved','fresh','gateway'])
        self.assertEqual(profiles[0].remote.resource_state_dir,'/private/saved')
        self.assertEqual(profiles[1].remote.resource_state_dir,'')
        self.assertEqual(profiles[1].ssh.user,'')

    def test_framework_probe_uses_remote_home_and_skips_missing_installs(self):
        with patch.object(shared.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'"/home/test/LLM-AWAY/remote"','')) as run:
            found=shared.probe_framework(AppConfig())
            self.assertEqual(found.remote.workdir,'/home/test/LLM-AWAY/remote')
            self.assertIn('~/LLM-AWAY/remote',json.loads(run.call_args.kwargs['input']))
            self.assertIn('BatchMode=yes',run.call_args.args[0])
            run.return_value.stdout='null'
            self.assertIsNone(shared.probe_framework(AppConfig()))

    def test_discovers_all_framework_hosts_without_an_allocation(self):
        cfg=AppConfig();profiles=[replace(cfg,ssh=replace(cfg.ssh,host=name)) for name in ('one','two','gateway')]
        def remote(profile,token,port,action):
            self.assertEqual(action,'list');return {'sessions':[{'token':profile.ssh.host}]}
        with patch.object(resources,'load_config',return_value=cfg),patch.object(shared,'discovery_profiles',return_value=profiles),patch.object(shared,'probe_framework',side_effect=lambda p:None if p.ssh.host=='gateway' else p),patch.object(resources,'remote',side_effect=remote),patch.object(shared,'import_session',return_value=1) as imported:
            result=webapp.run_action('discover-remote')
        self.assertEqual(imported.call_count,2)
        self.assertIn('2 additional remote allocations',result)
        self.assertIn('3 hosts checked',result)
        self.assertNotIn('Unavailable',result)

    def test_unreachable_host_does_not_hide_other_jobs(self):
        cfg=AppConfig();other=replace(cfg,ssh=replace(cfg.ssh,host='offline'))
        def probe(profile):
            if profile.ssh.host=='offline':raise RuntimeError('SSH unavailable')
            return profile
        with patch.object(resources,'load_config',return_value=cfg),patch.object(shared,'discovery_profiles',return_value=[cfg,other]),patch.object(shared,'probe_framework',side_effect=probe),patch.object(resources,'remote',return_value={'sessions':[{'token':'test'}]}),patch.object(shared,'import_session',return_value=1):
            result=shared.discover()
        self.assertIn('1 additional remote allocations',result)
        self.assertIn('offline: SSH unavailable',result)


class DashboardUpgradeTests(unittest.TestCase):
    def test_current_config_accepts_host_model_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.toml';path.write_text('[llamacpp.model_host_batch_defaults.hydra]\n"glm-5.3-flash-q4" = [2048,512]\n')
            self.assertEqual(load_config(path).llamacpp.model_host_batch_defaults['hydra']['glm-5.3-flash-q4'],[2048,512])

    def test_restart_selects_only_owned_dashboard_on_exact_port(self):
        uid=os.getuid()
        processes=f'''100 {uid} /repo/.venv/bin/python3 -m llm_away.webapp --serve --port 8766
101 {uid} /repo/.venv/bin/python3 -m llm_away.resources daemon /repo/run/1
102 {uid} /repo/.venv/bin/python3 -m llm_away.webapp --serve --port 8767
103 {uid+1} python3 -m llm_away.webapp --serve --port 8766
104 {uid} ssh host python3 -m llm_away.webapp --serve --port 8766
105 {uid} python3 -m llm_away.webapp --restart --port 8766
'''
        self.assertEqual(webapp.dashboard_processes(processes,8766),[100])
        with patch.object(webapp.subprocess,'check_output',return_value=processes),patch.object(webapp.os,'kill') as kill,patch.object(webapp,'dashboard_running',return_value=False):webapp.stop_dashboard(8766)
        kill.assert_called_once_with(100,webapp.signal.SIGTERM)

    def test_opening_dashboard_restarts_outdated_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(webapp,'__file__',str(Path(tmp)/'src/llm_away/webapp.py')),patch.object(webapp.sys,'argv',['res-mon-web']),patch.object(webapp,'dashboard_running',side_effect=[True,True]),patch.object(webapp,'dashboard_current',return_value=False),patch.object(webapp,'stop_dashboard') as stop,patch.object(webapp.subprocess,'Popen') as start,patch.object(webapp,'open_dashboard') as show:
                webapp.main();stop.assert_called_once_with(8766)
                self.assertIn('--serve',start.call_args.args[0]);show.assert_called_once_with(8766)

    def test_runtime_endpoint_reports_loaded_revision(self):
        handler=object.__new__(webapp.DashboardHandler);handler.path='/api/runtime'
        with patch.object(handler,'_json') as send,patch.object(webapp,'source_revision',return_value='new-files'):handler.do_GET()
        self.assertEqual(send.call_args.args[0],{'revision':webapp.RUNTIME_REVISION})

    def test_legacy_backend_is_outdated(self):
        with patch.object(webapp,'urlopen',side_effect=OSError('404')):self.assertFalse(webapp.dashboard_current(8766))

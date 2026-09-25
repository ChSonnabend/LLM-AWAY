"""Remember launch defaults by remote host, repository, and model."""
import fcntl
import json
import os
from pathlib import Path


def preferences_path():
    from .resources import ROOT
    return Path(os.environ.get('LLM_REMOTE_CONFIG',str(ROOT/'config/model.toml'))).with_suffix('.preferences.json')


def host_key(cfg):
    return json.dumps([cfg.ssh.connection,cfg.ssh.destination,cfg.remote.workdir])


def read(cfg):
    path=preferences_path()
    if not path.exists():return {}
    return json.loads(path.read_text()).get(host_key(cfg),{})


def save(cfg,model,settings):
    path=preferences_path();path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(path.read_text()) if path.exists() else {}
        host=data.setdefault(host_key(cfg),{})
        entry=host.setdefault(model,{})
        for key in ('rag','rag_compute','rag_paths_location'):
            if key in settings:entry[key]=settings[key]
        if settings.get('build_mode') in ('native','container'):
            entry['build_mode']=settings['build_mode']
        if settings.get('container_path','').strip():host['_container_path']=settings['container_path'].strip()
        from .resources import write
        write(path,data)

"""Remote allocation discovery and explicit ownership transfer between clients."""
from dataclasses import asdict, replace
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid


def client_identity():
    from . import resources as r
    r.STORE.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (r.STORE/'.client-id').open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.seek(0);identifier=stream.read().strip()
        if not identifier:
            identifier=uuid.uuid4().hex
            stream.write(identifier);stream.flush()
    return identifier


def discover():
    from . import resources as r
    cfg=r.load_config(os.environ.get('LLM_REMOTE_CONFIG',str(r.ROOT/'config/model.toml')))
    profiles=[cfg.with_host(host.name) for host in cfg.host_configs()]
    imported=0;errors=[];seen=set()
    for profile in profiles:
        key=(profile.ssh.destination,profile.remote.workdir,profile.remote.resource_state_dir)
        if key in seen:continue
        seen.add(key)
        try:
            result=r.remote(profile,'0'*32,0,'list')
            for item in result['sessions']:
                imported+=import_session(profile,item)
        except Exception as exc:errors.append(f'{profile.ssh.destination}: {exc}')
    message=f'Discovered {imported} additional remote allocations.'
    if errors:message+=' '+ '; '.join(errors)
    return message


def import_session(profile,item):
    from . import resources as r
    r.STORE.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (r.STORE/'registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for saved in r.STORE.glob('[0-9]*/session.json'):
            old=json.loads(saved.read_text())
            if old.get('token')==item['token']:return 0
        number=1+max([int(p.name) for p in r.STORE.iterdir() if p.name.isdigit()]+[0])
        path=r.STORE/str(number);path.mkdir(mode=0o700)
        data=asdict(profile);data.update(item['session']);cfg=r.config(data)
        provider_port=r.free_port();tunnel_port=r.free_port()
        while tunnel_port==provider_port:tunnel_port=r.free_port()
        cfg=replace(cfg,ssh=profile.ssh,server=replace(cfg.server,host='127.0.0.1',port=provider_port),
                    gateway=replace(cfg.gateway,server_port=item['server_port'],local_port=tunnel_port,
                                    cancel_on_exit=False,cancel_reused_on_exit=False),hosts={},saved_hosts={},active_host='')
        r.write(path/'session.json',dict(id=number,token=item['token'],remote_port=item['server_port'],
            config=asdict(cfg),host=profile.ssh.destination,gpus=cfg.slurm.gpus,phase=item['slurm_state'],
            allocation=item,model=cfg.model.name if item.get('model_state') in ('STARTING','LOADING','LOADED','READY') else '',imported=True))
    ensure_daemon(path)
    return 1


def ensure_daemon(path):
    from . import resources as r
    if r.rpc_alive(path):return
    with (path/'session.log').open('a') as log:
        subprocess.Popen([sys.executable,'-m','llm_away.resources','daemon',str(path)],
                         stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    deadline=time.monotonic()+10
    while not r.rpc_alive(path):
        if time.monotonic()>deadline:raise RuntimeError('Session monitor is starting; retry shortly')
        time.sleep(.1)


def current(path):
    from . import resources as r
    data=json.loads((path/'session.json').read_text())
    return r.remote(r.config(data['config']),data['token'],data['remote_port'],'status')


def claim(path,expected_owner,expected_generation):
    from . import resources as r
    data=json.loads((path/'session.json').read_text());cfg=r.config(data['config'])
    return r.remote(cfg,data['token'],data['remote_port'],'claim',
                    expected_owner=expected_owner,expected_generation=expected_generation)

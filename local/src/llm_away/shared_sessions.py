"""Remote allocation discovery and explicit ownership transfer between clients."""
from dataclasses import asdict, replace
import fcntl
import json
import os
import shlex
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


def discovery_profiles(cfg):
    """Only hosts explicitly present in LLM-AWAY settings; never scan SSH aliases."""
    names=list(dict.fromkeys([name for name,profile in cfg.saved_hosts.items()
                              if not profile.get('_builtin')]+list(cfg.hosts)))
    # Do not also probe a bundled example/default host when saved profiles exist.
    return [cfg.with_host(name) for name in names] if names else [cfg]


def probe_framework(profile):
    """Only inspect conventional/configured paths; never install on a probed host."""
    candidates=list(dict.fromkeys([profile.remote.workdir,'~/LLM-AWAY/remote','~/remote']))
    script="""import json,os,sys
from pathlib import Path
for candidate in json.load(sys.stdin):
    if not candidate:continue
    root=Path(os.path.expandvars(candidate)).expanduser()
    if (root/'bin/resource-control').is_file():
        print(json.dumps(str(root.resolve())));break
else:print('null')
"""
    command=['python3','-c',script]
    if profile.ssh.connection!='local':
        command=['ssh','-x','-o','BatchMode=yes','-o','ConnectTimeout=10',
                 profile.ssh.destination,shlex.join(command)]
    result=subprocess.run(command,input=json.dumps(candidates),text=True,capture_output=True,timeout=20)
    if result.returncode:raise RuntimeError(result.stderr.strip() or 'Framework probe failed')
    root=json.loads(result.stdout)
    if not root:return None
    return replace(profile,remote=replace(profile.remote,workdir=root))


def direct_profile(profile,item,profiles=None):
    """A worker hostname is display metadata, not a client-side SSH address."""
    worker=item.get('worker_host','').split('.')[0].lower()
    if item.get('session',{}).get('backend_type')!='direct' or not worker:return profile
    if profile.ssh.connection=='local' or profile.ssh.host.lower()==worker:return profile
    if profiles is None:
        from . import resources as r
        profiles=discovery_profiles(r.load_config(os.environ.get('LLM_REMOTE_CONFIG',str(r.ROOT/'config/model.toml'))))
    for candidate in profiles:
        if candidate.ssh.host.split('.')[0].lower()==worker:
            return replace(profile,ssh=candidate.ssh)
    raise ValueError('Configure an SSH host for direct worker '+item['worker_host']+'; discovery will not use its internal hostname as an SSH address')


def reconcile(scans,active_tokens):
    """Hide ended allocations only after a successful scan and explicit status check."""
    from . import resources as r
    removed=0
    for saved in r.STORE.glob('[0-9]*/session.json'):
        try:data=json.loads(saved.read_text())
        except (OSError,ValueError):continue
        path=saved.parent
        if data.get('native') or data.get('token') in active_tokens or (path/'discovery-retired').exists():continue
        try:cfg=r.config(data['config'])
        except (KeyError,ValueError):continue
        for profile in scans:
            same_store=(cfg.remote.workdir==profile.remote.workdir and cfg.remote.resource_state_dir==profile.remote.resource_state_dir)
            if not same_store:continue
            # Aliases differ across clients. Verify through this session's own
            # configured connection, not the host used by the inventory scan.
            try:status=r.remote(cfg,data['token'],data['remote_port'],'status')
            except Exception:continue # SSH failures are not proof that an allocation ended.
            if status.get('active') is False:
                r.write(path/'discovery-retired',{'allocation':status})
                removed+=1
            break
    return removed


def discover():
    from . import resources as r
    from concurrent.futures import ThreadPoolExecutor
    cfg=r.load_config(os.environ.get('LLM_REMOTE_CONFIG',str(r.ROOT/'config/model.toml')))
    profiles=[];seen=set()
    for profile in discovery_profiles(cfg):
        key=(profile.ssh.connection,profile.ssh.destination,profile.remote.workdir,profile.remote.resource_state_dir)
        if key not in seen:profiles.append(profile);seen.add(key)
    def scan(profile):
        try:
            found=probe_framework(profile)
            if found is None:return profile,[],None,False
            result=r.remote(found,'0'*32,0,'list')
            return found,result['sessions'],None,True
        except Exception as exc:
            detail=str(exc)
            if 'Unknown action' in detail:
                detail='Update remote/bin/resource-control on this host to enable discovery'
            return profile,[],detail,False
    imported=0;errors=[];frameworks=0;checked=[];scanned=[];active_tokens=set()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for profile,items,error,found in pool.map(scan,profiles):
            checked.append(profile.ssh.destination)
            frameworks+=int(found)
            if error:errors.append(f'{profile.ssh.destination}: {error}');continue
            if found:scanned.append(profile)
            active_tokens.update(item['token'] for item in items)
            for item in items:
                try:imported+=import_session(profile,item)
                except Exception as exc:errors.append(f'{profile.ssh.destination}: {exc}')
    removed=reconcile(scanned,active_tokens)
    message=f'Discovered {imported} additional remote allocations across {frameworks} framework hosts ({len(profiles)} hosts checked).'
    message+=f' Removed {removed} ended allocations from monitoring.'
    message+=' Checked hosts: '+(', '.join(checked) or 'none')+'.'
    if errors:message+=' Unavailable hosts: '+ '; '.join(errors)
    return message


def import_session(profile,item):
    from . import resources as r
    profile=direct_profile(profile,item)
    r.STORE.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (r.STORE/'registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for saved in r.STORE.glob('[0-9]*/session.json'):
            old=json.loads(saved.read_text())
            if old.get('token')==item['token']:
                (saved.parent/'discovery-retired').unlink(missing_ok=True)
                if item.get('worker_host') and item.get('session',{}).get('backend_type')=='direct':
                    old['config']['ssh']=asdict(profile.ssh);old['host']=profile.ssh.destination
                    old['allocation']=item;r.write(saved,old)
                return 0
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


def attach(path,expected_generation):
    from . import resources as r
    data=json.loads((path/'session.json').read_text())
    return r.remote(r.config(data['config']),data['token'],data['remote_port'],'attach',
                    expected_generation=expected_generation)


def client_key():
    import hashlib
    return hashlib.sha256(client_identity().encode()).hexdigest()[:16]


def claim(path,expected_owner,expected_generation):
    from . import resources as r
    data=json.loads((path/'session.json').read_text());cfg=r.config(data['config'])
    return r.remote(cfg,data['token'],data['remote_port'],'claim',
                    expected_owner=expected_owner,expected_generation=expected_generation)

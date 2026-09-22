"""Create RAG source snapshots when files and embedding compute live apart."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess


def remote_state(cfg,token):
    base=cfg.remote.resource_state_dir or (cfg.remote.workdir+'/.state')
    return base.rstrip('/')+'/resources/'+token+'/rag-snapshot'


def ssh_prefix(cfg):
    return ([] if cfg.ssh.connection=='local' else
            ['ssh','-x','-o','BatchMode=yes','-o','ConnectTimeout='+str(cfg.ssh.connect_timeout_seconds),cfg.ssh.destination])


def snapshot(cfg,allocation,token,paths,source,target,local_root):
    """Return paths visible on target, copying a launch-time snapshot if needed."""
    if source=='shared' or source==target:return list(paths)
    if (source,target)==('local','remote'):
        destination=remote_state(cfg,token)
        _local_to_remote(cfg,allocation,paths,destination)
        return [destination+'/'+str(i)+'-'+Path(path).expanduser().name for i,path in enumerate(paths)]
    if (source,target)==('remote','local'):
        destination=Path(local_root)/'rag-snapshot';destination.mkdir(parents=True,exist_ok=True,mode=0o700)
        _remote_to_local(cfg,allocation,paths,destination)
        return [str(destination/(str(i)+'-'+Path(path).name)) for i,path in enumerate(paths)]
    raise ValueError('RAG paths location and compute location must be local, remote, or shared')


def _remote_command(cfg,allocation,command,input_stream=None,output_stream=None):
    if cfg.backend_type=='kubernetes':
        command=['kubectl',*(['--context',cfg.kubernetes.context] if cfg.kubernetes.context else []),
                 '-n',cfg.kubernetes.namespace,'exec','-i',allocation['host'],'--',*command]
    full=ssh_prefix(cfg)+([shlex.join(command)] if ssh_prefix(cfg) else command)
    return subprocess.run(full,stdin=input_stream,stdout=output_stream,check=True)


def _local_to_remote(cfg,allocation,paths,destination):
    _remote_command(cfg,allocation,['mkdir','-p',destination])
    if cfg.backend_type=='kubernetes':
        for i,raw in enumerate(paths):
            path=Path(raw).expanduser().resolve();name=str(i)+'-'+path.name
            _remote_command(cfg,allocation,['rm','-rf',destination+'/'+name,destination+'/'+path.name])
            producer=subprocess.Popen(['tar','-C',str(path.parent),'-cf','-',path.name],stdout=subprocess.PIPE)
            try:_remote_command(cfg,allocation,['tar','-C',destination,'-xf','-'],input_stream=producer.stdout)
            finally:
                producer.stdout.close();producer.wait()
                if producer.returncode:raise subprocess.CalledProcessError(producer.returncode,producer.args)
            _remote_command(cfg,allocation,['mv',destination+'/'+path.name,destination+'/'+name])
        return
    if cfg.ssh.connection=='local':
        root=Path(destination);root.mkdir(parents=True,exist_ok=True)
        for i,raw in enumerate(paths):_copy(Path(raw).expanduser().resolve(),root/(str(i)+'-'+Path(raw).expanduser().name))
        return
    for i,raw in enumerate(paths):
        path=Path(raw).expanduser().resolve();target=cfg.ssh.destination+':'+destination+'/'+str(i)+'-'+path.name
        if path.is_dir():
            _remote_command(cfg,allocation,['mkdir','-p',destination+'/'+str(i)+'-'+path.name])
            subprocess.run(['rsync','-a','--delete',str(path)+'/',target+'/'],check=True)
        else:subprocess.run(['rsync','-a',str(path),target],check=True)


def _remote_to_local(cfg,allocation,paths,destination):
    if cfg.backend_type=='kubernetes':
        for i,path in enumerate(paths):
            target=destination/(str(i)+'-'+Path(path).name)
            if target.exists():shutil.rmtree(target) if target.is_dir() else target.unlink()
            command=['tar','-C',str(Path(path).parent),'-cf','-',Path(path).name]
            prefix=ssh_prefix(cfg)
            remote=['kubectl',*(['--context',cfg.kubernetes.context] if cfg.kubernetes.context else []),
                    '-n',cfg.kubernetes.namespace,'exec','-i',allocation['host'],'--',*command]
            producer=subprocess.Popen(prefix+([shlex.join(remote)] if prefix else remote),stdout=subprocess.PIPE)
            consumer=subprocess.run(['tar','-C',str(destination),'-xf','-'],stdin=producer.stdout)
            producer.stdout.close();producer.wait()
            if producer.returncode or consumer.returncode:raise ValueError('Could not snapshot Kubernetes RAG paths locally')
            extracted=destination/Path(path).name
            if extracted!=target:extracted.rename(target)
        return
    if cfg.ssh.connection=='local':
        for i,raw in enumerate(paths):_copy(Path(raw).expanduser().resolve(),destination/(str(i)+'-'+Path(raw).name))
        return
    for i,path in enumerate(paths):
        target=destination/(str(i)+'-'+Path(path).name)
        probe=subprocess.run(ssh_prefix(cfg)+[shlex.join(['test','-d',path])]).returncode==0
        if probe:
            target.mkdir(exist_ok=True)
            subprocess.run(['rsync','-a','--delete',cfg.ssh.destination+':'+path.rstrip('/')+'/',str(target)+'/'],check=True)
        else:subprocess.run(['rsync','-a',cfg.ssh.destination+':'+path,str(target)],check=True)


def _copy(source,target):
    if target.exists():shutil.rmtree(target) if target.is_dir() else target.unlink()
    shutil.copytree(source,target) if source.is_dir() else shutil.copy2(source,target)

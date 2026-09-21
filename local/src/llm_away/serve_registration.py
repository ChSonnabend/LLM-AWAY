"""Register a loaded session with Codex and retire the registration on model exit."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid
from .resources import ROOT, path_for, rpc, identity


def alive(record):
    return bool(record.get('provider_identity') and
                identity(record['provider_pid']) == record['provider_identity'])


def edit(record, block=''):
    target=Path(record['config'])
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_name(target.name+'.session-helper.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        text=target.read_text() if target.exists() else ''
        start=f"# BEGIN LLM-AWAY session {record['session']} {record['token']}"
        end=f"# END LLM-AWAY session {record['session']} {record['token']}"
        pattern=re.compile(r'^'+re.escape(start)+r'\n.*?^'+re.escape(end)+r'\n?',re.M|re.S)
        updated=pattern.sub('',text)
        if block:
            name=f"session_helper_{record['session']}"
            if re.search(r'^\s*\[mcp_servers\.'+re.escape(name)+r'\]\s*$',updated,re.M):
                raise ValueError(f'{name} already exists; remove or rename that entry first')
            updated=updated.rstrip()+'\n\n'+start+'\n'+block+end+'\n'
        if updated==text:return
        mode=target.stat().st_mode & 0o777 if target.exists() else 0o600
        fd,tmp=tempfile.mkstemp(dir=target.parent,prefix='.session-helper-')
        try:
            with os.fdopen(fd,'w') as out:out.write(updated)
            os.chmod(tmp,mode);os.replace(tmp,target)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)


def registered(record):
    try:
        current=json.loads(Path(record['record']).read_text())
        marker=f"# BEGIN LLM-AWAY session {record['session']} {record['token']}"
        return current['token']==record['token'] and marker in Path(record['config']).read_text()
    except (OSError,ValueError,KeyError):return False


def watch(record):
    while registered(record) and alive(record):time.sleep(2)
    edit(record)


def remove(path):
    record_path=path/'serve-registration.json'
    if record_path.exists():edit(json.loads(record_path.read_text()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=int)
    parser.add_argument('--rag',action='append',metavar='FOLDER')
    parser.add_argument('--log-helper',action='store_true')
    parser.add_argument('--watch',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args();os.umask(0o077)
    if args.watch:
        try:watch(json.loads(args.watch.read_text()))
        finally:args.watch.unlink(missing_ok=True)
        return
    if args.session is None:parser.error('--session is required')
    register(args.session,args.rag,args.log_helper)


def register(session,rag=None,log_helper=False):
    from argparse import Namespace
    args=Namespace(session=session,rag=rag,log_helper=log_helper)
    os.umask(0o077)
    path=path_for(args.session);data=rpc(path,'status')
    if not data.get('model') or not alive(data):
        raise ValueError(f'Load the model first: run --session {args.session} --helper')
    roots=[str(Path(p).expanduser().resolve()) for p in (args.rag or [os.getcwd()])]
    if any(not Path(p).is_dir() for p in roots):raise ValueError('--rag must be a directory')
    # Prepare dependencies before registration, avoiding MCP startup installation delays.
    from .rag import runtime
    runtime()
    record_path=path/'serve-registration.json'
    with (path/'serve-registration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if record_path.exists():edit(json.loads(record_path.read_text()))
        record=dict(session=args.session,token=uuid.uuid4().hex,
                    config=str((Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'config.toml').resolve()),
                    record=str(record_path),provider_pid=data['provider_pid'],provider_identity=data['provider_identity'])
        tool_args=['--session',str(args.session),'--registration',str(record_path)]
        for root in roots:tool_args+=['--rag',root]
        if log_helper:
            tool_args+=['--log-helper']
            # Make the monitor's helper-log view available immediately, before
            # the first MCP request has something to append.
            log_path=path/'helper.log'
            log_path.touch(exist_ok=True)
            log_path.chmod(0o600)
        block=(f'[mcp_servers.session_helper_{args.session}]\n'
               f'command = {json.dumps(str(ROOT/"bin/session-tool"))}\n'
               f'args = {json.dumps(tool_args)}\nenv_vars = ["LLM_AWAY_SESSION_ID", "LLM_AWAY_SESSION_TOKEN", "TMUX"]\nstartup_timeout_sec = 120\ntool_timeout_sec = 600\n')
        record_path.write_text(json.dumps(record));record_path.chmod(0o600)
        edit(record,block)
        # Watch an immutable generation: replacing a registration cannot retarget its watcher.
        snapshot=path/('serve-watch-'+record['token']+'.json');snapshot.write_text(json.dumps(record))
        try:
            with (path/'session.log').open('a') as log:
                subprocess.Popen([sys.executable,'-m','llm_away.serve_registration','--watch',str(snapshot)],
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        except Exception:
            edit(record);snapshot.unlink(missing_ok=True);raise
    print(f'Added session_helper_{args.session} for '+', '.join(roots)+'. Restart the Codex MCP connection to load it.')


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError) as exc:
        print(str(exc),file=sys.stderr);sys.exit(1)

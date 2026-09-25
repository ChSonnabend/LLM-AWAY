"""Preview and selectively clean local leftovers and released remote session state."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from .resources import STORE, identity, released_ids, remote, config, write
from .serve_registration import remove


def busy(path, check_lease=True):
    try:
        data=json.loads((path/'session.json').read_text())
        if check_lease:
            with (path/'client.lock').open('a') as lease:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:return True
        for key in ('pid','provider_pid','client_pid'):
            if data.get(key):
                try:os.kill(int(data[key]),0);return True
                except ProcessLookupError:pass
                except PermissionError:return True
        return False
    except (OSError,ValueError):return True  # Unknown ownership/state: preserve it.


def add_session_items(items, path, delete_folders=False):
    data=json.loads((path/'session.json').read_text())
    if data.get('phase')=='RELEASED' and not data.get('remote_cleanup_done'):
        items.append(('remote',path,f"Delete released session {path.name} remote state/cache (verified before deletion)"))
    registration=path/'serve-registration.json'
    if registration.exists():items.append(('mcp',path,f'Remove managed MCP registration for session {path.name}; attached helpers exit'))
    tunnel=path/'tunnel.json'
    if tunnel.exists() and not tunnel.is_symlink():
        items.append(('ssh',path,f"Stop/remove session {path.name} SSH tunnel"))
    if (path/'ssh.sock').exists():
        items.append(('socket',path/'ssh.sock',f'Close released session {path.name} SSH control connection'))
    # Keep session.json and lock files: IDs must never be reused, locks must
    # retain their inode. Never infer ownership of arbitrary /tmp directories.
    for name in ('session.log','agent.log','helper.log','model-queries.log','rag.log','rag-bridge.log','rag-bridge.json','rag-snapshot','control.sock','tool-errors','models.json','tmp'):
        target=path/name
        if target.exists() and not target.is_symlink():items.append(('file',target,f'Delete {target}'))
    for target in path.glob('serve-watch-*.json'):
        if not target.is_symlink():items.append(('file',target,f'Delete stale watcher record {target}'))
    if delete_folders and int(path.name) in released_ids():
        items.append(('folder',path,f'Delete released session folder {path.name} (session.json and locks)'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=int)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--keep-folders',action='store_true',
                        help='Do not offer deletion of explicitly RELEASED session folders')
    args=parser.parse_args()
    delete_folders=not args.keep_folders
    sessions=list(STORE.glob('[0-9]*/session.json'))
    items=[]
    for state in sessions:
        path=state.parent
        if args.session is not None and path.name!=str(args.session):continue
        data=json.loads(state.read_text())
        if busy(path):
            print(f'Keep session {path.name}: active or uncertain state');continue
        if data.get('phase') != 'RELEASED' and not data.get('allocation_cleaned'):
            print(f'Keep session {path.name}: allocation has not confirmed cleanup');continue
        add_session_items(items,path,delete_folders)
    if args.session is None and not any(busy(p.parent) for p in sessions):
        for sock in Path('/tmp').glob('llm-away-ssh-*'):
            if sock.is_socket() and sock.lstat().st_uid==os.getuid():
                items.append(('master',sock,f'Close framework SSH master / remove stale socket {sock}'))
    if not items:print('No safely identifiable leftovers.');return
    for number,(_,_,label) in enumerate(items,1):print(f'{number}. {label}')
    if args.preview:return
    chosen=input('Item numbers to clean (comma/space separated), all, or Enter to keep everything: ').strip()
    if not chosen:return
    selected=list(range(len(items))) if chosen=='all' else sorted({int(n)-1 for n in chosen.replace(',',' ').split()})
    if any(i<0 or i>=len(items) for i in selected):raise ValueError('Invalid item number; nothing changed')
    print('Selected:\n'+'\n'.join(items[i][2] for i in selected))
    if input('Proceed? [y/N] ').strip().lower()!='y':return
    execute(items,selected)


def preview(items):
    """Expand cleanup targets without following directory symlinks."""
    paths=[];remote_paths={}
    for kind,target,label in items:
        if kind=='remote':
            data=json.loads((target/'session.json').read_text())
            result=remote(config(data['config']),data['token'],data['remote_port'],'cleanup-preview')
            remote_paths[str(target)]=result['paths']
            paths.extend(str(data.get('host','remote'))+':'+p for p in result['paths'])
        else:
            actual=target/'serve-registration.json' if kind=='mcp' else target/'tunnel.json' if kind=='ssh' else target
            paths.append(str(actual.absolute()))
            if actual.is_dir() and not actual.is_symlink():
                paths.extend(str(p.absolute()) for p in actual.rglob('*'))
    return {'paths':sorted(set(paths)),'remote_paths':remote_paths}


def execute(items,selected,expected_remote=None):
    for i in selected:
        kind,target,label=items[i]
        try:
            if kind=='master':
                if any(busy(p.parent) for p in STORE.glob('[0-9]*/session.json')):
                    print('Skipped SSH master: sessions became active');continue
                result=subprocess.run(['ssh','-S',str(target),'-O','exit','unused'],capture_output=True,timeout=5)
                # This socket was discovered under our private name and no
                # session is active. Remove it even when ssh's control command
                # cannot identify the already-dead master.
                target.unlink(missing_ok=True)
            else:
                session=target if kind in ('mcp','ssh','folder','remote') else target.parent
                with (session/'client.lock').open('a') as lease:
                    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if busy(session,check_lease=False):print(f'Skipped: session {session.name} became active');continue
                    if kind=='remote':
                        data=json.loads((session/'session.json').read_text())
                        if data.get('phase')!='RELEASED':raise ValueError('Session is no longer released')
                        error=None
                        for _ in range(3):
                            try:
                                extra=({'expected_paths':expected_remote[str(target)]} if expected_remote is not None else {})
                                result=remote(config(data['config']),data['token'],data['remote_port'],'cleanup',**extra)
                                error=None;break
                            except Exception as exc:
                                error=exc;time.sleep(1)
                        # A timed-out job from an older controller can leave
                        # local RELEASED state without the remote release file.
                        # Confirm it is inactive, mark it released remotely,
                        # then make one final ownership-checked cleanup attempt.
                        if error and expected_remote is None and 'not explicitly released' in str(error):
                            status=remote(config(data['config']),data['token'],data['remote_port'],'status')
                            if status.get('active'):
                                raise RuntimeError('Remote allocation is active; preserving its state')
                            remote(config(data['config']),data['token'],data['remote_port'],'release',explicit_release=True,only_if_inactive=True)
                            result=remote(config(data['config']),data['token'],data['remote_port'],'cleanup')
                            error=None
                        if error:raise error
                        if not result.get('cleaned'):raise ValueError('Remote cleanup was not confirmed')
                        data['remote_cleanup_done']=True
                        write(session/'session.json',data)
                    elif kind=='mcp':remove(session)
                    elif kind=='ssh':
                        tunnel=session/'tunnel.json'
                        try:record=json.loads(tunnel.read_text())
                        except (OSError,ValueError):record={}
                        pid=record.get('pid');born=record.get('identity')
                        if pid and born and identity(pid)==born:
                            os.kill(pid,signal.SIGTERM)
                            deadline=time.monotonic()+5
                            while identity(pid)==born and time.monotonic()<deadline:time.sleep(.1)
                            if identity(pid)==born:os.kill(pid,signal.SIGKILL)
                        tunnel.unlink(missing_ok=True)
                    elif kind=='folder':
                        data=json.loads((session/'session.json').read_text())
                        if data.get('phase')!='RELEASED' or not data.get('remote_cleanup_done'):
                            raise ValueError('Clean remote state first; retaining session record for retry')
                        shutil.rmtree(target)
                    elif kind=='socket':
                        result=subprocess.run(['ssh','-S',str(target),'-O','exit','unused'],capture_output=True,timeout=5)
                        target.unlink(missing_ok=True)
                    elif target.is_symlink():raise ValueError('Path became a symlink; preserved')
                    elif target.is_dir():shutil.rmtree(target)
                    else:target.unlink(missing_ok=True)
            print('Done: '+label)
        except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:print(f'Kept / failed: {label}: {exc}')


if __name__=='__main__':
    try:main()
    except (ValueError,EOFError,KeyboardInterrupt) as exc:print(str(exc) or 'Canceled')

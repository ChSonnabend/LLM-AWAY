"""Preview and selectively clean identifiable local framework leftovers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from .resources import STORE, identity, released_ids
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
    registration=path/'serve-registration.json'
    if registration.exists():items.append(('mcp',path,f'Remove managed MCP registration for session {path.name}; attached helpers exit'))
    tunnel=path/'tunnel.json'
    if tunnel.exists():
        try:
            record=json.loads(tunnel.read_text())
            if record.get('identity') and identity(record['pid'])==record['identity']:
                items.append(('ssh',path,f"Stop recorded session {path.name} tunnel PID {record['pid']}"))
        except (OSError,ValueError,KeyError):pass
    # Keep session.json and lock files: IDs must never be reused, locks must
    # retain their inode. Never infer ownership of arbitrary /tmp directories.
    for name in ('session.log','tool-errors','models.json','tmp'):
        target=path/name
        if target.exists() and not target.is_symlink():items.append(('file',target,f'Delete {target}'))
    for target in path.glob('serve-watch-*.json'):
        if not target.is_symlink():items.append(('file',target,f'Delete stale watcher record {target}'))
    if delete_folders and int(path.name) in released_ids():
        items.append(('folder',path,f'Delete released session folder {path.name} (session.json and locks)'))
    for lock in path.glob('*.lock'):
        if lock.is_file() and not lock.is_symlink():items.append(('file',lock,f'Delete lock {lock}'))


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
        if busy(path):
            print(f'Keep session {path.name}: active or uncertain state');continue
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
    for i in selected:
        kind,target,label=items[i]
        try:
            if kind=='master':
                if any(busy(p.parent) for p in STORE.glob('[0-9]*/session.json')):
                    print('Skipped SSH master: sessions became active');continue
                result=subprocess.run(['ssh','-S',str(target),'-O','exit','unused'],capture_output=True,timeout=5)
                # Only unlink a refused/missing master; preserve ambiguous errors.
                if result.returncode and b'Connection refused' in result.stderr:target.unlink(missing_ok=True)
                elif result.returncode:raise RuntimeError(result.stderr.decode(errors='replace').strip())
            else:
                session=target if kind in ('mcp','ssh','folder') else target.parent
                with (session/'client.lock').open('a') as lease:
                    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if busy(session,check_lease=False):print(f'Skipped: session {session.name} became active');continue
                    if kind=='mcp':remove(session)
                    elif kind=='ssh':
                        record=json.loads((session/'tunnel.json').read_text())
                        if record.get('identity') and identity(record['pid'])==record['identity']:
                            os.kill(record['pid'],signal.SIGTERM)
                    elif kind=='folder':
                        shutil.rmtree(target)
                    elif target.is_symlink():raise ValueError('Path became a symlink; preserved')
                    elif target.is_dir():shutil.rmtree(target)
                    else:target.unlink(missing_ok=True)
            print('Done: '+label)
        except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:print(f'Kept / failed: {label}: {exc}')


if __name__=='__main__':
    try:main()
    except (ValueError,EOFError,KeyboardInterrupt) as exc:print(str(exc) or 'Canceled')

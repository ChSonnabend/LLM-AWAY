"""Release a numbered session when its persistent runner dies unexpectedly."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid


def read(path):
    return json.loads(path.read_text())


@contextmanager
def locked(path):
    with (path/'runner.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def live(pid, born):
    from .resources import identity
    if not pid or not born:return False
    try:os.kill(int(pid),0)
    except ProcessLookupError:return False
    except PermissionError:return True
    current=identity(pid)
    # An unavailable process query is not evidence that the process died.
    return not current or current==born


def register(path, pid=None):
    from .resources import identity, write
    pid=pid or os.getpid();born=identity(pid)
    if not born:raise RuntimeError('Cannot identify session runner; watchdog not started')
    with locked(path):
        data=read(path/'session.json')
        if data.get('phase')=='RELEASED':raise ValueError('Session already released')
        try:
            old=read(path/'runner.json')
            if (old.get('token')==data['token'] and old.get('pid')==pid and old.get('identity')==born
                    and live(old.get('guard_pid'),old.get('guard_identity'))):return
        except (OSError,ValueError):pass
        record=dict(pid=pid,identity=born,token=data['token'],generation=uuid.uuid4().hex)
        write(path/'runner.json',record)
        with (path/'session.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','llm_away.session_guard',str(path),record['generation']],
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        record.update(guard_pid=child.pid,guard_identity=identity(child.pid))
        write(path/'runner.json',record)


def finished(path):
    """A normal native CLI exit keeps its record available for reopening."""
    from .resources import write
    with locked(path):
        record=read(path/'runner.json')
        if record['pid']==os.getpid():
            record['finished']=True;write(path/'runner.json',record)


def ensure(path, data):
    """Adopt existing runners when res-mon is opened after an upgrade."""
    from .resources import identity
    if data.get('allocation_cleaned'):return None
    try:
        record=read(path/'runner.json')
        if record.get('token')==data['token']:
            if record.get('finished'):return None
            if live(record.get('guard_pid'),record.get('guard_identity')):return record['pid']
            if live(record['pid'],record['identity']):
                register(path,record['pid']);return record['pid']
            with locked(path):
                current=read(path/'runner.json')
                if current['generation']!=record['generation']:return current['pid']
                if live(current.get('guard_pid'),current.get('guard_identity')):return current['pid']
                with (path/'session.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-m','llm_away.session_guard',str(path),record['generation']],
                        stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                record.update(guard_pid=child.pid,guard_identity=identity(child.pid))
                from .resources import write
                write(path/'runner.json',record)
            return record['pid']
    except (OSError,ValueError,KeyError):pass
    pid=data.get('pid')
    if data.get('native'):
        from .terminals import engine
        e=engine()
        if not e['alive'](path):return None
        pid=int(subprocess.check_output(e['tmux'](path)+['display-message','-p','-t',e['name'](path),'#{pane_pid}'],text=True).strip())
    if not pid or not identity(pid):return None
    # Never adopt a recycled PID from an old session record.
    command=shlex.split(subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True).strip())
    role='worker' if data.get('native') else 'daemon'
    if len(command)<2 or command[-2:]!=[role,str(path)]:return None
    register(path,pid)
    return pid


def stop_process(pid, born):
    from .resources import identity
    if not pid or not born or identity(pid)!=born:return
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:return
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and identity(pid)==born:time.sleep(.2)
    if identity(pid)==born:
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass


def cleanup(path, data):
    from .resources import remote, config, write
    from .terminals import engine
    from .serve_registration import remove
    # Attempt every cleanup step even if another step fails.
    errors=[]
    def attempt(action):
        try:action()
        except Exception as exc:errors.append(str(exc))
    attempt(lambda:engine()['stop'](path))
    attempt(lambda:remove(path))
    for filename,pid_key,born_key in [('attachment.json','client_pid','client_identity'),('tunnel.json','pid','identity')]:
        try:record=read(path/filename)
        except (OSError,ValueError):continue
        attempt(lambda:stop_process(record.get(pid_key),record.get(born_key)))
    attempt(lambda:stop_process(data.get('provider_pid'),data.get('provider_identity')))
    if not data.get('native'):
        attempt(lambda:remote(config(data['config']),data['token'],data['remote_port'],'release'))
    if errors:raise RuntimeError('; '.join(errors))
    data.update(phase='RELEASED',model='',error='',provider_pid=None,client_pid=None)
    write(path/'session.json',data)
    (path/'control.sock').unlink(missing_ok=True)
    print('Runner exited; agent stopped and resources released.',flush=True)


def watch(path, generation):
    from .resources import write
    while True:
        with locked(path):
            record=read(path/'runner.json');data=read(path/'session.json')
            if (record['generation']!=generation or record['token']!=data['token']
                    or record.get('finished') or data.get('allocation_cleaned') or data.get('phase')=='RELEASED'):return
            dead=not live(record['pid'],record['identity'])
            if dead:
                try:cleanup(path,data);return
                except Exception as exc:
                    data.update(phase='RELEASE FAILED',error='Runner exited; cleanup will retry: '+str(exc))
                    write(path/'session.json',data)
                    print(data['error'],flush=True)
        time.sleep(5 if dead else 1)


if __name__=='__main__':watch(Path(sys.argv[1]),sys.argv[2])

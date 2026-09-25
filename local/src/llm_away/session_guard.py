"""Track local runner loss without releasing persistent remote allocations."""
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


GUARD_VERSION = 2  # Remote runner loss detaches; it never releases the allocation.


def read(path):
    return json.loads(path.read_text())


@contextmanager
def locked(path):
    with (path/'runner.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def process_key(pid):
    """Linux process identity independent of locale, timezone and recycled PIDs."""
    try:
        stat=Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return boot+':'+stat[19]
    except (OSError,ValueError,IndexError,TypeError):return ''


def same_birth(left, right):
    """Compare legacy ps lstart records written in English or German."""
    if left==right:return True
    months=dict(zip('Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split(),range(1,13)))
    months.update(Mär=3,Mrz=3,Mai=5,Okt=10,Dez=12)
    def parts(value):
        fields=str(value).split()
        if len(fields)!=5 or fields[1] not in months:return None
        return (months[fields[1]],fields[2],fields[3],fields[4])
    a,b=parts(left),parts(right)
    return a is not None and a==b


def live(pid, born, key=None):
    from .resources import identity
    if not pid or not born:return False
    try:os.kill(int(pid),0)
    except ProcessLookupError:return False
    except PermissionError:return True
    if key:
        current=process_key(pid)
        return not current or current==key
    current=identity(pid)
    # An unavailable process query is not evidence that the process died.
    return not current or same_birth(current,born)


def register(path, pid=None):
    from .resources import identity, write
    pid=pid or os.getpid();born=identity(pid)
    if not born:raise RuntimeError('Cannot identify session runner; watchdog not started')
    with locked(path):
        data=read(path/'session.json')
        if data.get('phase')=='RELEASED':raise ValueError('Session already released')
        try:
            old=read(path/'runner.json')
            if (old.get('guard_version')==GUARD_VERSION and old.get('token')==data['token'] and old.get('pid')==pid and same_birth(old.get('identity'),born)
                    and live(old.get('guard_pid'),old.get('guard_identity'),old.get('guard_key'))):return
        except (OSError,ValueError):pass
        record=dict(pid=pid,identity=born,process_key=process_key(pid),token=data['token'],generation=uuid.uuid4().hex,guard_version=GUARD_VERSION)
        write(path/'runner.json',record)
        with (path/'session.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','llm_away.session_guard',str(path),record['generation']],
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        record.update(guard_pid=child.pid,guard_identity=identity(child.pid),guard_key=process_key(child.pid))
        write(path/'runner.json',record)


def finished(path):
    """Mark a local runner finished without changing remote allocation state."""
    from .resources import write
    with locked(path):
        record=read(path/'runner.json')
        if record['pid']==os.getpid():
            record['finished']=True;write(path/'runner.json',record)


def ensure(path, data):
    """Adopt existing runners when res-mon is opened after an upgrade."""
    from .resources import identity
    if data.get('allocation_cleaned') or data.get('phase') in ('RELEASED','STOPPED'):return None
    try:
        record=read(path/'runner.json')
        if record.get('token')==data['token']:
            if record.get('finished'):return None
            if not data.get('native') and record.get('guard_version')!=GUARD_VERSION:
                # Old Python watchdogs retain their destructive cleanup policy in
                # memory after a checkout update. Fence them under the same lock
                # they take before cleanup; they exit on their next iteration.
                from .resources import write
                with locked(path):
                    record=read(path/'runner.json')
                    if record.get('token')!=data['token'] or record.get('finished'):return None
                    if record.get('guard_version')==GUARD_VERSION:return record['pid']
                    record.update(generation=uuid.uuid4().hex,guard_version=GUARD_VERSION,
                                  guard_pid=None,guard_identity='',guard_key='')
                    write(path/'runner.json',record)
                    with (path/'session.log').open('a') as log:
                        child=subprocess.Popen([sys.executable,'-m','llm_away.session_guard',str(path),record['generation']],
                            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                    record.update(guard_pid=child.pid,guard_identity=identity(child.pid),guard_key=process_key(child.pid))
                    write(path/'runner.json',record)
                return record['pid']
            if live(record.get('guard_pid'),record.get('guard_identity'),record.get('guard_key')):return record['pid']
            if live(record['pid'],record['identity'],record.get('process_key')):
                register(path,record['pid']);return record['pid']
            with locked(path):
                current=read(path/'runner.json')
                if current['generation']!=record['generation']:return current['pid']
                if live(current.get('guard_pid'),current.get('guard_identity'),current.get('guard_key')):return current['pid']
                with (path/'session.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-m','llm_away.session_guard',str(path),record['generation']],
                        stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                record.update(guard_pid=child.pid,guard_identity=identity(child.pid),guard_key=process_key(child.pid))
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
    from .resources import write
    if not data.get('native'):
        # A local process disappearing says nothing about remote job lifetime.
        data.update(pid=None,monitor_detached=True,
                    error='Local monitor stopped; remote allocation and model retained. Reattach to reconnect.')
        write(path/'session.json',data)
        (path/'control.sock').unlink(missing_ok=True)
        print('Local monitor exited; remote allocation and model retained.',flush=True)
        return
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
    if errors:raise RuntimeError('; '.join(errors))
    data.update(phase='RELEASED',model='',error='',provider_pid=None,client_pid=None)
    write(path/'session.json',data)
    (path/'control.sock').unlink(missing_ok=True)
    print('Runner exited; agent stopped and resources released.',flush=True)


def watch(path, generation):
    from .resources import write, identity
    while True:
        with locked(path):
            record=read(path/'runner.json');data=read(path/'session.json')
            if (record['generation']!=generation or record['token']!=data['token']
                    or record.get('finished') or data.get('allocation_cleaned') or data.get('phase')=='RELEASED'):return
            dead=not live(record['pid'],record['identity'],record.get('process_key'))
            if dead:
                print(json.dumps(dict(event='runner_lost',time=time.time(),runner_pid=record['pid'],
                    expected_identity=record['identity'],observed_identity=identity(record['pid']),
                    expected_key=record.get('process_key'),observed_key=process_key(record['pid']),guard_pid=os.getpid())),flush=True)
                try:
                    cleanup(path,data)
                    record['finished']=True;write(path/'runner.json',record)
                    return
                except Exception as exc:
                    data.update(error='Local runner exited; cleanup will retry: '+str(exc))
                    write(path/'session.json',data)
                    print(data['error'],flush=True)
        time.sleep(5 if dead else 1)


if __name__=='__main__':watch(Path(sys.argv[1]),sys.argv[2])

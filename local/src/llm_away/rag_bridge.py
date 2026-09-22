"""Bridge a local stdio RAG server to an allocation-side Unix socket server."""
import json
import os
import signal
import subprocess
import sys
import threading
import time


def copy(source,target):
    try:
        while True:
            data=os.read(source.fileno(),65536)
            if not data:break
            target.write(data);target.flush()
    except (BrokenPipeError,OSError):pass
    finally:
        try:target.close()
        except OSError:pass


def main():
    local_command=json.loads(sys.argv[1]);remote_command=json.loads(sys.argv[2])
    local=subprocess.Popen(local_command,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    remote=subprocess.Popen(remote_command,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    children=(local,remote)
    def stop(*_):
        for child in children:
            if child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    threads=[threading.Thread(target=copy,args=(remote.stdout,local.stdin),daemon=True),
             threading.Thread(target=copy,args=(local.stdout,remote.stdin),daemon=True)]
    for thread in threads:thread.start()
    while all(child.poll() is None for child in children):
        time.sleep(.2)
    stop()
    for child in children:
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()

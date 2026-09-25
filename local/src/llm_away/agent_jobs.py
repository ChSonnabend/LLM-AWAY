"""Detached local CLI tasks, with a lease held for their entire lifetime."""
import json
import os
from pathlib import Path
import runpy
import signal
import sys


def main():
    from .resources import write, identity
    path=Path(sys.argv[1]);fd=int(sys.argv[2]);job=json.loads((path/'agent-job.json').read_text())
    signal.signal(signal.SIGTERM,lambda *args:sys.exit('Agent stopped'))
    record={'id':job['id'],'status':'RUNNING','text':'','error':'','pid':os.getpid(),'identity':identity(os.getpid())}
    write(path/'agent-result.json',record)
    try:
        data=json.loads((path/'session.json').read_text())
        write(path/'attachment.json',{'client_pid':os.getpid(),'client_identity':identity(os.getpid())})
        engine=runpy.run_path(str(Path(__file__).resolve().parents[3]/'remote/bin/resource-agent'))
        def emit(text):
            record['text']=(record['text']+text)[-32768:];write(path/'agent-result.json',record)
        job['api_key_file']=str(path/'api-key')
        engine['run_agent'](job,'http://127.0.0.1:'+str(data['config']['server']['port']),job['cwd'],path/'local-agent-history.json',emit)
        record['status']='DONE'
    except BaseException as exc:record.update(status='FAILED',error=str(exc) or 'Agent interrupted')
    finally:
        write(path/'agent-result.json',record)
        (path/'attachment.json').unlink(missing_ok=True)
        os.close(fd)


if __name__=='__main__':main()

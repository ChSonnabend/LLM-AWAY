"""Track live helper MCP connections between numbered resource sessions."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import uuid


def master_session(store):
    number=os.environ.get('LLM_AWAY_SESSION_ID','')
    token=os.environ.get('LLM_AWAY_SESSION_TOKEN','')
    # Older running tmux agents have no explicit session environment tags.
    socket=Path(os.environ.get('TMUX','').split(',')[0]).name
    for path in store.glob('*/session.json'):
        try:
            data=json.loads(path.read_text())
            if data.get('phase')=='RELEASED':continue
            explicit=str(data['id'])==number and data['token']==token
            inherited=bool(data['token']) and socket=='llm-away-'+data['token']
            if explicit or inherited:return data
        except (OSError,ValueError,KeyError):continue
    return None


@contextmanager
def connection(helper_path):
    """An OS-held lock disappears even if the MCP process is killed abruptly."""
    master=master_session(helper_path.parent)
    if master is None or str(master['id'])==helper_path.name:
        yield
        return
    helper=json.loads((helper_path/'session.json').read_text())
    folder=helper_path/'helper-clients';folder.mkdir(mode=0o700,exist_ok=True)
    path=folder/(uuid.uuid4().hex+'.json')
    try:
        with path.open('x+') as stream:
            path.chmod(0o600)
            fcntl.flock(stream,fcntl.LOCK_EX)
            json.dump({'master':master['id'],'master_token':master['token'],
                       'helper':helper['id'],'helper_token':helper['token']},stream)
            stream.flush()
            yield
    finally:path.unlink(missing_ok=True)


def annotate(store, rows):
    by_id={str(row['id']):row for row in rows if row.get('phase')!='RELEASED'}
    masters={key:set() for key in by_id};slaves={key:set() for key in by_id}
    for helper_id,helper in by_id.items():
        for path in (store/helper_id/'helper-clients').glob('*.json'):
            try:
                with path.open() as stream:
                    try:
                        fcntl.flock(stream,fcntl.LOCK_SH|fcntl.LOCK_NB)
                        continue  # No live MCP connection holds this record.
                    except BlockingIOError:pass
                    record=json.load(stream)
                master_id=str(record['master']);master=by_id.get(master_id)
                if (master is None or master_id==helper_id or str(record['helper'])!=helper_id
                        or record['master_token']!=master.get('token')
                        or record['helper_token']!=helper.get('token')):continue
                masters[master_id].add(int(helper_id));slaves[helper_id].add(int(master_id))
            except (OSError,ValueError,KeyError,TypeError):continue
    for key,row in by_id.items():
        row['is_master']=', '.join(map(str,sorted(masters[key]))) or '—'
        row['is_slave']=', '.join(map(str,sorted(slaves[key]))) or '—'
    return rows

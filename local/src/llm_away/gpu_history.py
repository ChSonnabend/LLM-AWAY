"""Durable per-allocation GPU samples, independent of browser lifetime."""
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3


def append(path, telemetry):
    timestamp=(telemetry or {}).get('timestamp')
    lines=(telemetry or {}).get('lines')
    if not isinstance(timestamp,(int,float)) or not math.isfinite(timestamp):return
    if not isinstance(lines,list) or not lines or not all(isinstance(line,str) for line in lines):return
    database=Path(path)/'gpu-history.sqlite3'
    with closing(sqlite3.connect(database,timeout=10)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, timestamp REAL UNIQUE NOT NULL, lines TEXT NOT NULL)')
        db.execute('INSERT OR IGNORE INTO samples(timestamp,lines) VALUES (?,?)',(timestamp,json.dumps(lines)))
        db.commit()


def read(path, after=0, limit=2000):
    after=max(0,int(after));limit=max(1,min(2000,int(limit)))
    database=Path(path)/'gpu-history.sqlite3'
    if not database.exists():return {'samples':[],'cursor':after,'more':False}
    with closing(sqlite3.connect(database,timeout=10)) as db:
        rows=db.execute('SELECT id,timestamp,lines FROM samples WHERE id>? ORDER BY id LIMIT ?',(after,limit+1)).fetchall()
    more=len(rows)>limit;rows=rows[:limit]
    return {'samples':[{'gpu_timestamp':stamp,'gpu_lines':json.loads(lines)} for _,stamp,lines in rows],
            'cursor':rows[-1][0] if rows else after,'more':more}


def follow(paths):
    """Bridge already-running, pre-history daemons without restarting allocations."""
    import fcntl
    import time
    locks=[];active=[]
    for path in map(Path,paths):
        try:
            lock=(path/'gpu-history.lock').open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lock.close();continue
            locks.append(lock);active.append(path)
        except OSError:continue
    try:
        while active:
            for path in active[:]:
                try:
                    data=json.loads((path/'session.json').read_text())
                    append(path,(data.get('allocation') or {}).get('gpu_telemetry'))
                    if data.get('native') or data.get('allocation_cleaned') or data.get('phase')=='RELEASED':active.remove(path)
                except FileNotFoundError:active.remove(path)
                except Exception as exc:print(f'GPU history {path.name}: {exc}',flush=True)
            if active:time.sleep(5)
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    import sys
    follow(sys.argv[1:])

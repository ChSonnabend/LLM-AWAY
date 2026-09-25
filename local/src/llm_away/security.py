"""Private per-allocation credentials; never store secrets in public session metadata."""
import os
import tempfile
from pathlib import Path


def write_key(path, key):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix='.api-key-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:stream.write(key+'\n')
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def headers(path):
    path=Path(path)/'api-key'
    key=path.read_text().strip() if path.exists() else ''
    return {'Authorization':'Bearer '+key} if key else {}

#!/usr/bin/env python3
"""Cache an Apptainer image before reserving GPUs; concurrent callers share a lock."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

image = Path(os.path.expandvars(os.path.expanduser(sys.argv[1])))
source = sys.argv[2]
if not image.is_absolute():
    sys.exit('Container path must be absolute: ' + str(image))
image.parent.mkdir(parents=True, exist_ok=True)
with open(str(image) + '.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if image.is_file() and image.stat().st_size:
        sys.exit(0)
    if image.exists():
        sys.exit('Container exists but is empty or not a file: ' + str(image))
    if not source.startswith('docker://'):
        sys.exit('Missing container; configure container_source with a docker:// image URI')
    if not shutil.which('apptainer'):
        sys.exit('apptainer is required on the login host to download the container')
    print('llm-away: downloading container once to ' + str(image), file=sys.stderr, flush=True)
    # Build in the same filesystem, then publish atomically. A failed pull is never reused.
    with tempfile.TemporaryDirectory(prefix='.' + image.name + '-', dir=str(image.parent)) as temporary:
        env = dict(os.environ)
        env['APPTAINER_TMPDIR'] = temporary
        env['APPTAINER_CACHEDIR'] = str(image.parent / '.apptainer-cache')
        staged = str(Path(temporary) / 'image.sif')
        subprocess.run(['apptainer', 'pull', staged, source], env=env,
                       stdout=sys.stderr, stderr=sys.stderr, check=True)
        if not Path(staged).is_file() or not Path(staged).stat().st_size:
            sys.exit('Container pull did not produce a nonempty SIF')
        os.replace(staged, str(image))

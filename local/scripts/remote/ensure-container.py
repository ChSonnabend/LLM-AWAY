#!/usr/bin/env python3
"""Cache an Apptainer image before reserving GPUs; concurrent callers share a lock."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import json

image = Path(os.path.expandvars(os.path.expanduser(sys.argv[1])))
source = sys.argv[2]
runtime = sys.argv[3] if len(sys.argv) > 3 else 'apptainer'
if not image.is_absolute():
    sys.exit('Container path must be absolute: ' + str(image))
image.parent.mkdir(parents=True, exist_ok=True)
with open(str(image) + '.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if runtime == 'docker':
        if not image.is_file() or image.suffix == '.sif':
            sys.exit('Docker requires a readable docker-save archive, not a SIF')
        if not shutil.which('docker'):
            sys.exit('Docker is not available on this host')
        metadata = Path(str(image)+'.docker.json')
        signature = [image.stat().st_size, image.stat().st_mtime_ns]
        cached = json.loads(metadata.read_text()) if metadata.exists() else {}
        if cached.get('signature') == signature:
            found = subprocess.run(['docker','image','inspect',cached['image']], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if found.returncode == 0:
                print(cached['image'])
                sys.exit(0)
        result = subprocess.run(['docker','load','--input',str(image)], stdout=subprocess.PIPE,
                                stderr=sys.stderr, universal_newlines=True, check=True)
        names = [line.split(': ',1)[1] for line in result.stdout.splitlines()
                 if line.startswith(('Loaded image: ', 'Loaded image ID: '))]
        if len(names) != 1:
            sys.exit('Provide a Docker archive containing exactly one image')
        # Store the immutable ID, not a tag another load could reassign.
        image_id = subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',names[0]], universal_newlines=True).strip()
        metadata.write_text(json.dumps({'image':image_id,'signature':signature}))
        print(image_id)
        sys.exit(0)
    if runtime != 'apptainer':
        sys.exit('Unsupported container runtime: '+runtime)
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

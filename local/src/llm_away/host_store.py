"""Machine-local host profiles; never store credentials."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile


def store_path(config_path):
    path = Path(config_path).with_suffix('.hosts.json')
    legacy = path.with_name('away.hosts.json')
    if path.name == 'model.hosts.json' and not path.exists() and legacy.exists():
        return legacy
    return path


def read_store(config_path):
    path = store_path(config_path)
    if not path.exists():
        return {'active': '', 'hosts': {}}
    data = json.loads(path.read_text())
    if not isinstance(data.get('hosts'), dict):
        raise ValueError('Invalid host settings: ' + str(path))
    return data


def write_store(config_path, data):
    path = store_path(config_path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        temporary = f.name
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def apply_profile(cfg, profile):
    changes = {}
    for key in ('ssh', 'remote', 'slurm', 'llamacpp', 'model', 'kubernetes'):
        if key not in profile:
            continue
        values = dict(profile[key])
        if key == 'ssh':
            values.setdefault('connection', 'ssh')  # Profiles saved before local transport existed.
        changes[key] = replace(getattr(cfg, key), **values)
    changes['backend_type'] = profile['backend_type']
    return replace(cfg, **changes)

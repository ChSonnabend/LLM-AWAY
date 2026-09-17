"""Interactive, persistent setup for SSH compute hosts."""
from dataclasses import asdict, replace
import glob
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from .config import load_config, LlamaCppConfig, SlurmConfig, RemoteConfig, KubernetesConfig
from .host_store import apply_profile, read_store, write_store, store_path
from .models import discover_models, choose_model, choose_mtp


def ssh_hosts(path=None):
    """Enumerate concrete Host aliases, including Include files; skip patterns."""
    aliases, visited = [], set()
    def read(file):
        file = Path(file).expanduser().resolve()
        if file in visited or not file.is_file():
            return
        visited.add(file)
        for line in file.read_text().splitlines():
            parts = shlex.split(line, comments=True)
            if not parts:
                continue
            # OpenSSH accepts both `Host name` and `Host=name`.
            first = parts.pop(0)
            if '=' in first:
                key, value = first.split('=', 1)
                parts.insert(0, value)
            else:
                key = first
            if parts and parts[0] == '=':
                parts.pop(0)
            if key.lower() == 'include':
                for pattern in parts:
                    expanded = Path(pattern).expanduser()
                    if not expanded.is_absolute():
                        expanded = Path.home() / '.ssh' / expanded
                    for match in sorted(glob.glob(str(expanded))):
                        read(match)
            elif key.lower() == 'host':
                for name in parts:
                    if not any(c in name for c in '*?![') and name not in aliases:
                        aliases.append(name)
    read(path or Path.home() / '.ssh/config')
    return aliases


def ask(label, default='', options=None):
    if not sys.stdin.isatty():
        raise ValueError('First-time setup requires a terminal: ' + label)
    while True:
        value = input(label + (f' [{default}]' if default != '' else '') + ': ').strip() or str(default)
        if options is None or value.lower() in options:
            return value.lower() if options else value
        print('Choose ' + ', '.join(options))


def select_host(aliases, current, requested=None):
    if not aliases:
        raise ValueError('No concrete SSH Host aliases found in ~/.ssh/config (including Include files)')
    if requested:
        if requested not in aliases:
            raise ValueError('SSH alias is not configured: ' + requested)
        return requested
    for i, alias in enumerate(aliases, 1):
        print(f'  {i}. {alias}')
    while True:
        value = ask('SSH host (name or number)', current if current in aliases else aliases[0])
        if value.isdigit() and 1 <= int(value) <= len(aliases):
            return aliases[int(value)-1]
        if value in aliases:
            return value


def remote_json(alias, script, *args):
    command = 'python3 - ' + ' '.join(shlex.quote(str(a)) for a in args)
    result = subprocess.run(['ssh', '-o', 'ConnectTimeout=30', alias, command],
                            input=script, text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise ValueError('Remote inspection failed: ' + result.stderr.strip())
    return json.loads(result.stdout)


def absolute(value):
    if not value.startswith('/') or any(c in value for c in '\n\r\0'):
        raise ValueError('Enter a full absolute remote path')
    return value.rstrip('/')


PROBE = '''import json, os, shutil, subprocess, sys
root=sys.argv[1]
if not all(os.path.isfile(root+'/'+p) for p in ['bin/run-server','scripts/lib/llamacpp-env.sh']):
    sys.exit('Not an LLM-AWAY-remote installation: '+root)
tools={n: bool(shutil.which(n)) for n in ['apptainer','docker','sbatch','kubectl','nvidia-smi','rocminfo']}
parts=[]
if tools['sbatch']:
    p=subprocess.run(['sinfo','-h','-o','%P'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    parts=list(dict.fromkeys(p.stdout.replace('*','').split()))
print(json.dumps({'tools':tools,'partitions':parts}))
'''


def configure_local(config_path, alias=None, workdir=None, restart=False,
                    model=None, mtp=None, skip_model_selection=False):
    cfg = load_config(config_path)
    store = read_store(config_path)
    alias = select_host(ssh_hosts(), store.get('active') or cfg.ssh.host, alias)
    old = store['hosts'].get(alias)
    if old and not restart:
        profile = old
    else:
        # Use an existing named profile for sensible defaults, never another host's settings.
        try:
            defaults = cfg.with_host(alias)
        except ValueError:
            defaults = None
        root = absolute(workdir or ask('Full path to LLM-AWAY-remote',
                        old['remote']['workdir'] if old else (defaults.remote.workdir if defaults else '')))
        info = remote_json(alias, PROBE, root)
        use_container = ask('Container required? yes/no', 'yes' if (old and old['llamacpp']['container']) or (defaults and defaults.llamacpp.container) else 'no', ('yes','no')) == 'yes'
        container, runtime = '', 'apptainer'
        if use_container:
            container = absolute(ask('Full path to container (SIF or Docker archive)',
                                 old['llamacpp']['container'] if old else (defaults.llamacpp.container if defaults else '')))
        scheduler = ask('Submission system', 'slurm', ('slurm','kubernetes'))
        if use_container and scheduler == 'slurm':
            available = [x for x in ('apptainer','docker') if info['tools'][x]]
            if not available:
                raise ValueError('Neither Apptainer nor Docker is available on this SSH host')
            runtime = 'apptainer' if container.endswith('.sif') and 'apptainer' in available else available[0]
            if len(available) > 1 and not container.endswith('.sif'):
                runtime = ask('Container runtime', 'docker', available)
            print('Container runtime: ' + runtime)
            if runtime == 'docker' and container.endswith('.sif'):
                raise ValueError('Docker cannot run SIF images; provide a Docker archive or use Apptainer')
            remote_json(alias, "import json,os,sys; p=sys.argv[1]; assert os.path.isfile(p), 'Container not found: '+p; print(json.dumps(True))", container)
        if not info['tools']['sbatch' if scheduler == 'slurm' else 'kubectl']:
            raise ValueError('Required scheduler command is not installed on ' + alias)
        backend = ask('GPU backend', defaults.llamacpp.backend if defaults else 'cuda', ('cuda','rocm','cpu'))
        if scheduler == 'slurm' and use_container and runtime == 'docker' and backend == 'rocm':
            raise ValueError('ROCm Docker device mapping is not supported; use an Apptainer SIF or Kubernetes')
        arch = ask('ROCm architecture', 'gfx908') if backend == 'rocm' else 'auto'
        gpus = int(ask('GPUs per job (0 for CPU)', '0' if backend == 'cpu' else '1'))
        if gpus < 0 or (backend != 'cpu' and gpus < 1):
            raise ValueError('GPU jobs require a positive GPU count')
        llama = asdict(LlamaCppConfig(backend=backend, rocm_arch=arch, visible_devices='',
                        container=container, container_runtime=runtime, build_before_run=False, mtp='auto'))
        # Preserve application context and model-independent generation settings.
        for key in ('context_size','server_extra_args','max_tokens'):
            llama[key] = getattr(cfg.llamacpp, key)
        slurm = asdict(SlurmConfig(gpus=gpus, partition='', exclusive=False))
        kube = asdict(KubernetesConfig())
        if scheduler == 'slurm':
            print('Available partitions: ' + ', '.join(info['partitions']))
            slurm['partition'] = ask('Slurm partition', defaults.slurm.partition if defaults else (info['partitions'][0] if info['partitions'] else ''))
            slurm['exclusive'] = ask('Exclusive node?', 'no', ('yes','no')) == 'yes'
            slurm['custom_options'] = shlex.split(ask('Additional Slurm options', '--cpus-per-task=8 --mem=64G --time=02:00:00'))
            if alias == 'epnh' and defaults:
                slurm['node_class'], slurm['mi50_fallback'] = defaults.slurm.node_class, defaults.slurm.mi50_fallback
        else:
            print('Kubernetes runs OCI images, not SIF files or Docker archives.')
            kube['image'] = ask('Kubernetes OCI image', 'ghcr.io/ggml-org/llama.cpp:server-rocm' if backend == 'rocm' else 'ghcr.io/ggml-org/llama.cpp:server-cuda')
            kube['context'] = ask('kubectl context (blank uses current)', '')
            kube['namespace'] = ask('Kubernetes namespace', 'default')
            kube['pvc'] = ask('PVC containing the remote project at its root (blank: same shared host path)', '')
            kube['gpu_resource'] = 'amd.com/gpu' if backend == 'rocm' else 'nvidia.com/gpu'
            llama['container'] = ''  # No nested container inside a Kubernetes pod.
            probe = "import json,subprocess,sys; c=['kubectl']; c+=['--context',sys.argv[1]] if sys.argv[1] else []; c+=['--namespace',sys.argv[2],'auth','can-i','create','jobs.batch']; p=subprocess.run(c,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True); assert p.returncode==0 and p.stdout.strip()=='yes', 'kubectl cannot create Jobs: '+p.stderr; print(json.dumps(True))"
            remote_json(alias, probe, kube['context'], kube['namespace'])
        profile = {'ssh': {'host':alias,'user':''}, 'remote': asdict(RemoteConfig(
                    workdir=root, state_dir=root+'/.state/'+alias,
                    runner=root+'/scripts/remote/llm-away-slurm-run',
                    serverctl=root+'/scripts/remote/'+('llm-away-k8sctl' if scheduler == 'kubernetes' else 'llm-away-serverctl'))),
                    'llamacpp':llama, 'slurm':slurm, 'kubernetes':kube,
                    'backend_type':'kubernetes' if scheduler == 'kubernetes' else 'slurm_server'}
        required = 'scripts/remote/llm-away-k8sctl' if scheduler == 'kubernetes' else 'scripts/remote/llm-away-serverctl'
        remote_json(alias, "import json,os,sys; assert os.path.isfile(sys.argv[1]), 'Update the remote project: '+sys.argv[1]; print(json.dumps(True))", root+'/'+required)
    selected_cfg = apply_profile(cfg, profile)
    if not skip_model_selection:
        chosen = choose_model(discover_models(selected_cfg), selected_cfg.llamacpp.model_name, model)
        mode = choose_mtp(chosen, 'auto', mtp, interactive=False)
        profile = {**profile, 'model': {'name':chosen['alias']},
                   'llamacpp': {**profile['llamacpp'], 'model_name':chosen['name'], 'mtp':mode}}
    elif not old:
        raise ValueError('A new host needs a model selection; omit --skip-model-selection')
    store['hosts'][alias] = profile
    store['active'] = alias
    write_store(config_path, store)
    print('Saved host '+alias+' in '+str(store_path(config_path)))

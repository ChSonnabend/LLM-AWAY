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
from .prompt import choose_option

def pattern_matches(pattern, requested):
    """OpenSSH Host lists: * and ? wildcards, with stanza-local negation."""
    def match(item):
        expression = ''.join('.*' if c == '*' else '.' if c == '?' else re.escape(c)
                             for c in item)
        return re.fullmatch(expression, requested, re.IGNORECASE) is not None
    items = pattern.split()
    return (any(match(item) for item in items if not item.startswith('!'))
            and not any(match(item[1:]) for item in items if item.startswith('!')))


def ssh_hosts(path=None):
    """Enumerate concrete Host aliases and usable patterns, including Include files."""
    aliases, patterns, visited = [], [], set()
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
                negatives = [name for name in parts if name.startswith('!')]
                for name in parts:
                    if name.startswith('!'):
                        continue
                    group = ' '.join([name, *negatives])
                    if '*' in name or '?' in name:
                        if group not in patterns:
                            patterns.append(group)
                    elif pattern_matches(group, name) and name not in aliases:
                        aliases.append(name)
    read(path or Path.home() / '.ssh/config')
    return aliases, patterns


def ask(label, default='', options=None):
    if not sys.stdin.isatty():
        raise ValueError('First-time setup requires a terminal: ' + label)
    if options:
        choices = list(options)
        return choices[choose_option(choices, label, choices.index(default) if default in choices else 0)]
    value = input(label + (f' [{default}]' if default != '' else '') + ': ').strip() or str(default)
    if value == str(default) and default != '':
        print('  Selected: ' + value)
    return value


def select_host(aliases, current, requested=None, patterns=None):
    patterns = patterns or []
    def allowed(value):
        return (bool(value) and not value.startswith('-')
                and not any(c.isspace() or c in '*?!' for c in value)
                and (value in aliases or any(pattern_matches(p, value) for p in patterns)))
    if requested:
        if allowed(requested):
            print('SSH host: ' + requested)
            return requested
        raise ValueError('SSH alias is not configured: ' + requested)
    choices = [*aliases, *patterns]
    if not choices:
        raise ValueError('No SSH Host aliases or patterns found in ~/.ssh/config')
    default = next((i for i, p in enumerate(choices) if p == current or pattern_matches(p, current)), 0)
    selected = choices[choose_option(choices, 'SSH host', default)]
    if selected in aliases:
        return selected
    while True:
        value = ask('Concrete hostname for ' + selected,
                    current if allowed(current) and pattern_matches(selected, current) else '')
        if allowed(value) and pattern_matches(selected, value):
            return value
        print('Enter a concrete hostname matching ' + selected)


def remote_json(alias, script, *args, connection='ssh'):
    command = 'python3 - ' + ' '.join(shlex.quote(str(a)) for a in args)
    invocation = [sys.executable, '-', *map(str, args)] if connection == 'local' else ['ssh', '-o', 'ConnectTimeout=30', alias, command]
    result = subprocess.run(invocation,
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
    sys.exit('Not an LLM-AWAY/remote installation: '+root)
tools={n: bool(shutil.which(n)) for n in ['apptainer','docker','sbatch','kubectl','nvidia-smi','rocminfo']}
parts=[]
if tools['sbatch']:
    p=subprocess.run(['sinfo','-h','-o','%P'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    parts=list(dict.fromkeys(p.stdout.replace('*','').split()))
shared=next((p for p in ['/lustre/alice/users/csonnab/LLM-AWAY/remote', '/scratch/csonnabe/LLM-AWAY/remote'] if os.path.isdir(p)), '')
import getpass
user=getpass.getuser()
parent=('/lustre/alice/users/'+user if shared.startswith('/lustre/') else '/scratch/'+user)
state_default=(parent+'/.cache/llm-away') if os.path.isdir(parent) else os.path.expanduser('~/.cache/llm-away')
print(json.dumps({'tools':tools,'partitions':parts,'shared_root':shared,'state_default':state_default}))
'''


def configure_local(config_path, alias=None, workdir=None, restart=False,
                    model=None, mtp=None, skip_model_selection=False, connection=None, resources_only=False):
    cfg = load_config(config_path)
    store = read_store(config_path)
    connection = connection or ask('Connection: ssh or local', 'ssh' if alias else cfg.ssh.connection, ('ssh','local'))
    aliases, patterns = ssh_hosts()
    alias = '@local' if connection == 'local' else select_host(aliases, store.get('active') or cfg.ssh.host, alias, patterns)
    def inspect_host(script, *args):
        return remote_json(alias, script, *args, connection=connection)
    old = store['hosts'].get(alias)
    if old and not restart:
        profile = old
    else:
        # Use an existing named profile for sensible defaults, never another host's settings.
        try:
            defaults = cfg.with_host(alias)
        except ValueError:
            defaults = None
        root = absolute(workdir or ask('Full path to LLM-AWAY/remote',
                        old['remote']['workdir'] if old else ''))
        info = inspect_host(PROBE, root)
        shared_root = info.get('shared_root') or root
        # The remote filesystem identifies shared defaults, never the SSH alias.
        if defaults is None and info.get('shared_root'):
            template = 'hydra-h200' if shared_root.startswith('/lustre/') else 'epnh'
            try:
                defaults = cfg.with_host(template)
            except ValueError:
                pass
        previous = old.get('llamacpp', {}) if old else {}
        models_dir = absolute(ask('Full path to models directory (presets and GGUF files)',
                                 previous.get('models_dir') or shared_root+'/models'))
        installation_dir = absolute(ask('Full path to llama.cpp installation (contains builds/ or build/)',
                                       previous.get('installation_dir') or shared_root))
        inspect_host("import json,os,sys; paths=sys.argv[1:]; missing=[p for p in paths if not os.path.isdir(p) or not os.access(p,os.R_OK|os.X_OK)]; assert not missing, 'Directories not accessible: '+', '.join(missing); print(json.dumps(True))", models_dir, installation_dir)
        state_dir = absolute(ask('Your writable state directory (visible on compute nodes)',
                                 (old or {}).get('remote', {}).get('resource_state_dir') or info.get('state_default', '')))
        inspect_host("import json,os,sys,tempfile; p=sys.argv[1]; os.makedirs(p,mode=0o700,exist_ok=True); assert os.stat(p).st_uid==os.getuid(), 'State directory must belong to you'; f=tempfile.TemporaryFile(dir=p); f.close(); print(json.dumps(True))", state_dir)
        use_container = ask('Container required? yes/no', 'yes' if (old and old['llamacpp']['container']) or (defaults and defaults.llamacpp.container) else 'no', ('yes','no')) == 'yes'
        container, runtime = '', 'apptainer'
        shared_container = shared_root+'/containers/llama-server-'+('rocm' if defaults and defaults.llamacpp.backend == 'rocm' else 'cuda')+'.sif'
        if use_container:
            container = absolute(ask('Full path to container (SIF or Docker archive)',
                                 old['llamacpp']['container'] if old else (defaults.llamacpp.container if defaults and defaults.llamacpp.container else shared_container)))
        scheduler = ask('Submission system', 'slurm', ('slurm','kubernetes','direct'))
        if use_container and scheduler != 'kubernetes':
            available = [x for x in ('apptainer','docker') if info['tools'][x]]
            if not available:
                raise ValueError('Neither Apptainer nor Docker is available on this SSH host')
            runtime = 'apptainer' if container.endswith('.sif') and 'apptainer' in available else available[0]
            if len(available) > 1 and not container.endswith('.sif'):
                runtime = ask('Container runtime', 'docker', available)
            print('Container runtime: ' + runtime)
            if runtime == 'docker' and container.endswith('.sif'):
                raise ValueError('Docker cannot run SIF images; provide a Docker archive or use Apptainer')
            inspect_host("import json,os,sys; p=sys.argv[1]; assert os.path.isfile(p), 'Container not found: '+p; print(json.dumps(True))", container)
        if scheduler != 'direct' and not info['tools']['sbatch' if scheduler == 'slurm' else 'kubectl']:
            raise ValueError('Required scheduler command is not installed on ' + alias)
        backend = ask('GPU backend', defaults.llamacpp.backend if defaults else 'cuda', ('cuda','rocm','cpu'))
        if scheduler != 'kubernetes' and use_container and runtime == 'docker' and backend == 'rocm':
            raise ValueError('ROCm Docker device mapping is not supported; use an Apptainer SIF or Kubernetes')
        arch = ask('ROCm architecture', defaults.llamacpp.rocm_arch if defaults and defaults.llamacpp.rocm_arch else 'auto') if backend == 'rocm' else 'auto'
        gpus = int(ask('GPUs per job (0 for CPU)', '0' if backend == 'cpu' else '1'))
        if gpus < 0 or (backend != 'cpu' and gpus < 1):
            raise ValueError('GPU jobs require a positive GPU count')
        if scheduler == 'direct':
            # No Slurm job is ever submitted: the server runs on this host (direct ssh).
            print('Direct mode: no Slurm job will be submitted; the server runs on '
                  + alias + ' over the existing '
                  + ('SSH connection' if connection == 'ssh' else 'local connection')
                  + ' and uses this host directly.')
        llama = asdict(LlamaCppConfig(models_dir=models_dir, installation_dir=installation_dir, backend=backend, rocm_arch=arch, visible_devices='',
                        container=container, container_runtime=runtime, build_before_run=False, mtp='auto'))
        if scheduler == 'direct' and gpus:
            llama['visible_devices'] = ask('GPU device IDs (comma-separated)', ','.join(str(i) for i in range(gpus)))
            if not re.fullmatch(r'\d+(,\d+)*', llama['visible_devices']):
                raise ValueError('GPU device IDs must be comma-separated integers')
        # Preserve application context and model-independent generation settings.
        for key in ('context_size','server_extra_args','max_tokens'):
            llama[key] = getattr(cfg.llamacpp, key)
        slurm = asdict(SlurmConfig(gpus=gpus, partition='', exclusive=False))
        kube = asdict(KubernetesConfig())
        if scheduler == 'slurm':
            print('Available partitions: ' + ', '.join(info['partitions']))
            slurm['partition'] = ask('Slurm partition', defaults.slurm.partition if defaults else (info['partitions'][0] if info['partitions'] else ''), info['partitions'] or None)
            slurm['exclusive'] = ask('Exclusive node?', 'no', ('yes','no')) == 'yes'
            slurm['custom_options'] = shlex.split(ask('Additional Slurm options', '--cpus-per-task=8 --mem=64G --time=02:00:00'))
            if defaults:
                slurm['node_class'], slurm['mi50_fallback'] = defaults.slurm.node_class, defaults.slurm.mi50_fallback
        elif scheduler == 'kubernetes':
            print('Kubernetes runs OCI images, not SIF files or Docker archives.')
            kube['image'] = ask('Kubernetes OCI image', 'ghcr.io/ggml-org/llama.cpp:server-rocm' if backend == 'rocm' else 'ghcr.io/ggml-org/llama.cpp:server-cuda')
            kube['context'] = ask('kubectl context (blank uses current)', '')
            kube['namespace'] = ask('Kubernetes namespace', 'default')
            kube['pvc'] = ask('PVC containing the remote project at its root (blank: same shared host path)', '')
            kube['gpu_resource'] = 'amd.com/gpu' if backend == 'rocm' else 'nvidia.com/gpu'
            llama['container'] = ''  # No nested container inside a Kubernetes pod.
            probe = "import json,subprocess,sys; c=['kubectl']; c+=['--context',sys.argv[1]] if sys.argv[1] else []; c+=['--namespace',sys.argv[2],'auth','can-i','create','jobs.batch']; p=subprocess.run(c,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True); assert p.returncode==0 and p.stdout.strip()=='yes', 'kubectl cannot create Jobs: '+p.stderr; print(json.dumps(True))"
            inspect_host(probe, kube['context'], kube['namespace'])
        controller = {'slurm':'llm-away-serverctl', 'kubernetes':'llm-away-k8sctl', 'direct':'llm-away-directctl'}[scheduler]
        profile = {'ssh': {'host':alias,'user':'','connection':connection}, 'remote': asdict(RemoteConfig(
                    workdir=root, state_dir=state_dir, resource_state_dir=state_dir,
                    runner=root+'/scripts/remote/llm-away-slurm-run',
                    serverctl=root+'/scripts/remote/'+controller)),
                    'llamacpp':llama, 'slurm':slurm, 'kubernetes':kube,
                    'backend_type':'slurm_server' if scheduler == 'slurm' else scheduler}
        required = 'scripts/remote/'+controller
        inspect_host("import json,os,sys; assert os.path.isfile(sys.argv[1]), 'Update the remote project: '+sys.argv[1]; print(json.dumps(True))", root+'/'+required)
    selected_cfg = apply_profile(cfg, profile)
    if not skip_model_selection and not resources_only:
        chosen = choose_model(discover_models(selected_cfg), selected_cfg.llamacpp.model_name, model)
        mode = choose_mtp(chosen, 'auto', mtp, interactive=sys.stdin.isatty())
        profile = {**profile, 'model': {'name':chosen['alias']},
                   'llamacpp': {**profile['llamacpp'], 'model_name':chosen['name'], 'mtp':mode}}
    elif not old and not resources_only:
        raise ValueError('A new host needs a model selection; omit --skip-model-selection')
    store['hosts'][alias] = profile
    store['active'] = alias
    write_store(config_path, store)
    print('Saved host '+alias+' in '+str(store_path(config_path)))

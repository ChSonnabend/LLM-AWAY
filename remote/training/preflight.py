"""Run on the allocated node before unloading inference or starting training."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess


def check_memory(model,precision,capacities):
    if precision!='bf16':return
    base=Path(model).expanduser()
    adapter=base/'adapter_config.json'
    if adapter.is_file():base=Path(json.loads(adapter.read_text()).get('base_model_name_or_path',''))
    path=base/'model.safetensors.index.json'
    if not path.is_file():return
    size=int(json.loads(path.read_text()).get('metadata',{}).get('total_size',0))
    available=int(sum(capacities)*0.8)
    if size>available:
        raise ValueError('BF16 base weights need %.1f GB, but allocated GPUs provide %.1f GB total (%.1f GB after 20%% training reserve). Allocate more GPUs or explicitly choose QLoRA.' % (size/1e9,sum(capacities)/1e9,available/1e9))


def main():
    missing=[n for n in ('unsloth','torch','transformers','pypdf','markitdown') if importlib.util.find_spec(n) is None]
    if missing:raise ValueError('Training environment missing packages: '+', '.join(missing)+'. Select an Unsloth training image (unsloth-training-cuda.sif), not a llama.cpp server image, or install remote/training/requirements.txt in the selected native Python.')
    # Reject BF16 checkpoints that cannot even fit the entire node before a
    # potentially expensive cold PyTorch import from a shared filesystem.
    if sys.argv[2]=='bf16':
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            capacities=[int(line.strip())*1024*1024 for line in result.stdout.splitlines()] if result.returncode==0 else []
        except (OSError,subprocess.TimeoutExpired,ValueError):capacities=[]
        if capacities:check_memory(sys.argv[1],sys.argv[2],capacities)
    # Validate real imports too: package presence alone misses incompatible stacks.
    from runner import configure_compilation
    configure_compilation(sys.argv[1])
    from unsloth import FastModel
    import torch
    if not torch.cuda.is_available():raise ValueError('Training Python cannot access CUDA on the allocated node')
    if sys.argv[2]=='bf16' and not torch.cuda.is_bf16_supported():raise ValueError('Allocated GPUs do not support BF16 training')
    check_memory(sys.argv[1],sys.argv[2],[torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())])
    print('Allocated-node training preflight passed')

if __name__=='__main__':
    try:main()
    except Exception as exc:sys.exit(type(exc).__name__+': '+str(exc))

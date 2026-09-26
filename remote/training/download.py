#!/usr/bin/env python3
"""Resumable, revision-pinned BF16 model downloads; run outside GPU inference."""
import argparse, json, os, shutil, time
from pathlib import Path

def main():
    from huggingface_hub import HfApi, hf_hub_download
    parser=argparse.ArgumentParser();parser.add_argument('repository');parser.add_argument('directory');args=parser.parse_args()
    directory=Path(args.directory);directory.mkdir(parents=True,exist_ok=True)
    os.umask(0o077)
    import fcntl
    lock=(directory/'.download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    api=HfApi();info=api.model_info(args.repository,files_metadata=True)
    selected=[f for f in info.siblings if f.rfilename.endswith(('.safetensors','.json','.txt','.model','.tiktoken','.jinja','.md'))]
    config=json.loads(Path(hf_hub_download(args.repository,'config.json',revision=info.sha,local_dir=str(directory))).read_text())
    dtype=config.get('dtype') or config.get('torch_dtype') or config.get('text_config',{}).get('dtype') or config.get('text_config',{}).get('torch_dtype')
    if dtype not in ('bfloat16','bf16') or config.get('quantization_config'):raise ValueError('Repository is not an unquantized BF16 checkpoint')
    total=sum(f.size or 0 for f in selected);done=0
    if shutil.disk_usage(directory).free<total-sum(p.stat().st_size for p in directory.glob('*.safetensors')):
        raise ValueError('Insufficient free space for BF16 checkpoint')
    def status(phase,**extra):
        data=dict(phase=phase,repository=args.repository,revision=info.sha,total_bytes=total,completed_bytes=done,time=time.time(),**extra)
        temp=directory/'.download-status.tmp';temp.write_text(json.dumps(data));os.replace(temp,directory/'download-status.json')
    try:
        status('DOWNLOADING',files_total=len(selected),files_done=0)
        for i,f in enumerate(selected):
            hf_hub_download(args.repository,f.rfilename,revision=info.sha,local_dir=str(directory))
            done+=f.size or 0;status('DOWNLOADING',files_total=len(selected),files_done=i+1)
        status('COMPLETE',files_total=len(selected),files_done=len(selected))
    except Exception as exc:status('FAILED',error=str(exc));raise
if __name__=='__main__':main()

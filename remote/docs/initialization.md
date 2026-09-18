# Resource initialization

From `LLM-AWAY/local`, install command links once:

```sh
./scripts/install-resource-tools.sh
res-alloc
res-alloc --restart
res-alloc --host epnh --gpus 2
run --session ID --model PRESET_NAME
```

The allocator asks for SSH or local connection, host, remote project path,
container requirements and scheduler (Slurm, Kubernetes or direct). Known hosts
reuse saved settings; `--restart` repeats setup. Each allocation asks for job options.
Model and MTP selection happen in `run`, not during allocation.

Defaults: `config/model.toml`. Private host profiles: `config/model.hosts.json`.
Use the returned session ID with `run` and `res-mon`. Keep the allocation when
exiting the agent to reuse it later; release it explicitly when finished.

Kubernetes requires configured kubectl permissions, a GPU device plugin and a
writable shared project volume (hostPath or PVC), plus an OCI image containing bash
and the runtime. It cannot run SIF images directly. Kubernetes remains untested on
a live cluster. See the repository README for details.

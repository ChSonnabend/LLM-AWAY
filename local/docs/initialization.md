# Initialization and saved hosts

Run `./scripts/init-local.sh`. It lists concrete Host aliases from ~/.ssh/config
and Include files (wildcard/negated patterns cannot be selected as destinations).
Host labels use the SSH alias: epnh is shown as epnh. AWAY remains the project/provider name.

For a new host, setup asks for the remote project path, whether a container is
needed, its absolute path, SLURM/Kubernetes, GPU backend and resource settings.
It discovers available runtimes and installed model presets over SSH. Runtime
selection supports Apptainer SIF files or a single-image `docker save` archive.
Docker is supported for CUDA/CPU; ROCm requires Apptainer or Kubernetes.
Docker mounts only the selected project: model files must be accessible inside it.

Settings are saved atomically in config/away.hosts.json (ignored by Git, mode 0600).
Each SSH alias retains its own model, scheduler, paths, runtime and resources.
Repeat initialization asks only for host and model. `--restart` repeats all setup
questions for the chosen host, and replaces its settings only after successful
model discovery and selection; other hosts remain intact. No jobs are submitted
by initialization. Existing TOML host profiles remain usable.

```bash
./scripts/init-local.sh
./scripts/init-local.sh --restart
./scripts/init-local.sh --ssh-alias epnh --model MODEL_NAME
away-agent
# Explicit override, including saved hosts:
REMOTE_HOST=epnh away-agent
```

The last initialized host/model is the default for away-agent. REMOTE_HOST and
REMOTE_GPUS remain available. Model selection through llm-away select-model also
updates the active saved profile. Existing config/model.toml customizations are
not overwritten by the wizard.

## Kubernetes

The remote project must be updated to include scripts/remote/llm-away-k8sctl.
The SSH host needs kubectl configured with permission to create/get/delete Jobs,
list/get Pods, read logs, and port-forward in the selected namespace. Initialization
checks Job-create permission. The cluster needs the NVIDIA/AMD device plugin for
nvidia.com/gpu or amd.com/gpu resources.

Setup asks for the OCI registry image, kubectl context, namespace and optional
PVC. Kubernetes cannot directly run a SIF file or Docker archive. With a PVC,
its root must contain the remote project (bin, scripts and models); without one,
the same remote project path must be shared on eligible Kubernetes workers.
The project is mounted read-only. Models and referenced files must reside within
that mount. Update the PVC copy of the project when upgrading the local/remote code.

The backend creates a single-node Job, reports scheduling/errors/logs, and uses
kubectl port-forward through the same SSH connection as the local tunnel. No
Service/Ingress or cluster-wide network exposure is created. Ctrl+C cancels jobs
owned by the provider according to the existing gateway cancellation settings.
Custom server_command/model_path settings are not supported by this backend;
use managed model presets. Resource/image changes require canceling an active job
before starting it again.

Kubernetes and Docker execution have not been exercised on a live cluster during
this change. Only syntax/static review was performed, per the no-tests request.

References:
https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster/

# LLM-AWAY

One repository for the local agent gateway and remote llama.cpp runners.

- [local/](local/README.md): host/model selection, local gateway, agent commands and configuration.
- [remote/](remote/README.md): models, containers and direct/Slurm/Kubernetes execution.

```bash
git clone git@github.com:ChSonnabend/LLM-AWAY.git
cd LLM-AWAY/local
./scripts/init-local.sh
```

When setup asks for the remote folder, select the full path ending in
`LLM-AWAY/remote`, not the repository root. Remote setup is run from that folder.
Models, containers, builds and saved host profiles remain ignored by Git.

## Existing installations

- Mac: `/Users/jarvis/alice/misc/LLM-AWAY/local`
- epnh: `/scratch/csonnabe/cern-fellowship/misc/LLM-AWAY/remote`
- Hydra: `/lustre/alice/users/csonnab/cern-fellowship/misc/LLM-AWAY/remote`

The former `LLM-AWAY-local` and `LLM-AWAY-remote` filesystem paths are compatibility
symlinks. Keep them while existing environments, build caches or running sessions
reference those paths. The old `lamacpp-llm` alias on epnh remains valid too.
Both original Git histories are preserved through subtree merge commits. The
original GitHub repositories remain available as historical copies; new changes
belong here. Each installation's former Git metadata is preserved inside
`.git/migration-backup-local` or `.git/migration-backup-remote`.

Run `git pull --ff-only` from the repository root to update both components.
No model downloads or GPU jobs are required for migration.

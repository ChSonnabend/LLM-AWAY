# Qwen3-Coder-Next 80B: shared epnh/hydra layout

Both hosts use the official `Qwen/Qwen3-Coder-Next-GGUF` files and the same
relative paths beneath `remote/`. The recipes contain no host-dependent checks.

| Preset | Precision | Shards | Total bytes |
| --- | --- | --- | --- |
| `qwen3-coder-next-80b-q8` | Q8_0 | 4 | 84,812,055,968 |
| `qwen3-coder-next-80b-f16` | F16 | 4 | 159,458,215,744 |

Weights live in `models/Qwen3-Coder-Next-80B-GGUF/`, with the upstream
`Qwen3-Coder-Next-Q8_0/` and `Qwen3-Coder-Next-F16/` subdirectories and shard
filenames preserved. Recipes live in `models/<preset>/model.env`.

Verified download revision: `b82fb7382639d97b38fa7672e526c760c2fb358e`.
Both presets retain the 262144-token context setting and `--jinja`.
Select the new preset names when starting a server.

From `remote/`, the normal download commands are:

```bash
bin/download-model qwen3-coder-next-80b-q8
bin/download-model qwen3-coder-next-80b-f16
```

Hydra's previous Unsloth Q8_0 and BF16 files are retained separately in
`models/Qwen3-Coder-Next-80B-Unsloth-GGUF/`; these are not the files selected by
the shared presets. The former `Qwen3-Coder-Next-GGUF` weight path remains a
compatibility symlink to each host's original files. The Qwen3-Coder-30B-A3B
preset and downloads were not changed.

Validation checks all expected shard sizes against upstream metadata, GGUF
headers, shard numbers and counts, first-shard architecture and precision,
and Hugging Face download metadata. This is not a GPU inference test or an
independent full-file checksum pass.

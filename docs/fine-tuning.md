# Fine-tuning allocated models

Open **Fine-tuning** in the dashboard, or **7/F7** in `res-mon`. Use a retained GPU allocation. Training unloads inference and pauses its attached chats on all clients; finishing leaves the allocation available for loading a model again. The first implementation trains **LoRA adapters**, using either a BF16 base or a 4-bit QLoRA base. GGUF inference checkpoints cannot be trained directly.

Update the remote checkout as well as the local one. Run `remote/bin/setup-training` on the remote host once. It installs an isolated Python 3.12 environment with Unsloth and document readers; inference environments are unchanged. Alternatively provide an existing compatible Python executable in the training form. The environment must be visible inside the allocation (including containers, if used). Runtime versions are recorded in `remote/venvs/unsloth/installed-versions.txt`.

### Container runtime

Choose **Apptainer container** or **Singularity container** in the training form (also available through F7). Provide the existing image path on the remote host, such as `/containers/unsloth.sif`. The image must contain Unsloth, PyTorch, Transformers and the document readers from `remote/training/requirements.txt`. The optional Python path refers to an executable **inside** the image; the default is `python3`.

Preflight runs inside that image before changing ownership or unloading inference. Training/export uses the allocation’s GPUs (`--nv`), with the repository, allocation state, source folders, local base checkpoint and adapter paths bound at the same paths inside the container. These paths and the container runtime/image must be available on the allocated node. This initial container option targets NVIDIA GPUs through Apptainer/Singularity; it does not launch Docker services or create extra allocations.

The fine-tuning page’s bottom graph follows its selected allocation and has the same time ranges, hover values, scrolling and resizing as the monitoring page.

## Dataset

Enter colon-separated files/folders and choose local or remote paths. Folders are recursive. Local sources are uploaded as a separate snapshot, preserving the existing RAG snapshot. Explicit answer references are rewritten in the copy. Answer files must be within the supplied roots. Symlinks outside them are excluded.

Example `QandA_finetuning.txt`:

```text
What does the detector measure?, How is it calibrated?
[notes/calibration.pdf]
"Why does the signal, after correction, change?"
[results/explanation.txt]
```

Each question trains against the complete extracted answer file. Commas separate questions; quote a question containing commas. Relative paths resolve beside the manifest. Referenced answers are not duplicated as unlabeled examples. Other files become continued-pretraining text. Text, CSV, JSON and code are read directly; PDFs use pypdf; Office and other supported formats use MarkItDown. Scanned documents need OCR first. Unsupported/unreadable unlabeled files are recorded in `dataset-report.json`; an unreadable labeled answer fails the run. Individual files are limited to 128 MiB; split larger inputs first.

Long answers are split into token sequences, rather than silently truncated. File progress counts a source as trained only after all its sequences have participated in an optimizer update. It measures first-pass coverage; steps and epochs show subsequent passes. Fractional epochs may finish before 100% file coverage. Preparation and teacher generation have separate counters.

## Resources, saving and export

Unsloth splits layers across visible GPUs on one allocated node. QLoRA reduces base-weight memory. This is not multi-node training or unlimited-memory offload: if the model still cannot fit, allocate more GPUs or reduce sequence length. CUDA/architecture compatibility must be supported by the installed Unsloth/Transformers versions. The GLM-5.3-Flash and Qwen3.8-Flash-Next workflows have not yet been validated by an actual GPU training run.

**Stop and save weights** requests a stop after the next optimizer step and saves the adapter, tokenizer and trainer state. Regular checkpoints also contain optimizer state. During extraction or loading, stopping may wait for the current operation; no trained weights exist before training starts. Allocation cancellation/node loss can only preserve previously written checkpoints. Keep the base checkpoint alongside the adapter.

Use **Export Q4** (`q4_k_m`) or **Export Q8** (`q8_0`) with the saved adapter directory. Export merges/quantizes through Unsloth's GGUF exporter and needs enough host RAM, GPU memory and disk space. Support depends on the installed exporter/llama.cpp architecture support. Stopping an export is checked before conversion; an in-progress converter is allowed to finish. Outputs and errors appear in the run directory shown in both interfaces, including `training.log`.

## Student–teacher training

Load the teacher (e.g. GLM) in a **separate allocation on the same configured SSH host**, then enter its local session number when starting student training. The teacher endpoint must be reachable from the student's allocated node. The teacher generates responses grounded in document chunks; unlabeled inputs generate question/answer pairs. Responses are saved in `teacher-dataset.jsonl`, then used to train the student. This is response-based distillation, not matching teacher logits. Teacher credentials are kept in a private file, not published in status or run configuration.

## Hydra BF16 downloads

The requested repositories are `unsloth/GLM-5.3-Flash` and `unsloth/Qwen3.8-Flash-Next`. Downloads started under `/lustre/alice/users/csonnab/LLM-AWAY/remote/models/`, in `GLM-5.3-Flash-BF16` and `Qwen3.8-Flash-Next-BF16`. Each contains `download-status.json` and `download.log`; only `COMPLETE` confirms completion. Downloads are revision-pinned and resumable. They do not consume a GPU allocation.

References: [Unsloth multi-GPU training](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth), [GGUF export](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf), [FastModel fine-tuning example](https://unsloth.ai/docs/models/qwen3.8/train).

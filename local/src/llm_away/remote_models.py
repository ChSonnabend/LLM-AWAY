"""Read installed managed models on the SSH host; also sent over SSH on stdin."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys


def gguf_size(model_path):
    """Return the total size only when every shard is readable and nonempty."""
    shards = [model_path]
    split = re.fullmatch(r"(.+)-([0-9]{5})-of-([0-9]{5})\.gguf", model_path.name)
    if split:
        count = int(split[3])
        if count < 1 or int(split[2]) != 1:
            return 0
        shards = [model_path.with_name(f"{split[1]}-{i:05d}-of-{count:05d}.gguf")
                  for i in range(1, count + 1)]
    if not all(file.is_file() and os.access(file, os.R_OK) and file.stat().st_size > 0 for file in shards):
        return 0
    return sum(file.stat().st_size for file in shards)


def installed_models(workdir, models_dir=""):
    root = Path(os.path.expandvars(workdir)).expanduser().resolve()
    library = root / "scripts/lib/llamacpp-env.sh"
    if not library.is_file():
        raise ValueError("Remote model library not found: " + str(library))
    model_root = Path(models_dir).resolve() if models_dir else root / "models"
    models = []
    # Resolve each preset in a fresh shell, just as bin/run-server does.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MODEL", "LLAMACPP_SPEC_")) and key != "LLAMACPP_MTP"}
    env["LLAMACPP_MODELS_DIR"] = str(model_root)
    for preset in sorted(model_root.glob("*/model.env")):
        result = subprocess.run(
            ["bash", "-c", 'set -e; source "$1"; llamacpp_load_model_config "$2"; '
             'path=$(llamacpp_resolve_model); '
             'toggle=no; if declare -F llamacpp_apply_mtp_mode >/dev/null; then toggle=yes; fi; '
             'printf "%s\\0%s\\0%s\\0%s\\0%s\\0%s\\0%s\\0%s" "$MODEL_ALIAS" "$path" '
             '"${MODEL_DRAFT_GGUF:-}" "${LLAMACPP_SPEC_TYPE:-draft-mtp}" '
             '"${LLAMACPP_SPEC_DRAFT_N_MAX:-8}" "$toggle" "${MODEL_CONTEXT_SIZE:-0}" "${MODEL_MTP_EMBEDDED:-0}"',
             "discover-model", str(library), preset.parent.name],
            cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=15,
        )
        if result.returncode:
            continue
        parts = result.stdout.split("\0")
        if len(parts) != 8:
            continue
        alias, path, draft, spec_type, draft_max, toggle, context_size, embedded = parts
        model_path = Path(path)
        size = gguf_size(model_path)
        if not size:
            continue
        draft_path = ((model_root / draft[len("models/"):]) if draft.startswith("models/") else (root / draft)) if draft else None
        draft_size = gguf_size(draft_path) if draft_path else 0
        configured = embedded == '1' or bool(draft) and spec_type == "draft-mtp"
        models.append({"name": preset.parent.name, "alias": alias, "path": path,
                       "size_bytes": size,
                       "context_size": int(context_size) if context_size.isdigit() else 0,
                       "mtp": {"configured": configured,
                               "available": configured and (embedded == '1' or draft_size > 0),
                               "embedded": embedded == '1',
                               "draft_path": str(draft_path) if draft_path else "",
                               "draft_size_bytes": draft_size,
                               "draft_n_max": draft_max,
                               "toggle_supported": toggle == "yes",
                               "spec_type": spec_type if draft else "none"}})
    return models


if __name__ == "__main__":
    try:
        print(json.dumps(installed_models(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

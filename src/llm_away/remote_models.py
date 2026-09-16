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


def installed_models(workdir):
    root = Path(os.path.expandvars(workdir)).expanduser().resolve()
    library = root / "scripts/lib/llamacpp-env.sh"
    if not library.is_file():
        raise ValueError("Remote model library not found: " + str(library))
    models = []
    # Resolve each preset in a fresh shell, just as bin/run-server does.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MODEL", "LLAMACPP_SPEC_")) and key != "LLAMACPP_MTP"}
    for preset in sorted((root / "models").glob("*/model.env")):
        result = subprocess.run(
            ["bash", "-c", 'set -e; source "$1"; llamacpp_load_model_config "$2"; '
             'path=$(llamacpp_resolve_model); '
             'toggle=no; if declare -F llamacpp_apply_mtp_mode >/dev/null; then toggle=yes; fi; '
             'printf "%s\\0%s\\0%s\\0%s\\0%s\\0%s" "$MODEL_ALIAS" "$path" '
             '"${MODEL_DRAFT_GGUF:-}" "${LLAMACPP_SPEC_TYPE:-draft-mtp}" '
             '"${LLAMACPP_SPEC_DRAFT_N_MAX:-8}" "$toggle"',
             "discover-model", str(library), preset.parent.name],
            cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=15,
        )
        if result.returncode:
            continue
        alias, path, draft, spec_type, draft_max, toggle = result.stdout.split("\0")
        model_path = Path(path)
        size = gguf_size(model_path)
        if not size:
            continue
        draft_path = (root / draft) if draft else None
        draft_size = gguf_size(draft_path) if draft_path else 0
        configured = bool(draft) and spec_type == "draft-mtp"
        models.append({"name": preset.parent.name, "alias": alias, "path": path,
                       "size_bytes": size,
                       "mtp": {"configured": configured,
                               "available": configured and draft_size > 0,
                               "draft_path": str(draft_path) if draft_path else "",
                               "draft_size_bytes": draft_size,
                               "draft_n_max": draft_max,
                               "toggle_supported": toggle == "yes",
                               "spec_type": spec_type if draft else "none"}})
    return models


if __name__ == "__main__":
    try:
        print(json.dumps(installed_models(sys.argv[1])))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

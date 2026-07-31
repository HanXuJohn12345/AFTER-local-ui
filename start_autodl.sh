#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"

python - <<'PY'
import torch
print(f"cuda available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
PY

python after_local_ui.py --host "$HOST" --port "$PORT"
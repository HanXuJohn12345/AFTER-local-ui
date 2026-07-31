#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python -m pip install --upgrade pip
python -m pip install -r requirements-autodl-ui.txt

python - <<'PY'
from pathlib import Path
import torch
import torchaudio
import numpy as np

print(f"torch: {torch.__version__}")
print(f"torchaudio: {torchaudio.__version__}")
print(f"numpy: {np.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Use an AutoDL PyTorch CUDA image or install matching CUDA torch/torchaudio.")
print(f"gpu: {torch.cuda.get_device_name(0)}")
model = Path("pretrained/afterv2.audio.instr.ts")
if not model.exists():
    raise SystemExit(f"Missing model file: {model}")
print(f"model: {model}")
PY
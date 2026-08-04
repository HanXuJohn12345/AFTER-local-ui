# AutoDL Deployment Guide

This guide deploys the AFTER Local UI on an AutoDL Linux GPU instance. The model runs on AutoDL. A Windows laptop or MacBook only opens the browser UI or an SSH tunnel.

## Feasibility

This is feasible. The UI accepts `--host` and `--port`, so AutoDL should run it on `0.0.0.0:6006` and expose that port through AutoDL custom service or SSH port forwarding.

Current limitation: the UI is single-process and has one live model state. It is good for one player or several people taking turns. It is not designed for many simultaneous live performers.

## Recommended AutoDL Image

Choose an image that already includes CUDA PyTorch:

- Python 3.10 to 3.12
- PyTorch 2.x + torchaudio 2.x
- CUDA visible through `torch.cuda.is_available()`
- 16 GB VRAM minimum for basic tests, 24 GB+ recommended

The UI runtime mainly needs `torch`, `torchaudio`, and `numpy`. Avoid installing the full training stack unless you are training models.

## Required Files

Model weights are not committed to GitHub. Download the official pretrained model files first:

https://nubo.ircam.fr/index.php/s/8NFD5gWwbkT4G5P

The AutoDL project folder should contain at least:

```text
after_local_ui.py
pretrained/afterv2.audio.instr.ts
pretrained/afterv2.audio.instr.data335.range
pretrained/afterv2.audio.instr.data336.range
pretrained/afterv2.audio.instr.data337-342.range
pretrained/afterv2.audio.instr.png
install_autodl.sh
start_autodl.sh
requirements-autodl-ui.txt
benchmark_live_autodl.py
```

## Adding More Trained Instruments / Models

The UI scans local TorchScript exports in `pretrained/*.ts`. To add trained instrument or timbre models, copy the exported `.ts` files into the AutoDL instance's `pretrained/` folder and restart the UI.

Example:

```text
pretrained/afterv2.audio.instr.ts
pretrained/afterv2.audio.guitar.ts
pretrained/afterv2.audio.voice.ts
pretrained/afterv2.audio.choirs.ts
```

If a matching `.png` exists next to the model, the 2D timbre map image switches with the model:

```text
pretrained/afterv2.audio.guitar.png
```

The browser cannot directly browse arbitrary model paths from your laptop. It can only choose files that already exist on the server machine.

## Upload / Enter Project

Assume the project is placed at `/root/AFTER`:

```bash
cd /root/AFTER
```

If scripts are not executable:

```bash
chmod +x install_autodl.sh start_autodl.sh
```

## Install / Check

```bash
bash install_autodl.sh
```

A successful check prints PyTorch, torchaudio, numpy, CUDA status, GPU name, and model path.

## Start Service

```bash
bash start_autodl.sh
```

Default address inside AutoDL:

```text
0.0.0.0:6006
```

To use another port:

```bash
PORT=6008 bash start_autodl.sh
```

## Access

Use AutoDL custom service / port mapping for instance port `6006`. The custom service address is publicly reachable, so use it for private tests only.

Alternatively, use SSH tunneling:

```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@YOUR_AUTODL_HOST -p YOUR_SSH_PORT
```

Then open locally:

```text
http://127.0.0.1:6006/
```

## MacBook Access

AutoDL deployment does not mean deploying the model to a MacBook. The model, PyTorch, CUDA, and GPU inference run on the AutoDL Linux instance. The MacBook is only the client: browser, microphone permission, audio playback, and optional SSH tunnel.

On macOS Terminal, keep this tunnel running:

```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@YOUR_AUTODL_HOST -p YOUR_SSH_PORT
```

Then open Chrome or Safari on the MacBook:

```text
http://127.0.0.1:6006/
```

Allow microphone permission if using `Start Live`. Audio capture happens in the MacBook browser, while AFTER inference happens on the AutoDL GPU.

Running this model directly on a MacBook is not recommended. Regular MacBooks do not have CUDA, and Apple Silicon MPS may not be compatible with this exported real-time TorchScript path.

## Debugging MacBook `Load failed`

If `Run AFTER` or `Start Live` shows only `Load failed`, the browser did not receive a usable response. In Safari this often hides the real cause, so check these in order:

1. Open the same UI address plus `/health`:

```text
http://127.0.0.1:6006/health
```

It should show `"cuda_available": true` and a real NVIDIA device name. If it says CPU, the AutoDL image or PyTorch install is wrong and live inference will likely time out.

2. Prefer an SSH tunnel from the MacBook:

```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@YOUR_AUTODL_HOST -p YOUR_SSH_PORT
```

Then open:

```text
http://127.0.0.1:6006/
```

Microphone access is reliable on `localhost`. A plain remote `http://...` address may block microphone capture in macOS browsers. HTTPS custom service can also work.

3. Keep the AutoDL service log visible:

```bash
bash start_autodl.sh 2>&1 | tee after_ui.log
```

When a backend error happens, the UI now prints a Python traceback in this log and the browser shows the failing API name, such as `Run AFTER`, `Live reset`, or `Live chunk`.

4. For first tests, use conservative settings:

```text
Buffer Size: 8192
Live Quality: Fast / nb_steps=1
Post FX: off
Input: short WAV file first
```

MP3/M4A upload can fail if the AutoDL torchaudio build lacks FFmpeg support. Use WAV first when debugging `Run AFTER`.

5. Benchmark the live path on AutoDL:

```bash
python benchmark_live_autodl.py --steps 1 --buffer-size 8192 --chunks 5
```

If p95 is above the buffer duration, the browser/proxy may report a network-style failure even though the real problem is slow inference.

## Health Check

```bash
curl http://127.0.0.1:6006/health
```

## GPU Benchmark

Run this after deployment:

```bash
python benchmark_live_autodl.py --steps 1,2,4,6 --buffer-size 4096 --chunks 8
```

The UI can choose `2048 / 4096 / 8192` buffer size. At 44.1 kHz, these are about `46.4 ms / 92.9 ms / 185.8 ms` of audio.

Interpretation:

- p95 below the selected buffer duration: basically real-time.
- p95 close to the selected buffer duration: usable but sensitive to browser/network/system jitter.
- p95 above the selected buffer duration: latency will accumulate and it will feel stuck or delayed.

## nb_steps GPU Guidance

`nb_steps` scales inference cost roughly linearly. `nb_steps=6` can be about 5 to 6 times heavier than `nb_steps=1`.

Practical guidance:

- T4 / 2060 / 3060: offline or `nb_steps=1` tests only.
- RTX 3090 / RTX 4090: good for single-user live tests; `nb_steps=1/2` is realistic, `6` must be benchmarked.
- A40 / L40 / L40S: better for service deployment; `2/4` is more comfortable.
- A100 / H100 / H800: recommended if you want `nb_steps=6` with low latency.

Final judgment should come from `benchmark_live_autodl.py`: p95 should be below the selected buffer duration.

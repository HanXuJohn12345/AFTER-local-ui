# AFTER Local and AutoDL UI

This repository is a deployment-focused fork of [acids-ircam/AFTER](https://github.com/acids-ircam/AFTER). It adds a browser UI for local Windows testing and AutoDL GPU deployment.

AFTER is a diffusion-based audio-to-audio transfer model. This UI keeps the exported model's original audio-to-audio behavior, then adds performance-friendly controls and optional post-processing for live playing.

## What This Version Adds

- Browser UI: record microphone audio, upload audio, run offline inference, or use live microphone streaming.
- GPU support: uses CUDA automatically when PyTorch sees a GPU.
- Model / Instrument selector: choose any local exported `.ts` model placed in `pretrained/`.
- Timbre controls: original 2D XY map for coarse control plus six 6D latent sliders for fine control.
- Performance controls: `nb_steps`, `guidance_structure`, `Live Quality`.
- Performance/playability controls: `Buffer Size`, `Input Gain`, `Wet / Dry`, `Morph Speed`, `Timbre Presets`.
- Post-processing panel: Pedal Reverb, Pedal Level, Delay.
- AutoDL deployment scripts: run the same UI on an AutoDL GPU instance.

## Model Download

Model weights are not committed to this repository because `afterv2.audio.instr.ts` is about 220 MB and exceeds normal GitHub file limits.

Download the official pretrained AFTER models and Max patches here:

[AFTER pretrained models and Max patches](https://nubo.ircam.fr/index.php/s/8NFD5gWwbkT4G5P)

For this UI, download the audio-to-audio exported model files and place them under `pretrained/` like this:

```text
pretrained/afterv2.audio.instr.ts
pretrained/afterv2.audio.instr.data335.range
pretrained/afterv2.audio.instr.data336.range
pretrained/afterv2.audio.instr.data337-342.range
pretrained/afterv2.audio.instr.png
```

The `.png` is used by the restored 2D timbre map UI.

## Adding More Trained Instruments / Models

The UI scans local TorchScript exports in `pretrained/*.ts`. To add a trained instrument or timbre model, copy its exported `.ts` file into `pretrained/` and restart the UI.

Example:

```text
pretrained/afterv2.audio.instr.ts
pretrained/afterv2.audio.guitar.ts
pretrained/afterv2.audio.voice.ts
pretrained/afterv2.audio.choirs.ts
```

If a matching `.png` exists next to the model, the 2D timbre map image will switch with the model:

```text
pretrained/afterv2.audio.guitar.png
```

The browser can only choose files that already exist on the server machine. It cannot directly browse arbitrary model paths from your laptop for security reasons.

## Option 1: Local Windows Deployment

Use this when running on a local Windows machine with an NVIDIA GPU.

### Requirements

- Windows
- NVIDIA GPU with a CUDA-enabled PyTorch install
- Python environment with `torch`, `torchaudio`, and `numpy`
- Model files placed in `pretrained/`

### Start

```powershell
cd C:\Users\hanxu\AFTER
C:\Users\hanxu\miniconda3\envs\after_gpu\python.exe after_local_ui.py --host 127.0.0.1 --port 7860
```

Or use the helper script:

```powershell
.\start_local_windows.ps1
```

Then open:

```text
http://127.0.0.1:7860/
```

Health check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7860/health
```

More detail: [LOCAL_UI_DEPLOY.md](LOCAL_UI_DEPLOY.md)

## Option 2: AutoDL GPU Deployment

Use this when running on an AutoDL cloud GPU instance and sharing the UI through the browser.

### Recommended AutoDL Image

Choose an image that already includes CUDA PyTorch and torchaudio:

- Python 3.10 to 3.12
- PyTorch 2.x + torchaudio 2.x
- CUDA visible through `torch.cuda.is_available()`
- 16 GB VRAM minimum for basic testing; 24 GB+ recommended

### Upload Files

Put this repository and the downloaded model files on the AutoDL instance, for example:

```bash
/root/AFTER
```

Make sure model files are in:

```text
/root/AFTER/pretrained/
```

### Install / Check

```bash
cd /root/AFTER
chmod +x install_autodl.sh start_autodl.sh
bash install_autodl.sh
```

### Start

```bash
bash start_autodl.sh
```

Default service address inside AutoDL:

```text
0.0.0.0:6006
```

Use AutoDL custom service / port mapping for port `6006`, or use SSH tunnel:

```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@YOUR_AUTODL_HOST -p YOUR_SSH_PORT
```

Then open locally:

```text
http://127.0.0.1:6006/
```

### Access from a MacBook

The AutoDL model server still runs on the AutoDL Linux GPU machine. A MacBook is only the client: browser, microphone permission, and optional SSH tunnel. You do not need CUDA, PyTorch, or the model weights on the MacBook just to use the AutoDL UI.

On macOS Terminal, keep this tunnel running:

```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@YOUR_AUTODL_HOST -p YOUR_SSH_PORT
```

Then open Chrome or Safari on the MacBook:

```text
http://127.0.0.1:6006/
```

Allow microphone permission in the browser if using `Start Live`.

More detail: [AUTODL_DEPLOY.md](AUTODL_DEPLOY.md)

## AutoDL Performance Test

After deployment, run:

```bash
python benchmark_live_autodl.py --steps 1,2,4,6 --buffer-size 4096 --chunks 8
```

The default live buffer is `4096` samples at `44100 Hz`, about `92.9 ms` of audio. This exported model has been verified with `4096` and `8192` sample buffers. `2048` is not exposed because the TorchScript encoder can fail on that shorter window. For real-time performance, benchmark `p95` latency should stay below the selected buffer duration: `4096` = about `92.9 ms`, `8192` = about `185.8 ms`.

Rough guidance:

- RTX 3090 / RTX 4090: good for single-user live testing; `nb_steps=1/2` is realistic, `6` needs benchmarking.
- L40 / L40S / A40: better for deployment; `nb_steps=2/4` is more comfortable.
- A100 / H100 / H800: recommended if you want `nb_steps=6` to stay low-latency.
- T4 / small consumer GPUs: use offline generation or `nb_steps=1` only.

## Important Notes

- This UI is single-process and has one live model state. It is good for one performer or several people taking turns. It is not designed for many simultaneous live users.
- Model weights are intentionally external. Use the official download link above, Git LFS, Hugging Face, a release asset, or an AutoDL data disk if you want a fully packaged deployment.
- The original AFTER project, training guide, paper links, and Max/MSP devices are maintained upstream at [acids-ircam/AFTER](https://github.com/acids-ircam/AFTER).

## Credits

Original AFTER project by Nils Demerle, P. Esling, G. Doras, and D. Genova from ACIDS / IRCAM.

If you use AFTER in research or performance, cite the original project and paper:

[Combining audio control and style transfer using latent diffusion](https://arxiv.org/abs/2408.00196)

# AFTER Local UI - Windows PC Deployment

This guide is for running the AFTER Local UI on a Windows PC with an NVIDIA GPU, such as an RTX 4090 or RTX 5090.

The model and UI run locally on the PC GPU. Other laptops only open the browser if the PC exposes the port on the same network.

## 1. Clone The Repository

```powershell
git clone https://github.com/HanXuJohn12345/AFTER-local-ui.git
cd AFTER-local-ui
git switch local-ui
```

## 2. Install Miniconda

Install Miniconda for Windows:

```text
https://docs.conda.io/projects/miniconda/en/latest/
```

Open **Anaconda Prompt** or **PowerShell** after installation.

## 3. Create The Environment

```powershell
conda create -n after_gpu python=3.10 -y
conda activate after_gpu
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4
```

Install CUDA-enabled PyTorch from the official selector:

```text
https://pytorch.org/get-started/locally/
```

For a new GPU such as RTX 5090, use the newest NVIDIA driver and the newest CUDA-enabled PyTorch wheel available from the official PyTorch install page. Do not use a CPU-only PyTorch package.

Then install/check torchaudio matching the installed PyTorch version.

## 4. Download Model Files

Model weights are not stored in GitHub. Download the official AFTER pretrained files:

```text
https://nubo.ircam.fr/index.php/s/8NFD5gWwbkT4G5P
```

Place these files under:

```text
pretrained/
```

Minimum expected files:

```text
pretrained/afterv2.audio.instr.ts
pretrained/afterv2.audio.instr.data335.range
pretrained/afterv2.audio.instr.data336.range
pretrained/afterv2.audio.instr.data337-342.range
pretrained/afterv2.audio.instr.png
```

## 5. Verify GPU

```powershell
conda activate after_gpu
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected result:

```text
True
NVIDIA GeForce RTX 5090
```

The exact GPU name may vary. If `torch.cuda.is_available()` is `False`, fix the NVIDIA driver / PyTorch CUDA installation before starting the UI.

## 6. Start The UI

Option A: double-click:

```text
start_windows_pc.bat
```

Option B: run manually:

```powershell
conda run -n after_gpu python after_local_ui.py --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860/
```

Health check:

```text
http://127.0.0.1:7860/health
```

The health check should show `cuda_available: true` and the NVIDIA GPU name.

## 7. Recommended First Settings

```text
Buffer Size: 4096
Live Quality: Fast
nb_steps: 1
guidance_structure: 1.0
Pedal Reverb: 10-25%
Pedal Tail: 0.35-0.55
Pedal Level: 1.0x
Delay Mix: 0%
```

Use `8192` if the audio crackles or if the browser/network path is unstable. `4096` is lower latency.

## Notes

- Keep the PowerShell/terminal window open while using the UI. Closing it stops the backend.
- If the page opens but buttons show `Failed to fetch`, the backend is not running or the wrong port is open.
- The live UI is for one performer/session at a time.
- The exported model currently exposes `4096` and `8192` live buffer sizes. `2048` is not used because this TorchScript model can fail on that shorter live window.

# AFTER 本地 UI 部署说明

这个版本是在原 AFTER 项目上增加的本地浏览器 UI，用来测试 `pretrained/afterv2.audio.instr.ts`，支持：

- 麦克风录音或上传音频
- 实时 `Start Live`
- 6 维 timbre latent 滑块
- Input Gain / Wet Dry / Morph Speed / Live Quality
- Timbre Presets
- 后处理：Spring Reverb、Reverb Boost、Delay

## 目录要求

模型权重默认不提交到 GitHub。运行前需要本地有：

```text
pretrained/afterv2.audio.instr.ts
pretrained/afterv2.audio.instr.data335.range
pretrained/afterv2.audio.instr.data336.range
pretrained/afterv2.audio.instr.data337-342.range
pretrained/afterv2.audio.instr.png
```

## Windows 本地启动

建议使用已经装好 CUDA PyTorch 的 conda 环境，例如这里的 `after_gpu`：

```powershell
cd C:\Users\hanxu\AFTER
C:\Users\hanxu\miniconda3\envs\after_gpu\python.exe after_local_ui.py --host 127.0.0.1 --port 7860
```

也可以直接运行：

```powershell
.\start_local_windows.ps1
```

浏览器打开：

```text
http://127.0.0.1:7860/
```

健康检查：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7860/health
```

## GPU

启动时会自动使用 CUDA：

```python
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
```

如果 `/health` 里 `cuda_available` 是 `true`，说明会走 GPU。

## GitHub 权重说明

`afterv2.audio.instr.ts` 大约 220MB，超过 GitHub 普通文件 100MB 限制。推荐两种做法：

1. 默认做法：GitHub 只放代码和部署说明，权重通过网盘、Release、Hugging Face 或 AutoDL 数据盘单独放。
2. Git LFS 做法：如果确定要把权重也放 GitHub，需要启用 Git LFS，并把 `pretrained/*.ts` 和 `pretrained/*.range` 作为 LFS 文件提交。
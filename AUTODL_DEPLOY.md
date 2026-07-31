# AutoDL 部署说明

这个版本用于把本地 AFTER UI 部署到 AutoDL GPU 实例上，让其他人通过浏览器访问同一个 GPU 服务。

## 可行性结论

可行。当前 UI 本身已经支持命令行指定 `--host` 和 `--port`，AutoDL 上只需要监听 `0.0.0.0:6006`，再通过 AutoDL 的自定义服务或 SSH 端口转发访问。

需要注意：当前 UI 是单进程、单模型、单实时状态设计。它适合一个人演奏/测试，或多人轮流使用；不适合很多人同时点 `Start Live`。如果要多人同时实时演奏，应该开多个 AutoDL 实例，或者以后把 live state 改成按 session 隔离。

## 推荐 AutoDL 镜像

选择已经带 CUDA PyTorch 的镜像，建议：

- Python 3.10 到 3.12
- PyTorch 2.x + torchaudio 2.x
- CUDA 11.8/12.1/12.4 均可，重点是 `torch.cuda.is_available()` 必须为 True
- 显存建议 16GB 起步，24GB 更稳

这个 UI 推理主要需要 `torch`、`torchaudio`、`numpy`。不建议在 AutoDL 上从零安装完整训练依赖，容易被 PyTorch/CUDA 版本拖慢。

## 必需文件

仓库里至少要有：

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

## 上传/进入项目

假设项目放在 AutoDL 的 `/root/AFTER`：

```bash
cd /root/AFTER
```

如果脚本没有执行权限：

```bash
chmod +x install_autodl.sh start_autodl.sh
```

## 安装/检查

```bash
bash install_autodl.sh
```

成功时会打印 PyTorch、torchaudio、numpy、CUDA 状态和 GPU 名称。

## 启动服务

```bash
bash start_autodl.sh
```

默认监听：

```text
0.0.0.0:6006
```

也可以改端口：

```bash
PORT=6008 bash start_autodl.sh
```

## 访问

在 AutoDL 控制台里使用自定义服务/端口映射访问实例内的 `6006` 端口。注意 AutoDL 自定义服务是公网可访问地址，且平台协议对用途和链接转发有限制；只建议做私下测试，不要公开发布服务链接。

如果不用控制台映射，也可以用 SSH tunnel：

```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@你的AutoDL主机 -p 你的SSH端口
```

然后本地浏览器打开：

```text
http://127.0.0.1:6006/
```

健康检查：

```bash
curl http://127.0.0.1:6006/health
```

## GPU 实时性能测试

部署后先跑：

```bash
python benchmark_live_autodl.py --steps 1,2,4,6 --chunks 8
```

它会输出每个 `nb_steps` 的平均耗时和 p95 耗时。当前 live chunk 是 4096 samples，44.1kHz 下约等于 92.9 ms 音频。判断标准：

- p95 < 92.9 ms：基本能实时不堆积
- p95 接近 92.9 ms：能跑但容易因为浏览器/网络/系统抖动卡
- p95 > 92.9 ms：会逐渐堆积延迟，也就是听感上“卡”或越来越慢

## nb_steps 算力建议

`nb_steps` 基本近似线性增加推理耗时。`nb_steps=6` 大致可以按 `nb_steps=1` 的 5 到 6 倍预算。

实用建议：

- T4 / 2060 / 3060：不建议做实时演奏，只适合离线或 nb_steps=1 轻测试
- RTX 3090 / RTX 4090：适合单人实时测试，nb_steps=1/2 比较现实；nb_steps=6 要实际 benchmark，不保证稳
- A40 / L40 / L40S：更适合长时间服务，nb_steps=2/4 更稳，nb_steps=6 仍需 benchmark
- A100 / H100 / H800：更适合追求 nb_steps=6 还想低延迟的场景

如果目标是“nb_steps 拉满 6 还不卡”，建议至少从 RTX 4090 / L40S 级别开始试；更保守就是 A100/H100/H800。最终以 `benchmark_live_autodl.py` 的 p95 是否低于 92.9 ms 为准。
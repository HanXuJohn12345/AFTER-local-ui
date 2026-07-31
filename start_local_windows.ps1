param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7860,
    [string]$Python = "C:\Users\hanxu\miniconda3\envs\after_gpu\python.exe"
)

Set-Location $PSScriptRoot

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python"
}

& $Python "$PSScriptRoot\after_local_ui.py" --host $HostAddress --port $Port
@echo off
setlocal
cd /d "%~dp0"

echo Starting AFTER Local UI on http://127.0.0.1:7860
echo Keep this window open while using the UI.
echo.

where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda was not found in PATH.
  echo Open Anaconda Prompt, activate after_gpu once, or add Miniconda to PATH.
  pause
  exit /b 1
)

conda run -n after_gpu python after_local_ui.py --host 127.0.0.1 --port 7860

echo.
echo AFTER UI stopped.
pause

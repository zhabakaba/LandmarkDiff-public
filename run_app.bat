@echo off
setlocal
cd /d "%~dp0"

set "GRADIO_ANALYTICS_ENABLED=False"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Create it first:
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -e ".[app]"
  exit /b 1
)

"%CD%\.venv\Scripts\python.exe" app_before_after.py %*

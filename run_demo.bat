@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Engine Lab virtual environment is missing.
  echo Run the Phase 0 setup commands from README.md.
  pause
  exit /b 1
)
start "Engine Lab" ".venv\Scripts\pythonw.exe" -m app.main --source loop --scenario retail --mode spread --density 1 --fullscreen %*
exit /b 0

@echo off
REM ============================================================================
REM Engine Lab - START THE DEMO
REM
REM Just double-click this file. Nothing to install first.
REM
REM This bundle carries its own Python and its own pinned packages, so it does
REM not matter whether Python is installed on this machine, and it does not
REM matter which version is on PATH. It also needs no network: the models and
REM the video files are all inside this folder.
REM
REM Move or rename this folder freely - every path here is relative to it.
REM ============================================================================
setlocal
cd /d "%~dp0"

if not exist "%~dp0python\python.exe" (
    echo.
    echo ERROR: the bundled Python is missing from this folder.
    echo Expected: %~dp0python\python.exe
    echo The folder was probably not copied in full.
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0models\Qwen3-1.7B-int4-ov\openvino_model.bin" (
    echo.
    echo ERROR: the model files are missing from this folder.
    echo The folder was probably not copied in full.
    echo.
    pause
    exit /b 1
)

start "Engine Lab" "%~dp0python\pythonw.exe" -m app.main --source loop --scenario retail --mode spread --density 1 %*
exit /b 0

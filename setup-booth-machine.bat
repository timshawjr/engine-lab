@echo off
REM ============================================================================
REM Engine Lab - one-time booth machine setup.
REM
REM Run this ONCE on a new machine. It creates the pinned virtual environment
REM and installs the OpenVINO / PySide6 dependencies. Network is needed for
REM this step only; the demo itself never touches the network.
REM
REM Does NOT require administrator rights.
REM ============================================================================
setlocal
cd /d "%~dp0"

echo.
echo === Engine Lab booth setup ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: python was not found on PATH.
    echo Install Python 3.12 from python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Python found: %PYVER%

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Creating the pinned virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists; reusing it.
)

echo.
echo Installing pinned dependencies. This takes a few minutes...
echo The openvino pin must stay at 2026.4.0 - do not "upgrade" it.
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: dependency install failed.
    pause
    exit /b 1
)

echo.
echo Verifying the install...
.venv\Scripts\python.exe -c "import openvino, openvino_genai, PySide6; print('openvino', openvino.__version__); print('genai   ', openvino_genai.__version__)"
if errorlevel 1 (
    echo ERROR: imports failed after install.
    pause
    exit /b 1
)

echo.
echo === Next: models ===
echo The demo needs its model files, which are large and are NOT in this package.
echo Run:  download-models.bat
echo.
echo Then run:  verify-booth.bat   to prove the machine is booth-ready.
echo.
pause
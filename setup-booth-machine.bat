@echo off
REM ============================================================================
REM Engine Lab - one-time booth machine setup.
REM
REM Run this ONCE on a new machine. It creates the pinned virtual environment
REM and installs the OpenVINO / PySide6 dependencies. Network is needed for
REM this step only; the demo itself never touches the network.
REM
REM Requires Python 3.12. Not 3.11, not 3.13. The app checks this itself and
REM the pinned wheels are built for 3.12.
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
    echo.
    echo Install Python 3.12 from:
    echo     https://www.python.org/downloads/release/python-31210/
    echo During install, TICK "Add python.exe to PATH", then run this again.
    echo.
    pause
    exit /b 1
)

REM The version matters: the app asserts 3.12 at startup and the pinned
REM OpenVINO wheels are built for it. Checking here turns a confusing pip
REM failure into a clear instruction.
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

echo Python found: %PYVER%

if not "%PYMAJOR%"=="3" goto :badversion
if not "%PYMINOR%"=="12" goto :badversion
goto :versionok

:badversion
echo.
echo ERROR: Python %PYVER% is not supported. This project requires Python 3.12.
echo.
echo   - the application checks the version at startup and reports FAIL otherwise
echo   - the pinned OpenVINO and PySide6 wheels are built for 3.12
echo.
echo Install Python 3.12 from:
echo     https://www.python.org/downloads/release/python-31210/
echo TICK "Add python.exe to PATH" during install, then run this again.
echo.
echo If another Python is already on PATH, install 3.12 and run this script with
echo that interpreter, for example:
echo     "%%LOCALAPPDATA%%\Programs\Python\Python312\python.exe" -m venv .venv
echo.
pause
exit /b 1

:versionok
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
    echo If it was built with a different Python, delete .venv and run this again.
)

echo.
echo Installing pinned dependencies. This takes a few minutes...
echo The openvino pin must stay at 2026.4.0 - do not "upgrade" it.
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: dependency install failed.
    echo Check the messages above. requirements.txt pins specific versions.
    pause
    exit /b 1
)

echo.
echo Verifying the install...
.venv\Scripts\python.exe -c "import sys, openvino, openvino_genai, PySide6; print('python  ', sys.version.split()[0]); print('openvino', openvino.__version__); print('genai   ', openvino_genai.__version__)"
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
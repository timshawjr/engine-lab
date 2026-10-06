@echo off
REM ============================================================================
REM Engine Lab - VERIFY THIS COPY BEFORE THE SHOW
REM
REM Double-click this, read the output, and check for "0 FAIL" twice.
REM Nothing is installed and no network is used.
REM ============================================================================
setlocal
cd /d "%~dp0"

if not exist "%~dp0python\python.exe" (
    echo ERROR: bundled Python missing. Copy the folder again, in full.
    pause
    exit /b 1
)

echo.
echo === what this copy thinks it is running ===
"%~dp0python\python.exe" -c "import sys, openvino, openvino_genai; print('python  ', sys.version.split()[0]); print('prefix  ', sys.prefix); print('openvino', openvino.__version__); print('genai   ', openvino_genai.__version__)"
echo.

echo === GPU / NPU driver versions on THIS machine ===
"%~dp0python\python.exe" -c "from app.telemetry.devices_win import enumerate_display_adapters, is_intel; [print('  ', d.name, '| driver', getattr(d, 'driver_version', '?')) for d in enumerate_display_adapters() if is_intel(d)]" 2>nul
echo.

echo === preflight: models, devices, placement ===
"%~dp0python\python.exe" tools\preflight.py
echo.

echo === warming the device-availability cache ===
echo This machine has not probed its models yet, and the self-test below needs
echo that result. It takes a few seconds the first time only.
if not exist "%~dp0cache\availability.json" (
    "%~dp0python\python.exe" -m app.main --scenario retail --density 1 --exit-after 20 >nul 2>nul
    echo   cache written.
) else (
    echo   cache already present.
)
echo.

echo === self-test (no network) ===
"%~dp0python\python.exe" -m app.main --selftest
echo.

echo ============================================================================
echo Ready if preflight says "0 WARN, 0 FAIL" and the self-test says "0 FAIL".
echo.
echo NOTE: bench_report.md and SALES-GUIDE.md quote figures measured on the
echo development machine (NPU driver 32.0.100.5540, GPU driver 32.0.101.6737).
echo If this machine reports different drivers, re-measure before quoting them.
echo ============================================================================
pause

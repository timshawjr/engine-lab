@echo off
REM ============================================================================
REM Engine Lab - prove this machine is booth-ready BEFORE the show.
REM
REM Run this on the booth machine and read the summary. Every row must PASS.
REM A FAIL here is a problem to fix at the venue, not in front of a customer.
REM
REM Do NOT ship a copied OpenVINO cache between machines: it is driver
REM specific. This machine builds its own on first run.
REM ============================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: run setup-booth-machine.bat first.
    pause
    exit /b 1
)

echo.
echo === 1/4  sources verified ===
.venv\Scripts\python.exe tools\verify_sources.py
echo.

echo === 2/4  telemetry map ===
.venv\Scripts\python.exe tools\probe_telemetry.py
echo.

echo === 3/4  preflight (models, devices, placement, rag) ===
.venv\Scripts\python.exe tools\preflight.py
echo.

echo === 4/4  self-test (no network) ===
.venv\Scripts\python.exe -m app.main --selftest
echo.

echo ============================================================================
echo If step 3 says "0 WARN, 0 FAIL" and step 4 says "0 FAIL", this machine is
echo ready. Start the demo with run_demo.bat, then press R for document Q&A.
echo.
echo Numbers in bench_report.md were measured on the development machine
echo (Core Ultra9 288V, NPU driver 32.0.100.5540, GPU driver 32.0.101.6737).
echo If this machine reports different drivers, re-measure before quoting them.
echo ============================================================================
pause
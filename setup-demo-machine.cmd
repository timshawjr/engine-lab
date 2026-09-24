@echo off
setlocal
title Engine Lab - demo machine setup (cmd version)

rem ===========================================================================
rem  setup-demo-machine.cmd - bootstrap WITHOUT PowerShell scripts.
rem  Use this when PowerShell script execution is blocked on the machine.
rem  Batch files are not governed by PowerShell ExecutionPolicy.
rem
rem  What it does: checks the Windows build, installs Python 3.12 + Git +
rem  VC++ redistributable via winget, creates .venv, installs requirements.
rem  Everything else (driver checks, telemetry probe, model download,
rem  preflight) is Python and runs through python.exe - unaffected by
rem  script policy.
rem
rem  Run it by double-clicking, or from cmd.exe:
rem      cd C:\dev\engine-lab
rem      setup-demo-machine.cmd
rem  Safe to re-run.
rem ===========================================================================

cd /d "%~dp0"
echo.
echo Engine Lab - demo machine setup (cmd version)
echo   repo: %CD%
echo.

rem ---------------------------------------------------------------- 1. build
echo === Windows version
set "BUILD="
for /f "tokens=3" %%a in ('reg query "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion" /v CurrentBuildNumber 2^>nul ^| findstr /i CurrentBuildNumber') do set "BUILD=%%a"
if not defined BUILD (
  echo   [WARN]  could not read the Windows build number
) else (
  echo   build %BUILD%
  if %BUILD% LSS 26100 (
    echo   [FAIL]  Windows 11 24H2 ^(build 26100+^) is required - run Windows Update.
  ) else (
    echo   [ OK ]  Windows 11 24H2 or later
  )
)
echo.

rem ---------------------------------------------------------------- 2. tools
echo === Developer tools
where winget >nul 2>&1
if errorlevel 1 goto nowinget

echo   installing Python 3.12 ...
winget install --id Python.Python.3.12 -e --silent --scope user --accept-package-agreements --accept-source-agreements
echo   installing Git ...
winget install --id Git.Git -e --silent --accept-package-agreements --accept-source-agreements
echo   installing VC++ 2015-2022 Redistributable x64 ...
winget install --id Microsoft.VCRedist.2015+.x64 -e --silent --accept-package-agreements --accept-source-agreements

rem refresh PATH from the registry so the new tools are visible in this window
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%ProgramFiles%\Git\cmd"
goto havewinget

:nowinget
echo   [WARN]  winget not found.
echo           Option A: install "App Installer" from the Microsoft Store, then re-run this file.
echo           Option B: install Python 3.12 from https://www.python.org/downloads/windows/
echo                     ^(tick "Add python.exe to PATH" in the installer^) and Git from
echo                     https://git-scm.com/download/win, then re-run this file.
echo           Option C: no-admin route with uv - see README section "If scripts and installers are blocked".

:havewinget
echo.
where py >nul 2>&1
if errorlevel 1 (
  echo   [WARN]  py launcher not on PATH - open a NEW cmd window after installing Python, then re-run.
) else (
  echo   [ OK ]  py launcher present
)
where python >nul 2>&1
if errorlevel 1 (echo   [WARN]  python not on PATH yet - open a NEW cmd window) else (echo   [ OK ]  python on PATH)
where git >nul 2>&1
if errorlevel 1 (echo   [WARN]  git not on PATH yet - open a NEW cmd window) else (echo   [ OK ]  git on PATH)
echo.

rem ---------------------------------------------------------------- 3. venv
echo === Python virtual environment
if exist ".venv\Scripts\python.exe" (
  echo   [ OK ]  .venv already exists
  goto deps
)
py -3.12 -m venv .venv 2>nul
if exist ".venv\Scripts\python.exe" goto venvok
python -m venv .venv 2>nul
if exist ".venv\Scripts\python.exe" goto venvok
echo   [FAIL]  could not create .venv - is Python installed and on PATH in a NEW cmd window?
goto summary

:venvok
echo   [ OK ]  created .venv

:deps
if not exist "requirements.txt" (
  echo   [WARN]  requirements.txt not found - the coding agent creates it in Phase 0 ^(SPEC.md 13.3^).
  goto summary
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo   [FAIL]  pip install failed - see the output above.
) else (
  echo   [ OK ]  requirements installed
)
".venv\Scripts\python.exe" -c "import openvino as ov; print('  [ OK ]  openvino', ov.__version__)" 2>nul
if errorlevel 1 echo   [WARN]  openvino not importable yet ^(expected before Phase 0 finishes^)

rem ---------------------------------------------------------------- 4. summary
:summary
echo.
echo === Drivers ^(install by hand if missing - the checks are done by tools\preflight.py^)
echo   * Intel NPU driver, 32.0.100.5540 or later
echo     https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html
echo   * Intel Arc / Iris Xe graphics driver
echo     https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html
echo.
echo === Booth hygiene ^(this machine only^)
echo   powercfg /change standby-timeout-ac 0
echo   powercfg /change monitor-timeout-ac 0
echo.
echo === Next steps
echo   .venv\Scripts\activate
echo   python tools\verify_sources.py      ^<- verify model/video sources by content
echo   python tools\download_models.py     ^<- models + labels + videos
echo   python tools\probe_telemetry.py     ^<- discover GPU/NPU counters
echo   python tools\preflight.py           ^<- PASS/FAIL table for the booth
echo   run_demo.bat                        ^<- the demo app
echo.
echo Note: only this bootstrap needed cmd. The app and every tool are Python,
echo so script-execution policy does not apply to them.
echo.
pause
endlocal

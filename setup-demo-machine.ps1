<#
  setup-demo-machine.ps1 — one-time bootstrap for the Engine Lab demo machine.

  What it does:
    1. Checks the Windows version (needs Windows 11 24H2 / build 26100 or later).
    2. Installs Python 3.12, Git and the VC++ redistributable via winget (unless -SkipToolInstall).
    3. Creates .venv and installs the pinned requirements.
    4. Checks for the Intel NPU and GPU devices and prints the driver versions it can see.
    5. Prints the manual driver links and the next commands to run.

  Usage (elevated PowerShell, from the repo folder):
    powershell -ExecutionPolicy Bypass -File .\setup-demo-machine.ps1
    powershell -ExecutionPolicy Bypass -File .\setup-demo-machine.ps1 -SkipToolInstall

  Safe to re-run: every step is idempotent.
#>

[CmdletBinding()]
param(
    [switch]$SkipToolInstall,
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"
$script:Failures = @()

function Write-Step  { param($m) Write-Host "`n=== $m" -ForegroundColor Cyan }
function Write-Ok    { param($m) Write-Host "  [ OK ]  $m" -ForegroundColor Green }
function Write-Warn2 { param($m) Write-Host "  [WARN]  $m" -ForegroundColor Yellow; $script:Failures += $m }
function Write-Bad   { param($m) Write-Host "  [FAIL]  $m" -ForegroundColor Red;    $script:Failures += $m }

Write-Host "Engine Lab - demo machine setup" -ForegroundColor White

# ---------------------------------------------------------------- 0. location
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
Write-Host "  repo: $RepoRoot"

# ---------------------------------------------------------------- 1. windows
Write-Step "Windows version"
$build = [int](Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuildNumber
$ubr   = (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion").UBR
Write-Host "  build $build.$ubr"
if ($build -ge 26100) {
    Write-Ok "Windows 11 24H2 or later"
} else {
    Write-Bad "Windows 11 24H2+ (build 26100+) is required. Current build: $build. Run Windows Update."
}

# ---------------------------------------------------------------- 2. winget tools
Write-Step "Developer tools"
function Test-Cmd { param($n) [bool](Get-Command $n -ErrorAction SilentlyContinue) }

if ($SkipToolInstall) {
    Write-Warn2 "Tool installation skipped by request (-SkipToolInstall)"
} elseif (-not (Test-Cmd winget)) {
    Write-Bad "winget not found. Install 'App Installer' from the Microsoft Store, or install Python 3.12 and Git by hand, then re-run with -SkipToolInstall."
} else {
    $pkgs = @(
        @{ Id = "Python.Python.$PythonVersion"; Name = "Python $PythonVersion" },
        @{ Id = "Git.Git";                     Name = "Git" },
        @{ Id = "Microsoft.VCRedist.2015+.x64"; Name = "VC++ 2015-2022 Redistributable (x64)" }
    )
    foreach ($p in $pkgs) {
        Write-Host "  installing $($p.Name) ..."
        try {
            winget install --id $p.Id -e --silent `
                --accept-package-agreements --accept-source-agreements | Out-Null
            Write-Ok "$($p.Name) installed (or already present)"
        } catch {
            Write-Warn2 "$($p.Name) install failed: $($_.Exception.Message)"
        }
    }
    # make newly installed tools visible in this session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}

foreach ($c in @("python","git")) {
    if (Test-Cmd $c) { Write-Ok "$c -> $((Get-Command $c).Source)" }
    else             { Write-Bad "$c not on PATH (open a NEW PowerShell window after installing)" }
}

# ---------------------------------------------------------------- 3. venv + deps
Write-Step "Python virtual environment"
if (Test-Cmd python) {
    $ver = (& python -c "import sys; print('%d.%d.%d' % sys.version_info[:3])").Trim()
    Write-Host "  python $ver"
    if ($ver -notlike "$PythonVersion.*") {
        Write-Warn2 "Python $ver found, expected $PythonVersion. OpenVINO 2026.4.0 ships wheels for 3.10-3.13; continue if the version is in that range."
    }
    if (-not (Test-Path ".venv")) {
        & python -m venv .venv
        Write-Ok "created .venv"
    } else {
        Write-Ok ".venv already exists"
    }

    $py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    & $py -m pip install --upgrade pip | Out-Null
    if (Test-Path "requirements.txt") {
        & $py -m pip install -r requirements.txt
        if ($LASTEXITCODE -eq 0) { Write-Ok "requirements installed" }
        else { Write-Bad "pip install -r requirements.txt failed (see output above)" }
    } else {
        Write-Warn2 "requirements.txt not found - the coding agent creates it in Phase 0 (see SPEC.md 13.3)"
    }

    # what the runtime actually reports
    try {
        $ov = (& $py -c "import openvino as ov; print(ov.__version__)").Trim()
        Write-Ok "openvino importable, version $ov"
    } catch {
        Write-Warn2 "openvino not importable yet (expected before Phase 0 finishes)"
    }
} else {
    Write-Bad "python not available - cannot create the venv"
}

# ---------------------------------------------------------------- 4. devices
Write-Step "Intel devices"
try {
    $npu = Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
           Where-Object { $_.Class -eq "ComputeAccelerator" -or $_.FriendlyName -match "NPU|AI Boost|Neural" }
    if ($npu) {
        foreach ($d in $npu) {
            $drv = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName "DEVPKEY_Device_DriverVersion" -ErrorAction SilentlyContinue).Data
            Write-Ok "$($d.FriendlyName)  [$($d.Status)]  driver $drv"
        }
    } else {
        Write-Bad "No NPU / compute-accelerator device found. Install the Intel NPU driver (see links below) and reboot."
    }
} catch { Write-Warn2 "NPU device query failed: $($_.Exception.Message)" }

try {
    $gpu = Get-PnpDevice -PresentOnly -Class Display -ErrorAction SilentlyContinue
    if ($gpu) {
        foreach ($d in $gpu) {
            $drv = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName "DEVPKEY_Device_DriverVersion" -ErrorAction SilentlyContinue).Data
            Write-Ok "$($d.FriendlyName)  [$($d.Status)]  driver $drv"
        }
    } else {
        Write-Warn2 "No display device found"
    }
} catch { Write-Warn2 "GPU device query failed: $($_.Exception.Message)" }

Write-Host "`n  Required drivers (install by hand if the checks above failed):" -ForegroundColor White
Write-Host "   * Intel NPU driver, 32.0.100.5540 or later"
Write-Host "     https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html"
Write-Host "   * Intel Arc / Iris Xe graphics driver"
Write-Host "     https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html"

# ---------------------------------------------------------------- 5. booth hygiene
Write-Step "Booth hygiene (optional, applies to THIS machine only)"
Write-Host "  To disable sleep/display timeout for a show:"
Write-Host "    powercfg /change standby-timeout-ac 0"
Write-Host "    powercfg /change monitor-timeout-ac 0"
Write-Host "    powercfg /change hibernate-timeout-ac 0"
Write-Host "  (Restore afterwards with e.g. 'powercfg /change monitor-timeout-ac 10'.)"

# ---------------------------------------------------------------- 6. next steps
Write-Step "Next steps"
@"
  cd $RepoRoot
  .\.venv\Scripts\Activate.ps1
  python tools\verify_sources.py      # verify model/video sources by content
  python tools\download_models.py     # models + labels + videos
  python tools\probe_telemetry.py     # discover GPU/NPU counters -> config\telemetry_map.json
  python tools\preflight.py           # PASS/FAIL table for the booth
  run_demo.bat                        # the demo app
"@ | Write-Host

Write-Step "Summary"
if ($script:Failures.Count -eq 0) {
    Write-Host "  All checks passed." -ForegroundColor Green
} else {
    Write-Host "  $($script:Failures.Count) item(s) need attention:" -ForegroundColor Yellow
    $script:Failures | ForEach-Object { Write-Host "    - $_" }
    Write-Host "  (WARN rows are not necessarily fatal - re-run preflight after fixing.)"
}

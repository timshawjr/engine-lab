$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CacheDir = Join-Path (Split-Path -Parent $RepoRoot) "cache"
if (Test-Path $CacheDir) {
    Write-Host "Removing OpenVINO cache: $CacheDir"
    Remove-Item -LiteralPath $CacheDir -Recurse -Force
} else {
    Write-Host "Cache directory is already absent: $CacheDir"
}
Write-Host "Cache reset complete. Run tools\preflight.py before the booth."

# Building the self-contained booth bundle

The bundle that teammates run is a **build artifact**, not source. It goes stale
whenever the code changes, so rebuild it after any change you want them to have.

Output: `engine-lab-portable.zip`, about **2.6 GB** (3.7 GB unpacked).

## Why this approach

It bundles a **portable copy of Python 3.12** plus the pinned packages, rather
than using PyInstaller. Freezing OpenVINO's native plugins with PyInstaller is
the most likely thing to break; copying the interpreter and the exact pinned
environment avoids that risk entirely and keeps `openvino==2026.4.0` honest.

The venv's `site-packages` are merged **into** the bundled interpreter's own
`site-packages`, so there is no `pyvenv.cfg` pointing at an absolute path. That
is what makes the folder work from any location — a venv alone would not, because
`pyvenv.cfg` records the machine that built it.

## Prerequisites on the BUILD machine

- The repo checked out and set up (`python -m venv .venv`, dependencies installed)
- Models present under `models/` (2.3 GB) and video under `media/` (195 MB)
- Python 3.12 installed normally, for its interpreter directory

## Build steps

Run from the repo root. These are the exact commands the current bundle was built
with; each was run and verified.

```powershell
$root = "$env:TEMP\engine-lab-portable"
Remove-Item -Recurse -Force $root -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $root -Force | Out-Null

# 1. Bundled interpreter. A copied Python install runs standalone: no registry,
#    no PATH, and it reports its own prefix.
Copy-Item "$env:LOCALAPPDATA\Programs\Python\Python312" "$root\python" -Recurse -Force

# 2. Merge the pinned packages into it, so no pyvenv.cfg path is involved.
Copy-Item ".venv\Lib\site-packages\*" "$root\python\Lib\site-packages" -Recurse -Force

# 3. Application source.
foreach ($d in @("app","config","docs","tests","tools","video-assets")) {
  Copy-Item $d "$root\$d" -Recurse -Force
}
foreach ($f in @("requirements.txt","README.md","AGENTS.md","bench_report.md","SALES-GUIDE.md","START-HERE.md")) {
  Copy-Item $f $root -Force
}

# 4. Models and video, so the demo runs offline.
Copy-Item "models" "$root\models" -Recurse -Force
Copy-Item "media"  "$root\media"  -Recurse -Force

# 5. The two launchers and README-PORTABLE.md are NOT generated here.
#    Keep them in the repo under packaging/ and copy them in, so they are
#    versioned rather than retyped.
Copy-Item "packaging\*" $root -Force

# 6. Ship clean: the availability cache is keyed to the BUILD machine's driver
#    versions, so it must not be shipped. Each machine probes its own on first
#    launch.
foreach ($d in @("cache","logs","__pycache__")) {
  Get-ChildItem $root -Recurse -Directory -Filter $d -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}
Remove-Item -Recurse -Force "$root\cache","$root\logs" -ErrorAction SilentlyContinue

# 7. Zip. Compresses to roughly 2.6 GB; takes about four minutes.
$zip = "$env:TEMP\engine-lab-portable.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
  $root, $zip, [System.IO.Compression.CompressionLevel]::Fastest, $true)
```

## Verify before shipping

**Always do this on the built folder**, with Python removed from `PATH`, to prove
the bundle is not silently using the build machine's interpreter:

```powershell
$env:PATH = (($env:PATH -split ';' | Where-Object { $_ -notmatch 'Python' }) -join ';')
cd "$env:TEMP\engine-lab-portable"
.\python\python.exe -m app.main --selftest     # expect 12 PASS, 0 FAIL
```

On a genuinely fresh copy the self-test needs the availability cache, which does
not exist yet, so it reports one FAIL. Warm it first — `Verify Booth.bat` does
this automatically:

```powershell
.\python\python.exe -m app.main --scenario retail --density 1 --exit-after 20
.\python\python.exe -m app.main --selftest     # now 12 PASS, 0 FAIL
```

Then confirm the demo actually renders, not just that tests pass:

```powershell
.\python\python.exe -m app.main --scenario manufacturing --density 1 `
  --rag-page --rag-ask "What is a safety instrumented system?" `
  --exit-after 24 --screenshot logs\bundle-check.png
```

**Open the screenshot and look at it.** The last run should show video detection,
live engine gauges and a document answer together.

## Gotchas

- **GitHub cannot host this.** Its file limit is 100 MB; the zip is 2.6 GB. Use
  corporate OneDrive/SharePoint or a USB stick — not a public drive.
- The zip is written with **backslash** separators by .NET on Windows. Windows
  Explorer extracts it correctly, which is the audience, but some cross-platform
  unzip tools mis-handle it.
- `cache/` is deliberately excluded and `README-PORTABLE.md` explains why.
- Rebuild after any code change, or teammates keep running the old snapshot.

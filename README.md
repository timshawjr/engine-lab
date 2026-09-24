# Engine Lab

Engine Lab is a single-process Windows 11 Python demo for an Intel Core Ultra NPU, integrated GPU,
and CPU. The project follows an honesty-first rule: every displayed utilization or performance
value must be measured or come from a cited configuration source. Missing counters are labelled
as missing; no synthetic load or fallback zeros are used.

## Phase status

Phases 0 and 1 are complete. Phase 2 now provides the DeviceAvailability startup gate, a live
device-policy object, NPU/GPU operator toggles, explicit placement badges, density 1/2/4/6/8, and
one independent `AsyncInferQueue` per displayed stream. The visible video clock is independent of
inference, so policy swaps do not freeze the source image or 5 Hz telemetry.

The Phase 2 implementation is measured on this machine, but its report does **not** claim the
literal §11.8 strict-stream assertion as passed: at density 8, no individual tile held 90% of the
59.94 FPS source when accelerators were enabled. The measured aggregate real-time capacity was
3.391 source-rate stream equivalents with accelerators on versus 0.920 with both off. Both runs are
recorded honestly in `bench_report.md`; the strict count was 0 versus 0. This is an observed
hardware/workload limitation, not a hidden or simulated pass.

Phases 3 features—the other three scenarios, business-event overlays, attract mode, and the full
booth operator overlay—are not implemented yet. Raw command logs, screenshots, availability
results, and diagnostic samples are under `logs/` on the demo machine.

## What was installed on this machine

The successful route was direct `winget` commands, not a PowerShell build script:

```cmd
winget install --id Python.Python.3.12 -e --scope user --silent --accept-package-agreements --accept-source-agreements
winget install --id Git.Git -e --scope user --silent --accept-package-agreements --accept-source-agreements
```

The x64 Visual C++ runtime was already installed as `14.50.35710.00`. Python 3.12.10 was installed
under the user profile. The repository-local environment was then created directly:

```cmd
cd C:\dev\engine-lab
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No application command requires elevation. Driver installation is the only elevated operation and
must be performed by the operator.

## One-time Phase 0 verification

Run these commands directly; do not depend on script execution policy:

```cmd
cd C:\dev\engine-lab
.venv\Scripts\python.exe tools\verify_sources.py
.venv\Scripts\python.exe tools\download_models.py
.venv\Scripts\python.exe tools\probe_telemetry.py
.venv\Scripts\python.exe tools\preflight.py
.venv\Scripts\python.exe -m app.main --selftest
```

`verify_sources.py` validates response bodies: OpenVINO IR must start with an XML declaration,
Hugging Face label config must be JSON with labels, binary payloads must not be HTML or Git LFS
pointers, model cards must identify the configured repository/model, and videos must have an MP4
`ftyp` box. HTTP status alone is not accepted.

`download_models.py` is idempotent. It writes models to `models/`, videos to `media/`, and a
SHA-256 inventory to `models/download_manifest.json`. Hugging Face label metadata is retained as
`source_config.json`; OMZ label metadata is written with its official model-document URL.

`probe_telemetry.py` prints every observed GPU Engine instance, writes
`config/telemetry_map.json`, and exits non-zero if no NPU instance is found unless
`--allow-fallback` is explicitly supplied. The app-measured NPU fallback is always declared in the
map and must be labelled in the future HUD.

`preflight.py` is the phase exit gate. Do not use `--skip-compile` for phase verification; that
switch is only for local development while iterating on the checker. Phase 2 executes 20 real
`AsyncInferQueue` requests for every one of the 10 models on NPU, GPU, and CPU, verifies each
reported `EXECUTION_DEVICES` root, validates the fingerprinted availability cache, and retains the
Phase 1 real-retail-frame and physical-core CPU-stream assertions.

## Run the Phase 2 retail demo

The one-click wrapper starts the default `spread` policy fullscreen with no console window:

```cmd
cd C:\dev\engine-lab
run_demo.bat
```

Equivalent explicit startup commands are:

```cmd
.venv\Scripts\python.exe -m app.main --source loop --scenario retail --mode spread --density 1 --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario retail --mode npu_only --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario retail --mode gpu_only --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario retail --mode cpu_only --fullscreen
```

Operator keys:

- `N` / `G`: disable or re-enable the NPU/GPU; the header, gauges, tile badges, and compiled
  placement update from the same policy snapshot.
- `C`: cycle `auto → spread → npu_only → gpu_only → cpu_only → split`, skipping unavailable devices.
- `+` / `-`: change density through 1, 2, 4, 6, 8.
- `F1`: open the 10-model × 3-device availability matrix.
- `F11`: toggle fullscreen. `Q` asks for confirmation before quitting; `Esc` exits immediately.

The camera variant remains opt-in and uses the same policy and density pipeline:

```cmd
.venv\Scripts\python.exe -m app.main --source camera:0 --mode spread --density 1 --fullscreen
```

For auditable runs, `--exit-after`, `--screenshot`, and `--diagnostic-output` write raw frame and
telemetry samples. The two scheduled-toggle switches reproduce the §11.5–7 checks without synthetic
input:

```cmd
.venv\Scripts\python.exe -m app.main --mode spread --density 2 --npu-toggle-after 3 --exit-after 10 ^
  --screenshot logs\phase2-toggle-npu.png --diagnostic-output logs\phase2-toggle-npu.json
.venv\Scripts\python.exe -m app.main --mode spread --density 2 --gpu-toggle-after 3 --exit-after 10 ^
  --screenshot logs\phase2-toggle-gpu.png --diagnostic-output logs\phase2-toggle-gpu.json
.venv\Scripts\python.exe -m app.main --mode spread --density 8 --npu-off --gpu-off --exit-after 10 ^
  --screenshot logs\phase2-d8-off.png --diagnostic-output logs\phase2-d8-off.json
```

At startup, the gate compiles or loads the fingerprinted cache and shows per-model progress. Full
results are written to `cache/availability.json`; every probe and cache event is appended to
`logs/availability.log`. The cache key includes the OpenVINO version, Windows build, Intel NPU/GPU
driver versions, available device list, and SHA-256 of every model file.

The app reads no network resource at runtime. `python -m app.main --selftest` explicitly blocks
outbound Python sockets while checking local models, labels, telemetry, policy, availability, and
UI invariants. Pure policy/cache unit tests run with:

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Verified model-shape correction

The downloaded `vehicle-license-plate-detection-barrier-0106` IR and its official Open Model Zoo
`model.yml` both report the static input as `[1, 300, 300, 3]` (NHWC). The supplied configuration
had recorded `[1, 3, 256, 384]`, which did not match the pinned IR. The configuration value was
corrected to the measured IR shape before the Phase 0 preflight. No model, media, or URL source was
substituted, and `tools/verify_sources.py` was run again after the correction.

## Telemetry found on this machine

The actual provider is the Windows PDH **GPU Engine** counter set, reached through `ctypes` and
language-neutral `PdhAddEnglishCounterW`. SetupAPI reported:

```text
Compute accelerator: Intel(R) AI Boost
NPU driver: 32.0.100.5540, phys_id=0, LUID node=0x000128A6
GPU: Intel(R) Arc(TM) 140V GPU (16GB)
GPU driver: 32.0.101.6737, phys_id=0, LUID node=0x000124A5
```

Both devices expose `phys_0`; physical id alone is therefore ambiguous on this Lunar Lake machine.
The probe uses the exact SetupAPI LUID as the tie-breaker before considering physical id. An
observed NPU counter line was:

```text
pid_12332_luid_0x00000000_0x000128A6_phys_0_eng_0_engtype_Neural = 0.00% [NPU/Neural]
```

The `0.00%` above is the counter value observed while that process was idle; it is not a benchmark
result. With driver 32.0.100.5540, the NPU engine is reported as `Neural`. The probe also logged
actual GPU engine strings (`3D`, `Compute`, `Copy`, `VideoDecode`, `VideoProcessing`, and `GSC`) and
classified an unrelated LUID as `UNKNOWN` rather than guessing.
The complete raw output is `logs/probe_telemetry.log` on the demo machine.

## Driver gate and recovery

The Phase 0 specification requires Intel NPU driver `32.0.100.5540` or later. The official
Intel-signed `npu_win_32.0.100.5540.exe` package was installed on this machine. Its SHA-256 was
`FBA32699B699918A793FCA05CFC6C607C8745772057347476ECD8C4CF7746048`, and Authenticode verification
reported `Valid`, signer `Intel Corporation`. The installer returned `1014`, but SetupAPI after the
required reboot independently reported driver `32.0.100.5540`; preflight is the authority.

- Intel NPU driver: <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>
- Intel Arc/Iris Xe graphics driver: <https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html>

After any Intel NPU or graphics driver update, reset the OpenVINO blob cache before preflight. The
operator can run the convenience script or issue the equivalent command directly:

```powershell
Remove-Item -LiteralPath C:\dev\engine-lab\cache -Recurse -Force -ErrorAction SilentlyContinue
```

If a model compiled yesterday but not today:

1. Confirm the NPU and GPU driver versions in `tools/preflight.py` output.
2. Reset `cache/` after any driver change.
3. Re-run `tools/probe_telemetry.py` and confirm SetupAPI LUIDs and PDH instances still match.
4. Re-run `tools/preflight.py` without `--skip-compile`.
5. Inspect `logs/availability.log`; do not hide a model failure by silently falling back to CPU.

## Task Manager cross-check

At the booth, park Task Manager on the second monitor and enable the NPU, NPU engine, GPU, and GPU
engine columns. Run one stream on a single explicit device, then compare the app's measured gauge
with the matching process/engine row. Repeat after changing devices. The human Task Manager view is
the cross-check only; the application never screenshots, OCRs, or shells out to it.

## Pre-show machine hygiene

Before visitors arrive:

- Connect AC power and set the active power plan to Best performance.
- Disable sleep, monitor, and hibernate timeouts on AC.
- Keep the lid open or support the laptop so it cannot throttle or sleep.
- Connect the external display at 1920x1080 or higher.
- Park Task Manager on the second display for the live NPU/GPU placement check.
- Run the full preflight after the final reboot.
- Keep Wi-Fi disabled during the demo to prove runtime operation is local.

The current checkpoint is the measured Phase 2 retail policy/density build. The strict §11.8
per-stream real-time relation remains an explicit, documented miss on this machine; the raw
accelerator-on and both-off density-8 runs are preserved. Attract mode and the other verticals
remain intentionally absent until Phase 3.

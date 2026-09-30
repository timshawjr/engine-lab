# Engine Lab

Engine Lab is a single-process Windows 11 Python booth demo for an Intel Core Ultra NPU,
integrated GPU, and CPU. It plays deterministic, prerecorded video through real OpenVINO
inference and shows measured throughput, latency, placement, and business events at a glance.
Every displayed utilization or performance value is measured or comes from a cited configuration
source. Missing counters are labelled as missing; the app does not generate synthetic load or
smooth a value and present it as measured.

## Phase status

Phases 0-3 are implemented. The full acceptance matrix has been run on the demo machine
(300s per scenario at density 1, 60s per scenario at density 4, then a 600s density-4 attract
cycle): **8 PASS, 1 FAIL**. The FAIL is `gov_defense-d4`, which hung and was killed by Windows as
an Application Hang — see `docs/STATE.md` section 5A, which is the sign-off blocker.

Memory is not a leak: RSS climbs from 861 MiB to about 2.6 GB during the first ~70 seconds, then
stays flat. From the 2-minute checkpoint to the end of the attract run (8.5 minutes, 50 scenario
switches) growth is **+17.9 MiB (0.68%)**. The app settles around **2.6 GB**.

**Read `docs/STATE.md` before changing anything.** It records what is verified, what is still
broken, and the amendments to `docs/SPEC.md`.

## Repository map

- `app/main.py` — CLI, startup gate, scenario compiler, diagnostics, scheduled test actions.
- `app/hud.py` — Qt booth UI, overlays, gauges, stream tiles, ticker, attract mode, F1 operator view.
- `app/engine/device_policy.py` — modes, NPU/GPU toggles, density, and explicit round-robin placement.
- `app/engine/availability.py` — model × device startup gate, `EXECUTION_DEVICES` validation, fingerprint cache.
- `app/engine/pipelines.py` — four scenario graphs, model registry, CPU preprocessing, and persistent stream workers.
- `app/engine/stages.py` — static-shape preprocessing, YOLO/SSD/classification/pose postprocessing.
- `app/engine/events.py` — business-event dictionaries and track/zone state.
- `app/telemetry/` — PDH GPU/NPU counters, SetupAPI device identity, psutil CPU, and 5 Hz sampling.
- `config/scenarios.json` — the four normalized-zone scenario definitions and event thresholds.
- `tools/preflight.py` — the local PASS/FAIL gate.
- `tools/benchmark_matrix.py` — the long-run scenario × density acceptance matrix.
- `bench_report.md` — measured evidence, including the explicit Phase 2 miss.

`models/`, `media/`, `cache/`, and `logs/` are local/git-ignored runtime data. The repository root
on this machine is `C:\dev\engine-lab`; the only intentional machine-specific paths are in this
runbook and `run_demo.bat`.

## Machine setup (a machine with no dev tools)

Script execution may be blocked by policy on the demo machine, so nothing here depends on running
a `.ps1` or `.bat`:

```cmd
winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
winget install --id Git.Git -e --accept-package-agreements --accept-source-agreements
cd C:\dev\engine-lab
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open a **new** cmd window after installing Python so `py` and `python` are on PATH. No-admin
alternative: the standalone `uv` binary zip from
`https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip`, then
`uv.exe python install 3.12`, `uv.exe venv --python 3.12 .venv`,
`uv.exe pip install --python .venv\Scripts\python.exe -r requirements.txt`.

Only the bootstrap is a script. The app and every tool runs through `python.exe`, which script
policy does not restrict. If `.bat` files are blocked too, launch the app with
`.venv\Scripts\python.exe -m app.main` instead of `run_demo.bat`.

## One-time machine verification

Run from an ordinary standard-user PowerShell or command prompt. Driver installation is the only
elevated operation; the demo itself must not be elevated.

```cmd
cd C:\dev\engine-lab
.venv\Scripts\python.exe tools\verify_sources.py
.venv\Scripts\python.exe tools\download_models.py
.venv\Scripts\python.exe tools\probe_telemetry.py
.venv\Scripts\python.exe tools\preflight.py
.venv\Scripts\python.exe -m app.main --selftest
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`verify_sources.py` checks response bodies, not only HTTP status: IR must begin with an XML
 declaration, model labels must be valid metadata, binary payloads must not be HTML/LFS pointers,
and videos must contain an MP4 `ftyp` box. `download_models.py` is idempotent and records a
SHA-256 inventory. `probe_telemetry.py` writes the raw PDH instance map and fails closed when no
NPU counter is found unless its explicit development fallback is requested.

The current local checks on the demo machine are:

- source verification: **43 PASS, 0 FAIL**;
- preflight: **95 PASS, 0 WARN, 0 FAIL**, including 20 asynchronous inferences for every model x
  NPU/GPU/CPU combination and the four scenario graphs;
- network-blocked self-test: **11 PASS, 0 FAIL**;
- unit tests: **48 PASS, 0 FAIL** on `fix/demo-killers` (+5 hangwatch, +5 benchmark-reporting on
  the later branches).

Re-run them before a show; the commands are in `docs/STATE.md` section 7.

## Frame-level review harness

The development-only review runner uses the real scenario worker and retains frame-level evidence
without changing the production dependency set:

```powershell
$env:PYTHONPATH = "C:\Users\Intel Demo\AppData\Local\Temp\opencode\engine-lab-review-deps"
$env:QT_QPA_PLATFORM = "offscreen"
.venv\Scripts\python.exe tools\review_sessions.py --scenario retail --seconds 60 --output logs\review-retail
.venv\Scripts\python.exe tools\review_sessions.py --scenario medical --seconds 60 --output logs\review-medical
Remove-Item Env:PYTHONPATH
Remove-Item Env:QT_QPA_PLATFORM
```

Each run writes `annotated.mp4`, `contact-sheet.png`, `frames.jsonl`, and `summary.json` under the
selected output directory. The JSONL records raw detections, tracked IDs, labels, confidences,
classifications, keypoints, zones, events, stage metrics, and explicit execution placements.

### How retail finds and names a product

Retail detects on **YOLO11n** and names with **CLIP ViT-B/32** in zero-shot mode. The store's
inventory is declared in `tools/clip_vocabulary.py` as plain product names with prompt templates,
and CLIP may only ever return one of those words, so the overlay cannot invent a class.

**The classifier is more accurate than the detector's label, and that is the point.** On the
store-aisle clip YOLO11n reports `bowl` on every pass. It is a stainless steel saucepan with a rim
and handle, and CLIP independently names it `pot`: 75 of 75 confident calls agreed, verified by
inspecting the crops. The old failure was never only a bad classifier - we were grading CLIP
against a COCO label that was itself wrong, and an allowlist built on that wrong label.

Measured over 60 s (3,138 frames): `bowl -> pot` on 1,368 frames, plus `mixing bowl`, `soup bowl`
and `storage container`. **The person is never classified.** The FPS shown is measured.

| Stage | Device | Measured | Job |
|---|---|---|---|
| YOLO11n (`detector`) | **NPU** | 12.3 ms/frame (81 fps) | find the item, every frame, full rate |
| CLIP (`product_classifier`) | **GPU** | ~3 ms/crop | name it from the declared vocabulary |
| event logic | **CPU** | sub-ms | zones, tracks, business events |

All three engines report measured activity, and 4 concurrent streams hold **25.7 FPS** with NPU at
34%, GPU at 94% and CPU at 73%.

**No tokenizer on the booth machine.** The vocabulary is fixed configuration, so
`tools/build_clip_zero_shot.py` bakes the text embeddings for it once. The app ships only the
vision tower and runs no tokenizer and no text encoder. Both build tools are development-only and
need `torch`/`transformers`, which are deliberately absent from the production
`requirements.txt`.

Rebuild with:

```cmd
<dev-venv>\Scripts\python tools\build_clip_zero_shot.py
.venv\Scripts\python tools\convert_clip_onnx.py --fp16
```

**Three gates keep the output defensible, all measured rather than assumed:**

1. **Detector allowlist** - only the classes the detector actually gets right on this footage
   (`bowl`, `cup`, `bottle`, `wine glass`, `vase`) are submitted to CLIP. `person` is excluded
   explicitly, and so is every unlisted class, so the gate fails closed.
2. **Per-label business-event gate** - the declared inventory also gates `object_classified` and
   `object_picked_up`. Without it the tracker logged `object_picked_up person`, which is not a
   statement worth making, and on other footage it logged `object_picked_up microwave`.
3. **Temporal stability** - a label must repeat for `classify_min_frames` consecutive frames on one
   track before it is displayed or emitted, and it emits once per track per label rather than once
   per frame.

**Known limit, stated plainly:** the vocabulary is a *category* list, not a SKU database. `pot` is a
true statement about the crop; it is not a product code and the app does not claim it is one. The
second-shelf items sit at roughly 50x50 px, so the weaker vocabulary entries land near the 0.22
confidence floor and appear on only a few frames each.

### One-click default

```cmd
cd C:\dev\engine-lab
run_demo.bat
```

The launcher uses `pythonw.exe`, starts `app.main` fullscreen with the retail scenario, `spread`
policy, and density 1, and does not leave a console window visible. It is deterministic and does
not use a camera or network. On this machine the measured cache-cold startup was 26.09 s
(32.02 s including the five-second diagnostic exit), and the warm startup was 1.00 s
(10.57 s including the same exit); both were error-free.

### Exact scenario commands

Run any vertical explicitly with:

```cmd
.venv\Scripts\python.exe -m app.main --source loop --scenario retail      --mode spread --density 1 --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario smart_city  --mode spread --density 1 --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario medical      --mode spread --density 1 --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario gov_defense --mode spread --density 1 --fullscreen
```

The four graphs are:

| Key | Scenario | Video | Measured stage story |
|---:|---|---|---|
| `1` | Retail / Shelf Monitoring | `store-aisle-detection.mp4` | YOLO11n detection on the NPU → CLIP names the item from a declared store vocabulary → shelf/zone events |
| `2` | Smart city / traffic | `smart-city-traffic-montage.mp4` | busy crosswalk/traffic-light footage → lane/zone counts (detector class is the business answer; no weak classifier) |
| `3` | Medical / eldercare | `one-by-one-person-detection.mp4` | person detection → pose heatmap/PAF decoding → posture/zone events |
| `4` | Government / defense | `government-perimeter-montage.mp4` | worker-perimeter action → vehicle/plate footage → independent perimeter copies → person/plate detection → IoU tracks/plate events |

The initial `spread` assignment is explicit and visible in each tile. Retail, smart-city, and pose
work use the NPU where the measured graph calls for it; the medical person detector is explicitly
GPU-preferred because the local NPU output for that OMZ detector was not useful for the event
overlay, while its pose stage remains on the NPU. The operator overlay always reports the actual
`EXECUTION_DEVICES`; the requested preference is never presented as proof.

### Live failover controls

Press `G` during `SPREAD` or `SPLIT` to disable GPU assignment. GPU-preferred auxiliary stages then
fail over to NPU when NPU is available. Press `N` to disable NPU assignment; NPU-preferred auxiliary
stages then fail over to GPU. If both accelerators are disabled, they fall back to CPU. The status
panel shows `GPU OFF · NPU FAILOVER`, `NPU OFF · GPU FAILOVER`, or `NPU/GPU OFF · CPU FALLBACK`, and
the stage tiles continue to show the measured execution devices.

### Attract and unattended run

Attract mode hides the gauges, shows a large vertical headline, cycles deterministically through
all four scenarios, and continues processing/telemetry in the background. Any key returns to the
measured demo page.

```cmd
.venv\Scripts\python.exe -m app.main --source loop --scenario retail --mode spread --density 4 --attract
```

The full acceptance matrix is:

```cmd
.venv\Scripts\python.exe tools\benchmark_matrix.py
```

It runs each scenario for five minutes at density 1, each scenario for one minute at density 4,
and a ten-minute density-4 attract cycle. It fails a run on an application error, incomplete
stream count, missing scenario visit, or an RSS growth of 5% or more from the first sample at or
after two minutes to the final sample. Use `--quick` only for development smoke tests; quick
RSS numbers are not acceptance evidence.

### Camera variant (opt-in only)

The deterministic loop is the default. A camera is never used unless explicitly requested:

```cmd
.venv\Scripts\python.exe -m app.main --source camera:0 --scenario retail --mode spread --density 1 --fullscreen
```

## Keyboard map

| Key | Action |
|---|---|
| `1`–`4` | Select retail, smart city, medical, or government/defense |
| `N` | Toggle the NPU; the next inference request uses the reduced device set |
| `G` | Toggle the GPU |
| `C` | Cycle `auto → spread → npu_only → gpu_only → cpu_only → split` |
| `+` / `-` | Change density through 1, 2, 4, 6, 8 |
| `A` | Enter/leave attract mode; any key leaves attract mode |
| `F1` | Open the operator overlay |
| `F11` | Toggle fullscreen |
| `Q` | Quit after confirmation; `Esc` exits immediately |

Scenario switches are in-process. Workers retain compiled model stores and per-scenario
`AsyncInferQueue` caches, while the video clock, overlay, zones, ticker, and event rules switch
without a process reload. A placement transition is acknowledged only after every active worker
reports its new `EXECUTION_DEVICES` payload.

## What the HUD measures

- **Engine gauges:** NPU, GPU, and CPU utilization from the declared telemetry source. PDH
  `GPU Engine` is used when available; the app-measured NPU duty-cycle fallback is explicitly
  labelled `APP`. CPU is psutil. A disabled engine is grey, reads `OFF BY OPERATOR`, and retains its
  last measured value rather than displaying a fabricated zero.
- **Placement:** stream/stage badges and F1 tables use `compiled_model.get_property(
  "EXECUTION_DEVICES")`. A requested string alone is never used as proof.
- **Pipeline metrics:** end-to-end p50/p95, inference count/rate, detection rate, stage breakdown,
  rolling event count, strict real-time stream count, and source-rate equivalents. Percentiles,
  windows, and display thresholds live in `app/theme.py` or scenario config, not as hidden metric
  constants in widgets.
- **Business events:** every event is a dictionary with `ts`, `type`, `label`, `confidence`, and
  `zone`; it appears in the on-video ticker and rolling count. The event tracker implements
  `object_picked_up`, `object_classified`, `person_counted`, `vehicle_counted`, `posture_alert`, and
  `plate_detected`. Dwell, confidence, IoU, fall-angle, and cooldown thresholds are in
  `config/scenarios.json`.
- **Peak TOPS:** the platform card labels these values `peak`; the F1 System tab exposes the
  profile source. Runtime/measured values are not replaced with peak specifications.

## Operator overlay and diagnostics

Press `F1` at the booth. The overlay contains:

1. all ten models × NPU/GPU/CPU availability, compile status, exact execution devices, and source URLs;
2. the live stream/stage placement table;
3. the raw telemetry map, including PDH instance names, LUID/physical IDs, provider, and fallback;
4. CPU topology, driver versions, cache path, scenario IDs, and peak-profile source;
5. the business-event tail and the last 20 session-log lines.

For an auditable run, add `--diagnostic-output logs\<name>.json` and optionally
`--screenshot logs\<name>.png`. The JSON includes startup time, policy sequence and transitions,
scenario history, per-stage timings, all sampled telemetry, event dictionaries, availability
results, and the RSS timeline. The benchmark tool consumes these files rather than scraping the UI.

## Pre-show checklist

1. Connect AC power and use the Best performance power plan; disable sleep/hibernate timeouts on AC.
2. Keep the lid open or support the laptop so the platform cannot throttle or sleep.
3. Connect the external display at 1920×1080 or higher and set the display scale so the 18 px minimum UI text is legible from two metres.
4. Run the full preflight after the final reboot and confirm 0 FAIL.
5. Run the network-blocked self-test and the unit tests.
6. Run one short command for each scenario and one F1 operator-overlay check.
7. Run the quick matrix during development; run the full matrix before signing off Phase 3.
8. Disable Wi-Fi for the booth to demonstrate that the runtime is local.
9. Park Task Manager on the second monitor for the human cross-check.
10. Keep `run_demo.bat`, the current session log, and the latest diagnostic JSON with the show kit.

## Task Manager cross-check

Task Manager is a human cross-check, not an application data source. Enable the NPU/NPU engine,
GPU, and GPU engine columns. Run one explicit device, then compare the matching process/engine row
with the app gauge and F1 placement table. Repeat after `N`, `G`, and mode changes. The app never
screenshots, OCRs, or shells out to Task Manager.

## Recovery: “NPU compiled yesterday but not today”

1. Compare the current NPU and GPU driver versions in `tools/preflight.py` output with the
   configured minimums.
2. After any NPU/graphics driver update, remove the OpenVINO blob cache:

   ```powershell
   Remove-Item -LiteralPath C:\dev\engine-lab\cache -Recurse -Force -ErrorAction SilentlyContinue
   ```

3. Re-run `tools/probe_telemetry.py`; confirm the Intel LUIDs and PDH instance strings still match.
4. Re-run `tools/preflight.py` without `--skip-compile`.
5. Inspect `logs/availability.log` and the F1 model table. A model that is unavailable on its
   preferred accelerator may use the explicitly labelled CPU fallback; a hidden failure is not an
   acceptable recovery.
6. Re-run the affected scenario and a short toggle test. If the NPU remains unavailable, report the
   measured fallback rather than forcing a placement claim.

## Known measured limitation

The strict Phase 2 density-8 relation is still not claimed as a pass. The complete evidence and
numbers are in `bench_report.md`; Phase 3 does not hide or reinterpret that result. Phase 3 long-run
acceptance concerns the four scenario durations, density-4 operation, attract-mode stability, and
RSS criterion, while retaining the same honest metric definitions.

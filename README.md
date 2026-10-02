# Engine Lab

Engine Lab is a single-process Windows 11 Python booth demo for an Intel Core Ultra NPU,
integrated GPU, and CPU. It plays deterministic, prerecorded video through real OpenVINO
inference and shows measured throughput, latency, placement, and business events at a glance.
Every displayed utilization or performance value is measured or comes from a cited configuration
source. Missing counters are labelled as missing; the app does not generate synthetic load or
smooth a value and present it as measured.

## Phase status

Phases 0–2 are complete and committed locally. Phase 3 is implemented and its full acceptance
matrix has passed for the four scenarios that existed at the time (`retail`, `metro`, `health`,
`federal`): all four ran for five minutes at density 1 and one minute at density
4, followed by a ten-minute density-4 attract cycle. The attract RSS comparison was 3051.39 MiB
at the first sample after two minutes and 3099.17 MiB at the final sample, measured growth **1.57%**.
The generated evidence is in `logs/phase3-benchmark-summary.json`, the per-run JSON files, and the
marked section at the end of `bench_report.md`.

The Phase 2 literal §11.8 strict-stream result remains an explicit, documented miss on this machine:
at density 8, zero individual tiles held 90% of the 59.940 FPS retail source with accelerators on,
and zero did with both accelerators off. The separately labelled aggregate source-rate capacity was
3.391 stream equivalents with accelerators on versus 0.920 with both off. That aggregate number was
not substituted for the strict count, and no priority scheduling, source-FPS reduction, frame-drop
declaration, or metric relabeling was added.

## Repository map

- `app/main.py` — CLI, startup gate, scenario compiler, diagnostics, scheduled test actions.
- `app/hud.py` — Qt booth UI, overlays, gauges, stream tiles, ticker, attract mode, F1 operator view.
- `app/engine/device_policy.py` — modes, NPU/GPU toggles, density, and explicit round-robin placement.
- `app/engine/availability.py` — model × device startup gate, `EXECUTION_DEVICES` validation, fingerprint cache.
- `app/engine/pipelines.py` — seven scenario graphs, model registry, CPU preprocessing, and persistent stream workers.
- `app/engine/stages.py` — static-shape preprocessing, YOLO/SSD/classification/pose postprocessing.
- `app/engine/events.py` — business-event dictionaries and track/zone state.
- `app/telemetry/` — PDH GPU/NPU counters, SetupAPI device identity, psutil CPU, and 5 Hz sampling.
- `config/scenarios.json` — the seven normalized-zone scenario definitions and event thresholds.
- `tools/preflight.py` — the local PASS/FAIL gate.
- `tools/benchmark_matrix.py` — the long-run scenario × density acceptance matrix.
- `bench_report.md` — measured evidence, including the explicit Phase 2 miss.

`models/`, `media/`, `cache/`, and `logs/` are local/git-ignored runtime data. The repository root
on this machine is `C:\dev\engine-lab`; the only intentional machine-specific paths are in this
runbook and `run_demo.bat`.

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

The current local checks on this machine are:

- source verification: **43 PASS, 0 FAIL**;
- preflight: **89 PASS, 0 WARN, 0 FAIL**, including 20 asynchronous inferences for every model ×
  NPU/GPU/CPU combination and the seven Phase 3 scenario graphs;
- network-blocked self-test: **11 PASS, 0 FAIL**;
- unit tests: **25 PASS, 0 FAIL**.

## Frame-level review harness

The development-only review runner uses the real scenario worker and retains frame-level evidence
without changing the production dependency set:

```powershell
$env:PYTHONPATH = "C:\Users\Intel Demo\AppData\Local\Temp\opencode\engine-lab-review-deps"
$env:QT_QPA_PLATFORM = "offscreen"
.venv\Scripts\python.exe tools\review_sessions.py --scenario retail --seconds 60 --output logs\review-retail
.venv\Scripts\python.exe tools\review_sessions.py --scenario health --seconds 60 --output logs\review-health
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
<dev-venv>\Scripts\python tools\build_clip_zero_shot.py --scenario retail
<dev-venv>\Scripts\python tools\build_clip_zero_shot.py --scenario metro
.venv\Scripts\python tools\convert_clip_onnx.py --fp16
```

Each vertical that uses CLIP declares its vocabulary by name on the
`product_classifier` stage in `config/scenarios.json` (`"vocabulary": "retail"`,
`"vocabulary": "metro"`), and `--scenario <name>` bakes that vocabulary's
pair (`text_embeddings_<name>.npy` + `vocabulary_<name>.json`) into the CLIP
model directory. A plain run with no `--scenario` rewrites the default pair
(`text_embeddings.npy` + `vocabulary.json`), which is the fallback when a
stage's named pair is missing. **If the footage changes, the vocabulary must be
rebuilt per scenario** (`--scenario` once per vertical) so the declared
categories still match what the camera sees; the vision tower is shared and
does not change with the vocabulary.

**Three gates keep the output defensible, all measured rather than assumed:**

1. **Detector allowlist** - only the COCO classes whose crops are products on this footage
   (`bowl`, `cup`, `bottle`, `wine glass`, `vase`, `apple`, `banana`, `suitcase` - the chip bag -,
   `cake`, `sandwich`) are submitted to CLIP. `person` is excluded
   explicitly, and so is every unlisted class (on this footage YOLO also reports `oven`, `microwave`
   and `tv` - 89/21/18 raw detections in the 40 s review - and those crops stay out), so the gate
   fails closed.
2. **Per-label business-event gate** - the declared inventory gates `object_picked_up` only;
   `object_classified` is already gated by the CLIP vocabulary and `classify_min_frames`.
   Without it the tracker logged `object_picked_up person`, which is not a
   statement worth making, and on other footage it logged `object_picked_up microwave`.
3. **Temporal stability** - a label must repeat for `classify_min_frames` consecutive frames on one
   track before it is displayed or emitted, and it emits once per track per label rather than once
   per frame.

**Known limit, stated plainly:** the vocabulary is a *category* list, not a SKU database. `bananas`
is a true statement about the crop; it is not a product code and the app does not claim it is one. The
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
.venv\Scripts\python.exe -m app.main --source loop --scenario retail        --mode spread --density 1 --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario metro        --mode spread --density 1 --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario manufacturing --mode spread --density 1 --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario robotics      --mode spread --density 1 --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario education     --mode spread --density 1 --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario health       --mode spread --density 1 --fullscreen
.venv\Scripts\python.exe -m app.main --source loop --scenario federal      --mode spread --density 1 --fullscreen
```

The seven graphs are:

| Key | Scenario | Video | Measured stage story |
|---:|---|---|---|
| `1` | Retail | `retail-checkout.mp4` | `product-detection-0001` on the NPU → CLIP names the item from a declared store vocabulary on the GPU → shelf/zone events on the CPU |
| `2` | Metro | `smart-city-traffic-montage.mp4` | crossroad-1016 vehicle/pedestrian detection on the NPU → lane/zone counts on the CPU (detector class is the business answer; no weak classifier) |
| `3` | Manufacturing | `store-aisle-detection.mp4` | crossroad-1016 people-and-vehicles detection on the NPU → worker presence on the GPU → zone-occupancy counting on the CPU |
| `4` | Robotics | `one-by-one-person-detection.mp4` | crossroad-1016 detection on the NPU → person detection on the GPU → approach-zone breach events on the CPU |
| `5` | Education | `face-demographics-walking.mp4` | crossroad-1016 detection on the NPU → person detection on the GPU → pose estimation on the NPU → attendance/posture events on the CPU |
| `6` | Health and Life Sciences | `medical-eldercare.mp4` | person detection on the GPU → pose heatmap/PAF decoding on the NPU → posture/zone events on the CPU |
| `7` | Federal and Aerospace | `gov-vehicle-entry.mp4` | person/vehicle/plate detection across NPU and GPU → independent perimeter copies → IoU tracks and plate events on the CPU |

The initial `spread` assignment is explicit and visible in each tile. Detection and pose work use
the NPU where the measured graph calls for it. `person-detection-retail-0013` is explicitly
GPU-preferred everywhere it appears (health, federal, manufacturing, robotics, education) because
it was measured returning **zero detections on the NPU** on this footage regardless of cache
state, while its pose stage remains on the NPU. The operator overlay always reports the actual
`EXECUTION_DEVICES`; the requested preference is never presented as proof.

### Open Edge Platform suite alignment

The seven verticals follow the [Open Edge Platform](https://github.com/open-edge-platform) suite taxonomy:

| OEP suite | Scenario id | What it shows | Measured density |
|---|---|---|---|
| Retail | `retail` | Shelf-side product detection and naming on the store footage | 0.83 det/frame, 17 tracks |
| Metro | `metro` | Intersection vehicle and pedestrian counting on the traffic montage | 34.7 det/frame, 530 tracks |
| Manufacturing | `manufacturing` | People-and-vehicles aisle detection, worker presence, zone occupancy | 4.41 det/frame, 27 tracks |
| Robotics | `robotics` | Human-approach detection and zone-breach events | 1.65 det/frame, 4 tracks |
| Education | `education` | Presence detection plus on-device pose for posture/attendance | 0.88 det/frame, 6 tracks |
| Health and Life Sciences | `health` | Pose estimation and posture monitoring on the eldercare footage | 2.47 det/frame, 22 tracks |
| Federal and Aerospace | `federal` | Multi-stream perimeter detection and plate tracking on the entry footage | 8.79 det/frame, 317 tracks |

Density is mean raw detections per frame over a 20–30 s `tools\review_sessions.py` run, with the
frame and track counts in the table coming from those runs' `summary.json`. Metro is dense because
the traffic montage is a busy multi-vehicle scene; retail is sparse in box count but is the suite
that *names* what it finds, which is the point of that tile.

The three newly added suites are real but visibly sparser than metro and federal, because the
footage available for them contains fewer detectable subjects. Manufacturing is dense enough to
carry a booth tile on its own; robotics and education show one to four subjects at a time, and
robotics in particular has quiet stretches where the frame is briefly empty. This is stated here
rather than hidden: no throughput or detection figure in this demo is inflated to cover it.

### Live failover controls

Press `G` during `SPREAD` or `SPLIT` to disable GPU assignment. GPU-preferred auxiliary stages then
fail over to NPU when NPU is available. Press `N` to disable NPU assignment; NPU-preferred auxiliary
stages then fail over to GPU. If both accelerators are disabled, they fall back to CPU. The status
panel shows `GPU OFF · NPU FAILOVER`, `NPU OFF · GPU FAILOVER`, or `NPU/GPU OFF · CPU FALLBACK`, and
the stage tiles continue to show the measured execution devices.

One exception is deliberate: a model is never moved to a device where it was **measured to produce
no usable output**, even when that device is the natural fallback. `person-detection-retail-0013`
compiles on the NPU and reports `EXECUTION_DEVICES=['NPU']` but emits no boxes there (measured 0
detections against 78 on GPU and 78 on CPU over the same frames), so with the GPU off it moves to
the CPU instead of the NPU — slower, but it detects. That fact is recorded as verified data in
`config/models.json` under `no_usable_output_on`, with the measurement that justifies it. The
substitution is always visible: the `FALLBACK` indicator lights and the tile reports `Person: CPU`.

`NPU_ONLY` mode is the one place this rule does not apply, because the operator has explicitly asked
for NPU-only; that tile renders empty and says so rather than quietly becoming something else.

### Attract and unattended run

Attract mode hides the gauges, shows a large vertical headline, cycles deterministically through
all seven scenarios, and continues processing/telemetry in the background. Any key returns to the
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
| `1`–`4` | Select retail, metro, health and life sciences, or federal and aerospace |
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
acceptance concerns the seven scenario durations, density-4 operation, attract-mode stability, and
RSS criterion, while retaining the same honest metric definitions.

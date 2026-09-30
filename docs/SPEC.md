<!-- Original build contract. It is the brief this project was built from, kept verbatim.
     Current truth lives in docs/STATE.md; where the two disagree, STATE.md wins and the
     differences are listed in its final section. -->

# Engine Lab — heterogeneous AI demo for Intel Core Ultra (Windows)

<!-- BEGIN PROMPT -->

## 0. Task

Build **Engine Lab** from scratch in a new local repo at `C:\dev\engine-lab`: a single-process
Windows 11 Python application that plays a prerecorded video through a real OpenVINO AI pipeline
and shows, at a glance, **which engine is doing the work — NPU, iGPU, or CPU** — with live
toggles that disable the NPU and/or the GPU so the audience watches the workload move onto the
CPU.

It is a **booth demo** for embedded-IoT trade shows (retail, smart city, medical,
government/defense). A salesperson plugs a Core Ultra laptop into a monitor; passers-by get
30–90 seconds. The app must be legible from two meters, must never crash in front of a customer,
and must never display a number it did not measure or source.

The machine has **no development tools installed**. Phase 0 of this spec installs them, and
every install and every claim must be verified by a script, not by eye.

Deliverables: working app + `tools/preflight.py` that prints a PASS/FAIL table + `README.md`
with the booth run-book. Definition of done is in §12.

---

## 1. Why: what you cannot infer from the repo (read this before writing code)

- **The product story is "three engines, one SoC."** The demo exists to show a *real* AI
  workload distributed across CPU + iGPU + NPU, and to show it degrading gracefully onto the CPU
  when the accelerators are removed. Every design decision serves that story.
- **Determinism beats novelty.** A booth demo must produce the same result every time it is
  started. Use prerecorded MP4 files from disk, never a live camera by default. A camera variant
  exists behind a flag (§5.6) but is never the default.
- **Honest numbers or nothing.** Intel's own Quick Reference Guide footnote defines the TOPS
  figures as *calculated peak* values ("PTOPS … a calculated value based on several assumptions
  about operating conditions"). Peak TOPS on screen must therefore be labelled **peak** and carry
  a source URL, and the headline numbers must be **measured** throughput and latency. A demo that
  invents a utilization number is worse than no demo: the customer will open Task Manager next to
  it. §6.4 lists the cross-check the operator can perform live.
- **It must run on all three generations.** Lunar Lake (the machine you have now), Arrow Lake and
  Panther Lake. The same code, the same scenarios, per-platform data in a config file. On Arrow
  Lake the NPU will fail to compile some models — that is expected and must be handled as a
  normal outcome (§5.5), not an error.
- **A failed NPU compile at the booth is the worst possible failure.** Therefore every model is
  compiled on every available device at startup, and only devices that actually succeeded are
  offered in the UI. See the DeviceAvailability gate (§5.5).
- **Nothing on screen may be animated fiction.** No "demo mode" that fakes load, no random
  jitter to make bars move, no synthetic values when a counter is missing. If a counter is
  missing, the HUD says so (§6.3).

---

## 2. Non-negotiable constraints (pin these; do not "improve" them)

1. **Python 3.12** in a repo-local `.venv`. (OpenVINO 2026.4.0 ships cp310–cp313 win_amd64
   wheels; 3.12 has the widest coverage across the pinned deps.)
2. **Pinned dependencies** (see `requirements.txt` in §13.3): `openvino==2026.4.0`,
   `PySide6==6.11.2`, `psutil==7.2.2`, `numpy`, `opencv-python`, `huggingface_hub`, `pyyaml`.
3. **OpenVINO is the only inference runtime.** The NPU plugin ships inside the `openvino` wheel —
   no separate NPU package.
4. **Static shapes everywhere.** The NPU plugin supports only static-shape models. Every model is
   consumed at the fixed shape recorded in `config/models.json`. Never enable dynamic shapes,
   never reshape at runtime, never batch dynamically.
5. **NPU precision:** supported inference data types are F32/F16, and quantized U8 (INT8 or
   mixed FP16-INT8); the hardware computes in FP16. Ship the FP16 IR as the default for all three
   devices and the INT8 IR as an alternate precision toggle for CPU/GPU only.
6. **Device orchestration is `AUTO` + `ov::device::priorities`.** Do **not** use `HETERO` to split
   one model across NPU and GPU: OpenVINO documents HETERO support on the NPU as *partially
   supported, for certain models*. Heterogeneous execution in this demo is achieved by placing
   **different models (and different streams) on different devices**, which is fully supported.
7. **`EXECUTION_DEVICES` is the only acceptable proof of placement.** Any UI element that claims
   a model is running on a device must be derived from
   `compiled_model.get_property("EXECUTION_DEVICES")` or from the app's own explicit stream→device
   assignment. Never from the string you passed in.
8. **No network at runtime.** All models, labels and videos are downloaded once by
   `tools/download_models.py` and stored under `models/` and `media/`. The app must start and run
   with Wi-Fi disabled.
9. **Model caching on** (`ov::cache_dir = cache/`), so warm start is fast. The NPU blob cache is
   documented as development-only and blob compatibility is **not guaranteed across driver
   versions** — provide `tools/reset_cache.ps1` and say in the README to run it after any NPU or
   graphics driver update.
10. **Windows 11 24H2 or later** (build 26100+). Both Intel drivers must be present: NPU driver
   32.0.100.5540 or later (OpenVINO 2026.4 recommendation) and the current Intel Arc/Iris Xe
    graphics driver. Links and checks are in §13.
11. **No `pywin32`.** Read performance counters and device properties through `ctypes` against
    `pdh.dll`, `setupapi.dll` and `cfgmgr32.dll`. Fewer moving parts on a machine you cannot
    debug at a show.
12. **Qt (PySide6) UI, single process.** The HUD, the video canvas and the telemetry sampler live
    in one process so a screenshot of the app is a screenshot of the whole demo.
13. **Pre/post-processing stays on the CPU and is labelled as such.** Decode, resize and
    non-max-suppression are CPU work; showing them honestly (and separately) is a feature — it
    proves the whole pipeline is local.
14. **Every model, video and URL in `config/models.json` was content-verified** (see the
    `verified_on` field and the `rules` block). Do not substitute sources without re-running
    `tools/verify_sources.py`.
15. **The app runs as a standard user.** Elevation is only ever needed for package installs and
    driver installs; never for running the demo, and never as a workaround for a security control.

---

## 3. Repository layout (create exactly this)

```
engine-lab/
  run_demo.bat                 # one-click: activate venv, start app fullscreen
  setup-demo-machine.ps1       # (already provided) installs dev tools + deps
  requirements.txt             # pinned
  README.md                    # booth run-book: pre-show checklist, keys, recovery
  AGENTS.md                    # (already provided) working rules for the coding agent
  config/
    models.json                # (already provided) verified model + video sources
    platform_profiles.json     # (already provided) per-platform peak specs + sources
    scenarios.json             # you write: the 4 verticals
    telemetry_map.json         # GENERATED by tools/probe_telemetry.py
  app/
    main.py                    # entry point, arg parsing, fullscreen, keyboard
    hud.py                     # the whole HUD (Qt widgets, layout, attract mode)
    theme.py                   # colors, font sizes, spacing tokens
    overlay.py                 # boxes/labels/zones drawn onto the video frame
    engine/
      runner.py                # OpenVINO compile + AsyncInferQueue wrapper
      device_policy.py         # the toggle semantics (§5) — the core of the demo
      availability.py          # DeviceAvailability gate (§5.5)
      pipelines.py             # per-scenario stage graphs
      stages.py                # decode / preprocess / infer / postprocess primitives
      events.py                # business events (item picked up, person counted, ...)
    telemetry/
      cpu.py                   # psutil
      pdh.py                   # ctypes wrapper for pdh.dll (GPU + NPU)
      devices_win.py           # ctypes wrapper for setupapi/cfgmgr32 (NPU phys id)
      npu_fallback.py          # app-measured NPU duty cycle (§6.3)
      sampler.py               # background thread, 5 Hz, publishes a TelemetryFrame
    scenarios/
      retail.py  smart_city.py  medical.py  gov_defense.py
  tools/
    download_models.py         # reads config/models.json, downloads models + labels + videos
    verify_sources.py          # re-verifies every URL (content check, not status code)
    probe_telemetry.py         # writes config/telemetry_map.json (§6.2)
    preflight.py               # PASS/FAIL table for the booth (§13.4)
    benchmark_matrix.py        # device × model × streams table, writes bench_report.md
    reset_cache.ps1
  models/  media/  cache/      # git-ignored, populated by download_models.py
```

**Machine-specific paths to keep in one place:** `engine-lab/` (repo root), the venv at
`.venv\`, and `cache/`. Everything else is repo-relative. The repo lives at `C:\dev\engine-lab`
on the demo machine; do not hardcode that path anywhere except `run_demo.bat` and the README.

---

## 4. The four scenarios (verticals)

Each scenario is a **stage graph** over one looping video, with its own business copy and its own
engine story. All models and videos come from `config/models.json` (already verified — do not
invent names). Read the model's real input shape from the IR at load time; the shapes recorded in
the config are for preflight assertions, not for hardcoding.

| id | vertical | video (from models.json) | stages | primary models | business line on screen |
|----|----------|--------------------------|--------|----------------|-------------------------|
| `retail` | Retail / POS | `retail_aisle` | detect → classify(top-k) → zone event | `yolo11n-fp16` + `efficientnet-b0-fp16` (+ `product-detection-0001` as an alternate detector) | "Self-checkout and loss prevention: detect the item, classify it, log the event — all on one SoC." |
| `smart_city` | Smart city / traffic | `traffic` | detect → classify → count per lane zone | `person-vehicle-bike-detection-crossroad-1016` + `efficientnet-b0-fp16` | "Intersection analytics at the edge: vehicle and pedestrian counting without a server." |
| `medical` | Medical / eldercare | `people_queue` (stand-in; any MP4 in `media/` is accepted) | detect → pose → fall/posture event | `person-detection-retail-0013` + `human-pose-estimation-0001` | "Patient monitoring and fall detection on-device: no video leaves the room." |
| `gov_defense` | Government / defense | `street_cars` | N-stream perimeter: detect + track, multi-camera grid | `person-detection-retail-0013` (per stream) + `vehicle-license-plate-detection-barrier-0106` | "Multi-camera perimeter on one platform: stream density is the metric, not TOPS." |

**Retail is the reference scenario** — it is the customer-facing example the demo was built for:
a camera watches people at a shelf, object detection finds what was picked up, classification
names the item, and the pipeline emits a business event. That recipe (detect with YOLO, classify
the detection with EfficientNet-B0) is Intel's own published self-checkout pipeline shape
(`intel-retail/loss-prevention`), so keep the two-stage structure and name the models in the
operator overlay so a customer can see it is the same approach.

**Business events** (`app/engine/events.py`) — each event is a small dict with `ts`, `type`,
`label`, `confidence`, `zone`, and appears in an on-video ticker and a rolling 60-second count in
the HUD. Required event types: `object_picked_up` (retail: detection whose box starts in a shelf
ROI and then leaves it), `object_classified` (label + confidence), `person_counted` (zone entry),
`vehicle_counted`, `posture_alert` (medical: pose keypoint geometry crossing a fall threshold),
`plate_detected` (gov). The thresholds for each are config values in `scenarios.json`, not magic
numbers in code.

**Scenario switching must be instant and must not reload the process.** Precompile all models for
the scenarios the operator has enabled, keep compiled models in memory, and switch the video
source + overlay + event rules. A cold recompile is allowed only if memory pressure forces it,
and then the HUD shows a progress chip.

---

## 5. Device policy — the core feature

### 5.1 Modes

`app/engine/device_policy.py` exposes a single policy object; the HUD toggles mutate it and the
runner recompiles **only the affected models**. Modes:

| mode | device string passed to OpenVINO | hint | what the audience should see |
|------|----------------------------------|------|------------------------------|
| `auto` | `AUTO:NPU,GPU,CPU` | LATENCY | OpenVINO's own placement; per-model device badges show what it chose |
| `spread` **(default)** | per-stream explicit: stream *i* → `NPU`/`GPU`/`CPU` round-robin | LATENCY (1 stream) / THROUGHPUT (multi) | deterministic, explainable placement; every tile carries the exact device it runs on |
| `npu_only` | `NPU` | LATENCY | NPU bar high, CPU bar low, GPU idle |
| `gpu_only` | `GPU` | LATENCY | GPU bar high, CPU bar low |
| `cpu_only` | `CPU` with `ov::streams::num` = physical core count | THROUGHPUT | NPU and GPU bars flat at ~0, **CPU bar climbing** — this is the money shot |
| `split` | detector → `NPU`, classifier → `GPU`, pre/post → `CPU` | LATENCY | two accelerators busy at once, one model each |
| `off` | both toggles off ⇒ equivalent to `cpu_only` | — | label the mode "NPU off · GPU off" so the cause is obvious |

Two independent boolean toggles (**N** = NPU, **G** = GPU) sit on top of the mode: any mode that
includes a disabled device silently drops it from the priority list, and the mode name in the
header must render the actual device set in use (e.g. `AUTO:GPU,CPU`). Never render a mode name
that does not match the string that was actually compiled.

### 5.2 Toggle contract (write tests for this)

Toggling NPU or GPU off, while running:

1. must take effect within **2 seconds** — stop submitting new work to that device, recompile the
   affected models against the reduced device list, swap them in;
2. must keep the HUD live: the video never freezes, the bars never stop updating;
3. must render the disabled engine as a distinct state — bar greyed, a chip reading
   `OFF BY OPERATOR`, and the last measured value greyed rather than zeroed (zeroing it would be a
   lie: the engine is off, not idle);
4. must show the *consequence*: within ~5 seconds the CPU (or remaining accelerator) bar rises and
   end-to-end latency increases. If it does not, the workload was never on that device — the
   `EXECUTION_DEVICES` assertion in §11 will catch it;
5. must never crash, and must never leave the app in a state where a device is off but the
   compiled model still contains it.

### 5.3 Stream density — the second money shot

A density control (`+` / `-`, values 1, 2, 4, 6, 8) runs N independent copies of the scenario
video, each with its own `AsyncInferQueue`, all processed and all displayed in the tile grid.
Density is what turns "the NPU is fast" into "this platform runs N streams; remove the
accelerators and watch N drop." In `spread` mode assign stream *i* to
`devices[i % len(devices)]` and badge each tile with its exact device. In `auto` mode the tile
badge shows the full device list of the compiled model (that is what `EXECUTION_DEVICES` reports
for a multi-device AUTO compiled model) — say so in the tile tooltip rather than faking a single
device.

### 5.4 CPU mode must be configured honestly

Under `AUTO`, OpenVINO's default CPU stream count is 1, which makes CPU-only mode look
artificially bad. In `cpu_only` set `ov::streams::num` to the number of **physical** cores
(`psutil.cpu_count(logical=False)`), and use `THROUGHPUT` as the performance hint. This is the one
place where a wrong number would actively misrepresent the platform, so it is pinned here.

### 5.5 DeviceAvailability gate (startup)

At startup, for every model × every device in `{NPU, GPU, CPU}`: attempt `compile_model`, record
success/failure with the exception text, and store the result. Then:

- the UI only offers devices that compiled;
- a device that failed for *any* model still appears, but with a per-model badge
  (`NPU ✗ not supported for this model`) — this is a feature to talk about, not a defect to hide:
  it is exactly why the platform has three engines;
- failures are logged to `logs/availability.log` and shown in the operator overlay;
- **never** silently fall back to CPU without saying so. A fallback badge reads
  `ran on CPU (NPU compile failed)`.

Compile time for the whole matrix can be a minute on a cold cache; run the gate with a splash
screen that shows per-model progress, and cache the result in `cache/availability.json` keyed by
(driver versions, OpenVINO version, model hash) so warm start is quick.

### 5.6 Input sources

`--source loop` (default) = the scenario's MP4, looping. `--source camera:0` = a live webcam,
same pipeline, same toggles; the HUD must then say `LIVE CAMERA` in the header instead of the
video file name. Camera mode is an option for a customer meeting, never the booth default.

---

## 6. Telemetry — measure it or label it

`app/telemetry/sampler.py` runs a daemon thread at **5 Hz** and publishes an immutable
`TelemetryFrame` to the UI thread (Qt signal; never touch widgets from the sampler thread).

### 6.1 CPU (always available)

`psutil`: total %, per-core %, frequency, physical/logical core counts, per-process RSS. Show
total % as the CPU gauge and per-core as a small heat strip in the operator overlay (label
P-cores vs LP-E-cores if the topology can be read from `psutil.cpu_freq`/Windows API — if it
cannot be determined reliably, show a plain core grid and say nothing about core types).

### 6.2 GPU and NPU utilization on Windows — the real mechanism

Windows exposes both the GPU and the NPU through the **PDH "GPU Engine" counter set**. There is
no public NPU-specific counter set, so the NPU is identified by **physical adapter id**:

1. Enumerate the compute-accelerator device class with `SetupDiGetClassDevs` using
   `GUID COMPUTE_ACCELERATOR_CLASS_GUID = {0xf01a9d53, 0x3ff6, 0x48d2, {0x9f, 0x97, 0xc8, 0xa7,
   0x00, 0x4b, 0xe1, 0x0c}}` and read
   `DEVPKEY_Gpu_PhyId   = {60b193cb-5276-4d0f-96fc-f173abad3ec6, 3}` (and
   `DEVPKEY_GPU_LUID = {…, 2}` for completeness) with `SetupDiGetDevicePropertyW`.
2. Enumerate the `GPU Engine` PDH object with `PdhEnumObjectItems` (use `PERF_DETAIL_WIZARD`) and
   read the instance names. They look like
   `pid_1234_luid_0x00000000_0x0000ABCD_phys_0_eng_0_engtype_3D`.
3. Read `Utilization Percentage` (preferred) or `Running time` (delta-based fallback) per instance,
   and classify each instance by:
   - `phys_<N>` → matches the NPU's physical id ⇒ **NPU**; otherwise a GPU adapter ⇒ **GPU**;
   - `engtype_*` → `3D`/`Compute`/`Copy`/`VideoDecode` are GPU engines; the GPU's matrix engine
     appears as its own engine type (Task Manager renders it as "GPU 0 - Neural"). Log whatever
     strings you actually find; do not hardcode an engtype you have not seen.
   - `pid_(\d+)` → attribute the sample to the app's own process id for the per-process number.
4. Cache the classification in `config/telemetry_map.json` (written by `tools/probe_telemetry.py`)
   so the app does not re-derive it every launch, and re-probe automatically if the machine's
   device list or driver versions change.

**Probe first, then code.** `tools/probe_telemetry.py` must:
- print every `GPU Engine` instance name with its current value for ~3 seconds,
- print the compute-accelerator devices with their phys ids,
- write `config/telemetry_map.json` with the discovered instance patterns,
- exit non-zero if no NPU instances are found, with a message telling the operator the machine
  does not expose NPU counters and that the app will use the labelled fallback.
Run this on the actual demo machine **before** finishing the HUD, and paste its output into the
README. If the counters look different from the shape above, adapt to what is actually there —
the shapes in this spec come from Microsoft's documented approach and from Task Manager's own
data source, but the machine is the authority.

### 6.3 When a counter is missing — the fallback, labelled

If no NPU counters are present (older Windows build, driver that does not publish them), the NPU
gauge falls back to the **app-measured duty cycle**: sum of NPU inference durations divided by
wall-clock time, computed from the runner's own timestamps. Render it with the label
`NPU busy (app-measured)` and a tooltip explaining that it measures this application's NPU
occupancy, not the system-wide device utilization. It is a real measurement of the app's own
work — never blend it with a system counter, and never present it as Task Manager's number.
If neither source exists, the gauge shows `NPU — no counter` rather than an empty bar.

### 6.4 Measured metrics (the headline numbers)

Per stream and per pipeline stage, measured, never estimated:

- **end-to-end latency** ms, p50/p95 over a 10-second window, and the per-stage breakdown
  (decode / preprocess / infer / postprocess);
- **throughput**: inferences per second and detections per second;
- **streams in real time**: how many of the N streams are holding ≥ 90% of the source frame rate;
- **device placement per model**: from `EXECUTION_DEVICES`, shown as a table in the operator
  overlay (`model → NPU` etc.);
- **optional, only if the property exists**: read `DEVICE_GOPS` from each device in a try/except
  and display it beside the platform's peak spec, labelled as reported by the runtime. If the
  property is absent, omit it entirely.

**Live cross-check to put in the README** (this is the credibility move at the booth): leave
Windows Task Manager open on the second monitor with the **NPU**, **NPU engine**, **GPU** and
**GPU engine** columns enabled (available on current Windows 11 builds), and show that the
app's gauges track Task Manager's. Microsoft documents these columns for exactly this purpose —
verifying that an AI workload reached the NPU instead of falling back to the CPU.

### 6.5 Not allowed

- No screenshots or OCR of Task Manager or any external tool.
- No Linux telemetry paths (`/sys/...`, PMT, `perf_event_open`) — this is a Windows app. (The
  Linux tool `topswatch` reads `npu_busy_time_us` and PMT registers; that approach is
  Linux-only and must not be ported into this app.)
- No interpolated, smoothed-then-presented-as-measured, or randomly jittered values.
- No power (W) or temperature readings unless a real counter is found and labelled with its
  source; do not add a fake "power" tile to fill the layout.

---

## 7. The HUD (glanceable from two metres)

Layout tokens live in `app/theme.py`. At 1920×1080 the engine percentage numerals are ≥ 96 px,
the gauge bars ≥ 28 px tall, video overlay labels ≥ 22 px, and no text anywhere is smaller than
18 px. The app scales the whole layout from a single scale factor so 4K behaves identically.
High contrast, heavy weights, no thin fonts, no more than three accent colours.

**Regions**

1. **Header** — left: platform card (CPU name, iGPU name, NPU name, driver versions, and peak
   TOPS per engine each labelled `peak`); centre: scenario title + the business line from §4;
   right: mode name with the actual device set, a `LIVE`/`ATTRACT` badge, clock, and a red
   `FALLBACK` chip when any stage is not on its intended device.
2. **Main row** — left ~70%: the video canvas with overlays (boxes, class labels + confidence,
   shelf/lane zone outlines, and the event ticker along the bottom of the video). Right ~30%:
   three stacked engine gauges in the order **NPU, GPU, CPU**, each with a big percentage, a bar,
   a 60-second sparkline, and a state chip (`ACTIVE` / `OFF BY OPERATOR` / `NO COUNTER`).
3. **Bottom row** — left: the stream tile grid (up to 8 tiles, each with FPS, device badge and a
   miniature of its own overlay). Right: the metrics strip (end-to-end p50/p95, inferences/s,
   detections/s, stage breakdown bar, streams-in-real-time count).
4. **Ticker** — rotating one-line business copy for the active vertical, 8 seconds each.

**Operator overlay (F1)** — model list with device + compile status, raw telemetry map (PDH
instance names, phys ids, chosen provider), driver versions, cache path, CPU topology, and the
last 20 log lines. This is the screen you open when something looks wrong at a show.

**Keyboard**

| key | action |
|-----|--------|
| `1`–`4` | scenario (retail, smart city, medical, gov/defense) |
| `N` / `G` | toggle NPU / GPU (the core demo action) |
| `C` | cycle device mode (`auto` → `spread` → `npu_only` → `gpu_only` → `cpu_only` → `split`) |
| `+` / `-` | stream density 1 → 2 → 4 → 6 → 8 |
| `A` | attract mode: cycles scenarios with a large headline, no gauges; any key returns to demo mode |
| `F1` | operator overlay · `F11` fullscreen · `Q` quit (asks for confirmation) |

Attract mode is what runs when nobody is watching; the gauges are hidden there because an idle
machine showing 0% everywhere sells nothing.

---

## 8. Configuration schemas (exact)

**`config/scenarios.json`** — array of scenario objects:
`{id, title, vertical, business_line, video_id, stages:[{stage, model_id, device_pref}],
zones:[{name, roi:[x,y,w,h], kind}], event_rules:{picked_up_dwell_s, confidence_min,
fall_angle_deg, …}, ticker:[strings]}`. Zone ROIs are **normalised** (0–1) so they survive any
resize. `device_pref` is the placement used in `spread`/`split` modes.

**`config/telemetry_map.json`** — written by the probe, read by the app:
`{probed_on, windows_build, driver_versions:{npu, gpu}, npu_phys_ids:[ints], gpu_phys_ids:[ints],
counter_source:"pdh_gpu_engine", instances:[{pattern, device, engtype}],
fallback:"npu_duty_cycle", notes}`.

**`config/models.json`** and **`config/platform_profiles.json`** are already written — do not
regenerate them; consume them. `models.json` carries the verified source URLs, the exact file
names inside each HF repo, the OMZ `models_bin` URLs, the media URLs, and the list of models that
are **not** available as prebuilt IR. `platform_profiles.json` carries peak TOPS per platform with
a `source` URL per entry and the rule that a profile may not be displayed unless every value it
shows has a source.

---

## 9. Build order (each phase ends with something runnable)

**Phase 0 — machine setup + verification.** `setup-demo-machine.ps1` (provided) installs Python
3.12, Git, the VC++ redistributable, creates `.venv`, installs pinned deps, and checks for the
Intel GPU and NPU drivers. Then `tools/download_models.py` fetches every model, label file and
video from `config/models.json`, and `tools/probe_telemetry.py` writes the telemetry map.
*Exit criteria:* `tools/preflight.py` prints a PASS/FAIL table where every row is PASS except
rows explicitly allowed to be WARN, and `tools/verify_sources.py` re-confirms every source URL.

**Phase 1 — one stream, one model, three devices.** Retail scenario only, YOLO11n detector, no
density, no toggles yet. A minimal HUD with video + boxes + three gauges. *Exit criteria:* all
three devices run the same model at the same time (one after another is fine), the gauge for the
device in use is visibly non-zero, and `EXECUTION_DEVICES` matches the mode.

**Phase 2 — the toggles and density.** Device policy, toggle contract, stream density,
DeviceAvailability gate. *Exit criteria:* §11 items 4–9 pass.

**Phase 3 — all four scenarios + booth polish.** The remaining verticals, business events,
ticker, attract mode, operator overlay, `README.md` run-book, `benchmark_matrix.py`.
*Exit criteria:* §11 items 10–14 pass; a full 10-minute unattended run in attract mode with
density 4 and no leaks (`psutil` RSS growth < 5% over the run).

**Phase 4 — optional, only after Phase 3 is signed off.**
(a) a local vision-language stage on the NPU to name a product the detector only called "bottle"
(OpenVINO GenAI; Intel's loss-prevention repo has this exact use case) — keep it out of the
default path because it adds a second runtime and its own compile risk;
(b) medical imaging segmentation (e.g. OMZ `brain-tumor-segmentation-0002`) — note this model has
**no prebuilt IR** in `models_bin` and needs conversion before use; it is deliberately deferred so
it cannot put the main demo at risk.

---

## 10. Explicitly rejected — do NOT implement these

- **`HETERO` to split a single model across NPU+GPU.** OpenVINO documents NPU HETERO support as
  partially supported for certain models. Place *different models and streams* on different
  devices instead.
- **Dynamic shapes anywhere.** The NPU plugin supports static shapes only; a dynamic model is a
  compile failure at the booth.
- **`openvino-dev` / `omz_downloader` inside the demo venv.** That package is deprecated (last
  release 2024.6.0) and pins an old OpenVINO. Models come from the verified URLs in
  `config/models.json` via `huggingface_hub` and plain HTTP downloads.
- **A live camera as the default source.** Booth demos must be deterministic; camera is opt-in.
- **A browser or Electron UI, or a Flask/websocket dashboard.** One process, Qt, fullscreen.
- **Reading utilization by screenshotting Task Manager**, or by shelling out to any external
  monitoring tool. Task Manager is the *cross-check the human performs*, not the app's data source.
- **Linux sysfs / PMT telemetry.** Wrong OS.
- **Any fabricated value on screen** — including a "demo mode" with animated load, a smoothed
  value presented as measured, or a zeroed gauge for an engine that is simply switched off.
- **Hardcoded TOPS in Python.** Peak specs live in `platform_profiles.json` with a source URL.
- **Power and temperature tiles** unless a real, labelled counter exists.
- **Randomising scenario order or loop points to look livelier.** Determinism is a requirement.
- **Changing the pinned dependency versions** to resolve an install conflict without recording the
  change and re-running preflight.

---

## 11. Acceptance criteria (turn each into a test or a preflight row)

1. `python tools/verify_sources.py` re-verifies every URL in `config/models.json` by **content**
   (XML bodies start with `<?xml`, media returns `video/mp4`), not by HTTP status alone — the
   storage host returns `200 text/html` for missing files, so a status check is not a check.
2. `python tools/preflight.py` prints a table of: Windows build, Python version, OpenVINO version,
   NPU driver version, GPU driver version, NPU device present, GPU device present, NPU counters
   present, each model file present, each model compiles on each device, and the videos decodable.
   Every row is PASS/WARN/FAIL with an explicit remediation line for FAIL.
3. `python -m app.main --selftest` exits 0 with no network access available.
4. For each device in {NPU, GPU, CPU} and each model, the app compiles and runs ≥ 20 inferences;
   the reported `EXECUTION_DEVICES` equals the device under test for `npu_only`, `gpu_only`,
   `cpu_only`.
5. Toggling `N` off during a run: within 2 s the compiled device list no longer contains NPU, the
   NPU gauge reads `OFF BY OPERATOR` with its last value greyed, and CPU+GPU load rises. Assert on
   the device list, not on the gauge.
6. Toggling `G` off behaves identically for the GPU.
7. With both off, `device_policy` reports device string `CPU` and `ov::streams::num` equals
   `psutil.cpu_count(logical=False)`.
8. Density at 8 streams with both accelerators off does not crash, does not drop below 1 processed
   stream, and reports a monotonically lower streams-in-real-time count than the same run with
   the accelerators on. Record both numbers in `bench_report.md`.
9. `config/telemetry_map.json` exists and every gauge in the HUD has a declared source:
   `pdh_gpu_engine` (system) or `npu_duty_cycle` (app-measured, labelled). No gauge without a
   source.
10. All four scenarios run for ≥ 5 minutes each at density 1 and ≥ 1 minute at density 4 without
    an unhandled exception; the app never exits unexpectedly.
11. `psutil` RSS after 10 minutes in attract mode at density 4 is within 5% of the value after
    2 minutes.
12. Every number rendered in the HUD is traceable to a `TelemetryFrame` field or a config value
    with a `source`. Grep the UI code for numeric literals used as metric values; there must be
    none outside `theme.py` and the config files.
13. The README contains: the pre-show checklist, the key map, the Task Manager cross-check
    procedure, the recovery steps for "NPU compiled yesterday but not today" (reset cache, check
    driver version), and the exact command to run each scenario.
14. `run_demo.bat` starts the app fullscreen on the external monitor at 1920×1080 or higher with
    no console window visible, in under 60 s cold and under 20 s warm.

---

## 12. Sanity targets and the one diagnostic that matters

These are **smell tests from the reference class of hardware**, not assertions to contort the
implementation toward. Measure on the actual machine and record the real numbers in
`bench_report.md` — if the real numbers are lower, report them lower.

| run (Lunar Lake, YOLO11n 640, FP16, 1080p video) | expected shape |
|---|---|
| 1 stream, NPU only | NPU gauge clearly busy, CPU mostly pre/post work, end-to-end latency low |
| 1 stream, GPU only | GPU gauge busy, NPU at ~0, latency comparable to NPU |
| 1 stream, CPU only | CPU high, latency several times the NPU figure |
| 4 streams, accelerators on (`spread`) | streams-in-real-time ≥ 3, CPU well below the CPU-only case |
| 4 streams, both accelerators off | CPU near saturation, streams-in-real-time ≤ the accelerated case |
| 8 streams, both off | frame rate degrades visibly and honestly; no crash |

**The diagnostic threshold:** if `cpu_only` at 4 streams shows the CPU gauge under ~50% on a
Core Ultra 200V part, the CPU stream count is wrong (the default under `AUTO` is 1) — fix
`ov::streams::num`, do not touch the models. If NPU-only shows the CPU gauge as busy as the CPU-only
run, the workload never reached the NPU — check `EXECUTION_DEVICES` before anything else.

---

## 13. Setup (Windows, exact)

### 13.1 Drivers (do this first, by hand, once)

- Intel NPU driver, **32.0.100.5540 or later**:
  `https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html`
- Intel Arc / Iris Xe graphics driver:
  `https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html`
- Microsoft Visual C++ Redistributable (x64) — required by OpenVINO on Windows.
- Verify with `tools/preflight.py`, or by hand: Device Manager should show a
  **Neural processors → Intel(R) AI Boost** device, and Task Manager's Performance tab should show
  an **NPU** entry (current Windows 11 builds).

### 13.2 Machine setup

Preferred: run the provided `setup-demo-machine.ps1` in an elevated PowerShell
(`powershell -ExecutionPolicy Bypass -File .\setup-demo-machine.ps1`).

**If PowerShell scripts are blocked on the machine** (ExecutionPolicy, AppLocker, WDAC, or a
"scripts are disabled" message), use one of these, in order of preference — none of them needs
script execution:

1. `setup-demo-machine.cmd` — the same bootstrap as a batch file (not governed by PowerShell
   ExecutionPolicy). Double-click it, or run it from `cmd.exe`. If even `.cmd` files are blocked
   by AppLocker, go to option 2.
2. **Type the commands by hand** in `cmd.exe` — typing is always allowed:
   ```
   winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
   winget install --id Git.Git -e --accept-package-agreements --accept-source-agreements
   winget install --id Microsoft.VCRedist.2015+.x64 -e --accept-package-agreements --accept-source-agreements
   cd C:\dev\engine-lab
   py -3.12 -m venv .venv
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   Open a **new** cmd window after installing Python so `py` and `python` are on PATH. If `winget`
   itself is blocked, install Python 3.12 from python.org (tick "Add python.exe to PATH") and Git
   from git-scm.com through their normal installers.
3. **No-admin / nothing-installable route — `uv`.** Download the standalone binary zip from
   `https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip`,
   unzip it anywhere, then from `cmd.exe` in the repo folder:
   ```
   uv.exe python install 3.12
   uv.exe venv --python 3.12 .venv
   uv.exe pip install --python .venv\Scripts\python.exe -r requirements.txt
   ```
   `uv` needs no installer, no admin rights and no PowerShell.

**This is the important part:** only the *bootstrap* is a script. The application and every tool
(`tools\verify_sources.py`, `tools\download_models.py`, `tools\probe_telemetry.py`,
`tools\preflight.py`, `run_demo.bat`) runs through `python.exe` / `cmd.exe`, which script-execution
policy does not restrict. Once Python and the venv exist, nothing else in this spec needs
PowerShell. `run_demo.bat` is a batch file; if batch files are blocked by policy, launch the app
directly with `.venv\Scripts\python.exe -m app.main` instead.

Record in the README which route worked on the demo machine, so the next person does not have to
rediscover it.

### 13.3 Pinned requirements

```
openvino==2026.4.0
PySide6==6.11.2
psutil==7.2.2
numpy>=2.0,<3
opencv-python>=4.10,<6
huggingface_hub>=0.30
pyyaml>=6.0
```
(If `opencv-python` 5.x causes an API problem, pin the newest 4.x instead and record the change in
the README.)

### 13.4 Preflight, download, run

```powershell
cd C:\dev\engine-lab
.\.venv\Scripts\Activate.ps1
python tools\verify_sources.py          # re-verify every source URL by content
python tools\download_models.py         # models, labels, videos  ->  models\  media\
python tools\probe_telemetry.py         # ->  config\telemetry_map.json
python tools\preflight.py               # PASS/FAIL table
python tools\benchmark_matrix.py        # device x model x streams -> bench_report.md
run_demo.bat                            # the booth app
```

Booth hygiene the script should also apply (and the README must say so): AC power connected, power
plan set to best performance, sleep and display timeouts disabled, lid open or a stand under the
laptop so it does not throttle, and the Task Manager window parked on the second monitor for the
cross-check.

---

## 14. Definition of done

- Phases 0–3 complete, with the exit criteria of each phase demonstrated by real command output
  (paste the preflight table and the benchmark matrix into `bench_report.md`).
- `README.md` is a run-book a different salesperson can follow cold: setup, pre-show checklist,
  key map, scenario scripts (what to say while toggling the NPU off), recovery steps.
- `logs/` captures availability failures and any unhandled exception with a traceback; the app
  writes a `logs/session-<timestamp>.log` on every run and never silently swallows an error.
- No placeholder code, no `TODO` in a code path the demo uses, no commented-out metrics.
- `git log` shows one commit per phase with a message naming the phase; `AGENTS.md` rules
  (commits, verification, honesty about results) are followed.
- Report in the final message: the real measured numbers per device, which models failed to
  compile on which device, and which telemetry provider was actually used on the demo machine.

## 15. Working rules for the coding agent (save this section as `AGENTS.md` in the repo root)

These rules are binding for the agent doing the work. If you are an agent reading this file and
there is no `AGENTS.md` in the repo, create one containing this section verbatim, then follow it.

1. **Never claim a result you did not observe.** If you did not run it, write "not run". Paste real
   command output for every verification claim. A fabricated result is the worst possible failure
   mode for this project.
2. **Never put a number on screen that was not measured or sourced.** Missing counter ⇒ the UI says
   so. Peak spec ⇒ labelled `peak` with a source URL in `config/platform_profiles.json`.
3. **Static shapes only.** The NPU rejects dynamic shapes.
4. **`AUTO` + `ov::device::priorities`, never `HETERO`** for splitting work on the NPU.
5. **`EXECUTION_DEVICES` is the only proof of placement.** Assert on it.
6. **`config/models.json` and `config/platform_profiles.json` are verified data, not suggestions.**
   Re-run `tools/verify_sources.py` and record the result if a source must change.
7. **No network at runtime.** Models, labels and videos are downloaded once, up front.
8. **Ask before adding a dependency.** The venv is pinned.
9. **One phase at a time** (§9). Each phase ends runnable. After each phase run
   `tools/preflight.py`, paste the table into `bench_report.md`, and commit
   (`phase N: <what works now>`).
10. **Keep layout tokens in `app/theme.py`.** No metric values or magic numbers in UI code.
11. **Telemetry sampling runs on a background thread at 5 Hz** and reaches the UI only through Qt
    signals. Never touch widgets from a non-UI thread.
12. **Never crash in front of a customer.** Catch per-stream exceptions, mark that stream failed,
    keep the rest running, log a traceback.
13. After any Intel driver update, run `tools/reset_cache.ps1` — NPU blob cache compatibility is
    not guaranteed across driver versions.
14. **On the demo machine, script execution may be blocked by policy.** Never depend on running a
    `.ps1` or `.bat` file: issue the commands directly in the shell instead. Batch/PowerShell
    wrapper scripts in this repo are conveniences for a human operator, not the build path — the
    build path is the commands themselves.
15. **The application must run as a standard user.** Never require elevation at runtime, and never
    weaken a security setting (Defender exclusions, ASR rules, execution policy, AppLocker) to make
    something run. If a command needs elevation, it is an install step, not a runtime step — say so
    and let the human run it.

**Verification before declaring a phase done:**

```powershell
python tools\verify_sources.py     # URLs verified by content, not by status code
python tools\probe_telemetry.py    # telemetry map present and consistent
python tools\preflight.py          # every row PASS (WARN only where this spec allows it)
python -m app.main --selftest      # runs with no network, exits 0
```

Then run the actual demo for the duration the phase requires and report the numbers you saw.

<!-- END PROMPT -->

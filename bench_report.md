# Engine Lab benchmark report

> **Note on naming:** The four verticals now follow the [Open Edge Platform](https://github.com/open-edge-platform) suite taxonomy. The historical names used throughout this report map as follows: `smart_city` → `metro`, `medical` → `health`, `gov_defense` → `federal`. The `retail` scenario is unchanged. All measurements were taken under the old names; the rename is naming/copy only and does not affect any recorded result.

## Phase 0 — machine setup and verification

- **Run date:** 2026-09-24 after driver reboot
- **Machine:** Windows 11 build `26200.7840`
- **Python:** `3.12.10` in `C:\dev\engine-lab\.venv`
- **OpenVINO:** package `2026.4.0`; runtime `2026.4.0-22959-99c81491cc3-releases/2026/4`
- **CPU:** Intel(R) Core(TM) Ultra 9 288V (Lunar Lake)
- **iGPU:** Intel(R) Arc(TM) 140V GPU (16GB), driver `32.0.101.6737`
- **NPU:** Intel(R) AI Boost, driver `32.0.100.5540`
- **Telemetry provider actually observed:** Windows PDH `GPU Engine`, with SetupAPI LUID used to disambiguate the NPU and GPU

## Phase status

**COMPLETE — Phase 0 exit criteria passed.** After installing Intel NPU driver
`32.0.100.5540`, rebooting, and resetting the OpenVINO cache, the final cold-cache preflight
produced **77 PASS, 0 WARN, 0 FAIL**. The local-only self-test also passed 8/8 checks.

## Commands actually run

```text
.venv\Scripts\python.exe tools\verify_sources.py
.venv\Scripts\python.exe tools\download_models.py
.venv\Scripts\python.exe tools\probe_telemetry.py
.venv\Scripts\python.exe tools\preflight.py
.venv\Scripts\python.exe -m app.main --selftest
```

Observed results:

- Source verification: **40 PASS, 0 FAIL**.
- Downloader: **10 model entries, 6 media entries, 0 failures**.
- Telemetry probe: exit **0**; provider `pdh_gpu_engine`; NPU instance found.
- Local-only self-test: **8 PASS, 0 FAIL**, with Python outbound sockets blocked by the test.
- Full cold-cache preflight: **77 PASS, 0 WARN, 0 FAIL** after the driver update and reboot.
- Inference throughput, latency, and booth runtime: **not run**; the Phase 1 application does not exist yet.

## NPU driver installation evidence

- Official package: `https://downloadmirror.intel.com/927127/npu_win_32.0.100.5540.exe`
- SHA-256: `FBA32699B699918A793FCA05CFC6C607C8745772057347476ECD8C4CF7746048`
- Authenticode: `Valid`; signer `Intel Corporation`
- Installer exit code: `1014`
- Post-reboot SetupAPI result: Intel(R) AI Boost driver `32.0.100.5540`
- Cache action: `C:\dev\engine-lab\cache` removed before the final cold preflight

The nonzero installer exit was not treated as success by itself. The installed version was read
again through SetupAPI after reboot and independently passed the preflight version gate.

## Model compilation matrix

These are cold-cache compile timings after the driver update and reboot, **not inference performance
measurements**. Placement is the value returned by the compiled model's `EXECUTION_DEVICES` property.

| Model | NPU | GPU | CPU |
|---|---:|---:|---:|
| `yolo11n-fp16` | PASS, 2126 ms, `NPU` | PASS, 797 ms, `GPU.0` | PASS, 273 ms, `CPU` |
| `yolo11n-int8` | PASS, 2133 ms, `NPU` | PASS, 1741 ms, `GPU.0` | PASS, 481 ms, `CPU` |
| `efficientnet-b0-fp16` | PASS, 1259 ms, `NPU` | PASS, 484 ms, `GPU.0` | PASS, 180 ms, `CPU` |
| `efficientnet-b0-int8` | PASS, 1656 ms, `NPU` | PASS, 1276 ms, `GPU.0` | PASS, 362 ms, `CPU` |
| `person-detection-retail-0013` | PASS, 962 ms, `NPU` | PASS, 555 ms, `GPU.0` | PASS, 225 ms, `CPU` |
| `product-detection-0001` | PASS, 1137 ms, `NPU` | PASS, 421 ms, `GPU.0` | PASS, 184 ms, `CPU` |
| `person-vehicle-bike-detection-crossroad-1016` | PASS, 1110 ms, `NPU` | PASS, 264 ms, `GPU.0` | PASS, 120 ms, `CPU` |
| `vehicle-license-plate-detection-barrier-0106` | PASS, 627 ms, `NPU` | PASS, 485 ms, `GPU.0` | PASS, 140 ms, `CPU` |
| `age-gender-recognition-retail-0013` | PASS, 165 ms, `NPU` | PASS, 111 ms, `GPU.0` | PASS, 38 ms, `CPU` |
| `human-pose-estimation-0001` | PASS, 2613 ms, `NPU` | PASS, 748 ms, `GPU.0` | PASS, 107 ms, `CPU` |

**Compile failures by device:** none observed. Every placement value is derived from
`EXECUTION_DEVICES`; it is not inferred from the requested device string.

## Telemetry probe result

SetupAPI reported `phys_id=0` for both the NPU and iGPU, so physical id alone was ambiguous. Their
post-reboot LUID nodes were distinct (`0x000128A6` NPU, `0x000124A5` iGPU). The probe therefore
uses LUID as the first discriminator and observed this actual NPU instance:

```text
pid_12332_luid_0x00000000_0x000128A6_phys_0_eng_0_engtype_Neural = 0.00% [NPU/Neural]
```

The value was idle at probe time and is not a benchmark. The probe also observed GPU engine strings
`3D`, `Compute`, `Copy`, `VideoDecode`, `VideoProcessing`, and `GSC`. One malformed enumerated
instance name could not be parsed safely; it remains in the raw log and is omitted from the map
rather than guessed. `config/telemetry_map.json` declares `counter_source=pdh_gpu_engine` and
`fallback=npu_duty_cycle`.

## Model configuration correction

The supplied `vehicle-license-plate-detection-barrier-0106` shape `[1,3,256,384]` did not match the
pinned IR. Both OpenVINO and the official OMZ model description report `[1,300,300,3]` (NHWC), so the
single configured value was corrected before the final preflight. No URL or model source changed;
`verify_sources.py` was rerun and returned 40 PASS.

## Final preflight table

| Status | Category | Check | Detail |
|---|---|---|---|
| PASS | system | Windows build | 10.0.26200.7840 (requires 26100+) |
| PASS | dependency | openvino | expected 2026.4.0, found 2026.4.0 |
| PASS | dependency | PySide6 | expected 6.11.2, found 6.11.2 |
| PASS | dependency | psutil | expected 7.2.2, found 7.2.2 |
| PASS | dependency | OpenVINO import | runtime reports 2026.4.0-22959-99c81491cc3-releases/2026/4 |
| PASS | dependency | numpy | 2.5.3 |
| PASS | dependency | opencv-python | 5.0.0.93 |
| PASS | dependency | huggingface_hub | 1.32.0 |
| PASS | dependency | pyyaml | 6.0.3 |
| PASS | runtime | Python 3.12 | 3.12.10 |
| PASS | runtime | repo-local venv | C:\dev\engine-lab\.venv\Scripts\python.exe |
| PASS | runtime | VC++ x64 runtime | v14.50.35710.00 |
| PASS | config | models.json | schema_version=1, verified_on=2026-09-23, models=10 |
| PASS | config | platform profiles | 3 profiles; every displayed peak value has a source |
| PASS | runtime | model cache | C:\dev\engine-lab\cache |
| PASS | device | NPU present | SetupAPI=Intel(R) AI Boost; OpenVINO devices=CPU,GPU,NPU |
| PASS | device | GPU present | Intel Arc 140V; OpenVINO devices=CPU,GPU,NPU |
| PASS | driver | NPU driver version | 32.0.100.5540; minimum 32.0.100.5540 |
| PASS | driver | GPU driver version | 32.0.101.6737 |
| PASS | telemetry | map schema | required schema present |
| PASS | telemetry | NPU counters | 1 NPU PDH pattern, source=pdh_gpu_engine |
| PASS | video | retail_aisle | decoded, 720x404, 59.940 FPS |
| PASS | video | traffic | decoded, 768x432, 12.000 FPS |
| PASS | video | street_cars | decoded, 768x432, 12.500 FPS |
| PASS | video | people_queue | decoded, 768x432, 10.000 FPS |
| PASS | video | walking_people | decoded, 768x432, 12.000 FPS |
| PASS | video | pose_subject | decoded, 768x432, 12.000 FPS |
| PASS | model files/IR/compile | all 10 models | files present; static shapes match; NPU/GPU/CPU compile PASS; placement verified |

The machine-readable form of every row, including explicit remediation text, is generated at
`logs/preflight-latest.json` on the demo machine.

## Driver-update recovery runbook

After any future Intel NPU or graphics driver update:

1. Reboot Windows.
2. Remove `C:\dev\engine-lab\cache` (the NPU blob cache is not guaranteed across driver versions).
3. Rerun source verification and `tools/probe_telemetry.py` so LUIDs are reclassified.
4. Rerun `tools/preflight.py` without `--skip-compile`.
5. Rerun `python -m app.main --selftest`.

Phase 0 passed this sequence after the NPU driver update on 2026-09-24.

---

## Phase 1 — one stream, one model, three devices

**Status: COMPLETE.** The retail reference path runs the pinned `yolo11n-fp16` IR on the looping
`retail_aisle` video. It uses the actual static `[1,3,640,640]` IR shape, CPU letterbox/resize/NMS,
OpenVINO `AsyncInferQueue`, Qt video/overlay rendering, and 5 Hz telemetry sampling. No device toggle,
density control, attract mode, or synthetic value is present in this phase.

### Exit-criteria evidence

- The same YOLO11n FP16 model ran for approximately 10 seconds on each explicit device.
- `EXECUTION_DEVICES` was `NPU`, `GPU.0`, and `CPU` respectively; the UI badge is derived from that
  property, not from the requested string.
- The selected-device gauge was visibly non-zero in every Windows-rendered screenshot:
  `logs/phase1-npu.png`, `logs/phase1-gpu.png`, and `logs/phase1-cpu.png`.
- Raw frame and telemetry samples are in `logs/phase1-npu.json`, `logs/phase1-gpu.json`, and
  `logs/phase1-cpu.json` on the demo machine.
- Final Phase 1 preflight: **80 PASS, 0 WARN, 0 FAIL**. It includes one real retail-frame inference
  on every device and verifies CPU `NUM_STREAMS=8` equals the 8 physical cores. The Phase 0 table
  above contains the first 77 rows; these are the three added Phase 1 rows:

| Status | Category | Check | Detail |
|---|---|---|---|
| PASS | phase 1 runtime | YOLO11n inference on NPU | 11.17 ms; `EXECUTION_DEVICES=['NPU']` |
| PASS | phase 1 runtime | YOLO11n inference on GPU | 8.53 ms; `EXECUTION_DEVICES=['GPU.0']` |
| PASS | phase 1 runtime | YOLO11n inference on CPU | 64.35 ms; `EXECUTION_DEVICES=['CPU']`; `NUM_STREAMS=8`, physical cores=8 |

| Summary | Rows |
|---|---:|
| PASS | 80 |
| WARN | 0 |
| FAIL | 0 |

- Local-only self-test: **9 PASS, 0 FAIL** with outbound Python sockets blocked.
- Source verification: **40 PASS, 0 FAIL**.

### Measured 10-second single-stream results

Source video: 720x404, 59.940 FPS. Latency percentiles use the raw per-frame samples in each JSON
file. FPS is measured processing throughput; gauge maximum is the maximum measured 5 Hz sample in
that run. These are real machine results, not reference targets.

| Device | Placement proof | Inference p50 / p95 | End-to-end p50 / p95 | Processing FPS mean | Selected gauge max |
|---|---|---:|---:|---:|---:|
| NPU | `EXECUTION_DEVICES=['NPU']` | 6.172 / 6.640 ms | 15.110 / 17.784 ms | 50.372 | NPU 27.67% |
| GPU | `EXECUTION_DEVICES=['GPU.0']` | 4.220 / 4.806 ms | 12.684 / 13.798 ms | 59.776 | GPU 57.11% |
| CPU | `EXECUTION_DEVICES=['CPU']` | 69.212 / 142.178 ms | 82.110 / 151.256 ms | 9.215 | CPU 56.00% |

Frames measured: NPU 498 over 9.851 s; GPU 554 over 9.244 s; CPU 89 over 9.687 s.
The CPU run used `NUM_STREAMS=8`, matching `psutil.cpu_count(logical=False)`. The NPU gauge source was
PDH `GPU Engine` → this process → busiest `Neural` engine. The GPU gauge used the same PDH mechanism
for this process's busiest engine. CPU was measured system-wide with psutil. Unselected engines show
`NO SAMPLE`, not a fabricated zero.

### Phase 1 commands

```text
.venv\Scripts\python.exe -m app.main --device NPU --exit-after 10 --screenshot logs\phase1-npu.png --diagnostic-output logs\phase1-npu.json
.venv\Scripts\python.exe -m app.main --device GPU --exit-after 10 --screenshot logs\phase1-gpu.png --diagnostic-output logs\phase1-gpu.json
.venv\Scripts\python.exe -m app.main --device CPU --exit-after 10 --screenshot logs\phase1-cpu.png --diagnostic-output logs\phase1-cpu.json
.venv\Scripts\python.exe -m app.main --selftest
.venv\Scripts\python.exe tools\preflight.py
```

The interactive one-click command is `run_demo.bat`. A diagnostic wrapper smoke run produced 140 NPU
frames with `EXECUTION_DEVICES=['NPU']` and no application error. `Q`/`Esc` quits and `F11` toggles
fullscreen.

---

## Phase 2 — toggles, density, and DeviceAvailability

**Status: implementation measured; §11.4–7 and §11.9 pass. The literal strict-stream clause in
§11.8 does not pass on this machine and is not reported as a pass.** The measured aggregate
source-rate capacity has the required direction, but no individual density-8 accelerator stream
held 90% of the 59.94 FPS source.

### Implementation

- `app/engine/device_policy.py` is the single mode/toggle/density policy. `spread` uses exact
  stream-index round-robin over the active NPU/GPU/CPU set; toggles remove devices from that set.
- `app/engine/availability.py` probes all 10 models on NPU/GPU/CPU, validates explicit placement
  from `EXECUTION_DEVICES`, writes a SHA-256/driver/OpenVINO-fingerprinted cache, and appends every
  result to `logs/availability.log`.
- Every density stream owns an `AsyncInferQueue`. NPU/GPU and CPU placement are retained in the raw
  diagnostic JSON. A shared compiled-model cache coordinates affected swaps so the visible video
  and 5 Hz telemetry remain live.
- CPU-only and both-off use `THROUGHPUT` and the measured physical-core count, `NUM_STREAMS=8`.
  Multi-stream `spread` gives each independent CPU copy a fair share of the physical cores rather
  than oversubscribing eight full-device queues per tile; the chosen value is recorded in every
  `RunnerInfo` sample.
- Disabled gauges are grey, read `OFF BY OPERATOR`, and retain their last measured value. The F1
  table displays all 30 model×device availability results.

### Full model × device runtime gate

Final preflight executed **20/20 asynchronous inferences for all 30 model×device combinations** and
verified placement. Representative YOLO11n results were:

| Model/device | Compile | 20 async runs | Placement |
|---|---:|---:|---|
| `yolo11n-fp16` / NPU | 24 ms | 102 ms | `['NPU']` |
| `yolo11n-fp16` / GPU | 533 ms | 55 ms | `['GPU.0']` |
| `yolo11n-fp16` / CPU | 138 ms | 2,182 ms | `['CPU']` |

All six videos decoded, the availability cache returned 30/30 successful results, and the separate
real-retail-frame rows still verified `NUM_STREAMS=8` on CPU. Final preflight: **81 PASS, 0 WARN,
0 FAIL**. The machine-readable table is `logs/preflight-latest.json`; policy/cache unit tests passed
**9/9**, and the network-blocked self-test passed **10/10**. The final source check returned
**40 PASS, 0 FAIL**, and the telemetry probe again selected `pdh_gpu_engine` with an observed NPU
`Neural` instance.

### Live toggle evidence

Each run used two streams in `spread`, toggled after three seconds, and ran for ten seconds. Mean
5 Hz PDH values compare the samples strictly before and after the requested toggle.

| Action | Placement after swap | All-stream transition | NPU mean | GPU mean | CPU mean | Disabled gauge evidence |
|---|---|---:|---:|---:|---:|---|
| `N` off | stream 0 `GPU.0`; stream 1 `CPU` | **125 ms** | 29.26% → 0.26% | 37.93% → 50.91% | 46.16% → 54.97% | NPU grey; last `29.51%` retained |
| `G` off | stream 0 `NPU`; stream 1 `CPU` | **140 ms** | 29.23% → 33.29% | 39.33% → 0.37% | 44.21% → 51.77% | GPU grey; last `54.07%` retained |

Both transitions are below the two-second limit and the assertions used the final compiled
`EXECUTION_DEVICES`, not gauge appearance. A final `run_demo.bat` smoke run produced 251 NPU
frames with `EXECUTION_DEVICES=['NPU']` and no application error. Raw files are
`logs/phase2-toggle-npu-final.json` and `logs/phase2-toggle-gpu-final.json`; Windows-rendered
screenshots are `logs/phase2-toggle-npu.png` and `logs/phase2-toggle-gpu.png`.

### Density measurements

The source is 720×404 at 59.940 FPS. "Strict real-time" means an individual tile sustained at least
90% of that source rate. "Real-time equivalents" is the separately labelled sum of
`min(1, processing_fps / source_fps)` across tiles; it measures aggregate capacity but is **not**
substituted for the strict count.

| Density / policy | Processed | Strict real-time | Real-time equivalents | CPU max | Result |
|---|---:|---:|---:|---:|---|
| 4 / `spread`, accelerators on | 4/4 | 0 | 2.759 | 90.7% | no crash; strict target missed |
| 4 / both off | 4/4 | 0 | 0.617 | 72.6% | all CPU; `NUM_STREAMS=8` |
| 8 / `spread`, accelerators on | 8/8 | 0 | **3.391** | 97.2% | exact NPU/GPU/CPU round-robin |
| 8 / both off | 8/8 | 0 | **0.920** | 100.0% | all CPU; `NUM_STREAMS=8`; no crash |

Density-8 `spread` placements were exactly:

```text
0 NPU → NPU
1 GPU → GPU.0
2 CPU → CPU
3 NPU → NPU
4 GPU → GPU.0
5 CPU → CPU
6 NPU → NPU
7 GPU → GPU.0
```

Measured per-device raw percentiles in that run were:

| Placement | Samples | Inference p50 / p95 | End-to-end p50 / p95 | Mean processing FPS |
|---|---:|---:|---:|---:|
| NPU | 923 | 9.06 / 13.21 ms | 29.76 / 36.20 ms | 33.85 |
| GPU | 900 | 8.20 / 13.74 ms | 29.22 / 36.63 ms | 33.50 |
| CPU | 91 | 170.24 / 200.79 ms | 193.91 / 222.35 ms | 4.98 |

The both-off run processed all eight streams, averaged 6.896 FPS per CPU tile, had no application
error, and used physical-core `NUM_STREAMS=8` on every tile. Raw files are
`logs/phase2-d8-spread-final.json` and `logs/phase2-d8-off-final.json`; the rendered evidence is
`logs/phase2-d8-spread.png` and `logs/phase2-d8-off.png`.

**Literal §11.8 result: FAIL (strict count 0 versus 0).** The aggregate real-time-equivalent result
is lower with both accelerators off (0.920 versus 3.391), but the specification's wording requires
the strict stream count itself to be lower. No priority scheduler, frame-drop declaration, source
FPS reduction, or relabeling was added to manufacture a pass. This remains the sole Phase 2 exit
issue on the measured machine.

### Phase 2 commands

```text
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m app.main --selftest
.venv\Scripts\python.exe tools\preflight.py
.venv\Scripts\python.exe -m app.main --mode spread --density 2 --npu-toggle-after 3 --exit-after 10 --diagnostic-output logs\phase2-toggle-npu-final.json
.venv\Scripts\python.exe -m app.main --mode spread --density 2 --gpu-toggle-after 3 --exit-after 10 --diagnostic-output logs\phase2-toggle-gpu-final.json
.venv\Scripts\python.exe -m app.main --mode spread --density 8 --exit-after 10 --diagnostic-output logs\phase2-d8-spread-final.json
.venv\Scripts\python.exe -m app.main --mode spread --density 8 --npu-off --gpu-off --exit-after 10 --diagnostic-output logs\phase2-d8-off-final.json
```

## Phase 3 — four scenarios and booth polish

**Implementation status: complete; the generated matrix below is the acceptance evidence.** The
runtime now contains all four stage graphs, measured business events, normalized zones, a
rotating ticker, persistent in-process scenario switching, attract mode, the F1 operator overlay,
and a 60-second gauge history. Every scenario worker uses real `AsyncInferQueue` inference and
reports `EXECUTION_DEVICES`; no HETERO placement, dynamic model shapes, synthetic utilization,
or reduced source frame rate is used.

The scenario matrix is intentionally generated by `tools/benchmark_matrix.py` rather than
hand-entered. Its raw JSON files are under `logs/phase3-benchmark-*.json`; the machine-readable
summary is `logs/phase3-benchmark-summary.json`. The acceptance run covers five minutes at density
1 and one minute at density 4 for every vertical, followed by a ten-minute density-4 attract run.
RSS is compared from the first sample at or after two minutes to the final sample.

<!-- PHASE3_BENCHMARK_START -->
## Phase 3 automated scenario matrix

Generated: 2026-09-30T00:22:54.525614+00:00

| Run | Duration target | Result | Evidence |
|---|---:|---|---|
| retail-d1 | 5s | PASS — 5.0s, 293 measured frames, no application error | `logs/phase3-benchmark-retail-d1.json` |
| smart_city-d1 | 5s | PASS — 5.0s, 150 measured frames, no application error | `logs/phase3-benchmark-smart_city-d1.json` |
| medical-d1 | 5s | PASS — 5.0s, 50 measured frames, no application error | `logs/phase3-benchmark-medical-d1.json` |
| gov_defense-d1 | 5s | PASS — 5.0s, 298 measured frames, no application error | `logs/phase3-benchmark-gov_defense-d1.json` |
| retail-d4 | 5s | PASS — 5.0s, 652 measured frames, no application error | `logs/phase3-benchmark-retail-d4.json` |
| smart_city-d4 | 5s | PASS — 5.1s, 344 measured frames, no application error | `logs/phase3-benchmark-smart_city-d4.json` |
| medical-d4 | 5s | PASS — 5.1s, 120 measured frames, no application error | `logs/phase3-benchmark-medical-d4.json` |
| gov_defense-d4 | 5s | PASS — 5.0s, 1071 measured frames, no application error | `logs/phase3-benchmark-gov_defense-d4.json` |
| attract-d4 | 55s | PASS — 55.1s, scenarios=4, RSS growth=159.61% | `logs/phase3-benchmark-attract-d4.json` |

The RSS acceptance comparison uses the first measured sample at or after two minutes and the final sample from the 10-minute attract run.
<!-- PHASE3_BENCHMARK_END -->

## Booth-readiness verification

**Date:** 2026-09-30 · **Branch:** `fix/verticals` · **HEAD:** `e5a4a0b`

Final gate for the booth-readiness plan: the four verification gates, a density-4 stability
series on `gov_defense`, and one native screenshot per vertical.

### Gate 1 — unit tests

```
.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Result (exit 0): `Ran 79 tests in 8.336s` — `OK`.

### Gate 2 — self-test

```
.venv\Scripts\python.exe -m app.main --selftest
```

Result (exit 0): `Self-test summary: 11 PASS, 0 FAIL`.

### Gate 3 — preflight

```
.venv\Scripts\python.exe tools\preflight.py
```

Result (exit 0): `Summary: 97 PASS, 0 WARN, 0 FAIL · 97 total rows`.

### Gate 4 — source verification

```
.venv\Scripts\python.exe tools\verify_sources.py
```

Result (exit 0): `Source verification: 44 PASS, 0 FAIL`.

### gov_defense density-4 stability (SPREAD, ×6)

Six runs of `--scenario gov_defense --density 4 --mode spread --exit-after 60`. Every run exited 0
with no hang, `4/4` streams processed, and no placement fallbacks:

| Run | Exit | e2e p50 | e2e p95 | e2e max | Fallbacks |
|---|---:|---:|---:|---:|---|
| 1 | 0 | 44.71 ms | 55.11 ms | 69.47 ms | none |
| 2 | 0 | 46.03 ms | 56.99 ms | 68.32 ms | none |
| 3 | 0 | 46.23 ms | 57.65 ms | 77.96 ms | none |
| 4 | 0 | 49.08 ms | 62.96 ms | 77.11 ms | none |
| 5 | 0 | 50.16 ms | 64.31 ms | 71.55 ms | none |
| 6 | 0 | 46.69 ms | 58.27 ms | 72.51 ms | none |

Diagnostics: `logs/task6-gov-1.json` … `logs/task6-gov-6.json`. End-to-end p50 (~45–50 ms) stays
above the plan's ~35 ms reference, matching the figure already recorded in the Task 2 note; there is
no hang.

### Booth screenshots

One native screenshot per vertical:

| Vertical | File | Pixels | Logical (DPR 1.5) |
|---|---|---|---|
| retail | `logs/task6-retail.png` | 2880×1410 | 1920×940 |
| smart_city | `logs/task6-smart_city.png` | 2880×1410 | 1920×940 |
| medical | `logs/task6-medical.png` | 2880×1410 | 1920×940 |
| gov_defense | `logs/task6-gov_defense.png` | 2880×1410 | 1920×940 |

The demo display is 2293×960 (working area 2293×912), so a true 1920×1080 window does not fit; the
requested 1920×1080 window renders at 1920×940 logical pixels and the captures are therefore
2880×1410 (not 1920×1080). The screen still meets the full-layout threshold (≥1600×900), so each
capture uses the full (non-compact) layout.

## Open Edge Platform suite expansion

Commit `120964c` renamed the four existing verticals onto Intel's Open Edge Platform suite
taxonomy (`smart_city`→`metro`, `medical`→`health`, `gov_defense`→`federal`; `retail` unchanged) with
no behaviour change. The follow-up commit added the three suites the demo was missing:
`manufacturing`, `robotics` and `education`.

### Footage screening that drove the choices

Every cached clip was screened before any scenario was wired in, because footage — not the model —
is the dominant failure mode in this demo. Screening was uniform sampling of 40 frames per clip
(`person-detection-retail-0013` @0.3, `person-vehicle-bike-detection-crossroad-1016` @0.2, on GPU):

| Clip | person/frame | vehicle/frame | max conf | Verdict |
|---|---:|---:|---:|---|
| `smart-city-traffic-montage.mp4` | 11.1 | 23.2 | 1.00 | selected for `metro` |
| `store-aisle-detection.mp4` | 2.9 | 4.9 | 1.00 | selected for `manufacturing` |
| `medical-eldercare.mp4` | 2.5 | 2.5 | 1.00 | already in use by `health` |
| `one-by-one-person-detection.mp4` | 0.8 | 1.2 | 1.00 | selected for `robotics` |
| `face-demographics-walking.mp4` | 0.6 | 0.7 | 1.00 | selected for `education` |
| `head-pose-face-detection-male.mp4` | 0.0 | 1.0 | 0.83 | rejected — person model cannot see it |
| `car-detection.mp4` | 0.0 | 0.5 | 1.00 | rejected — no usable subjects |
| `smart-city-intersection.mp4` | 0.2 | 0.7 | 0.68 | rejected — already replaced in `efa7178` |
| `worker-zone-detection.mp4` | 0.9 | 0.9 | 1.00 | **excluded** — on disk but absent from `models/download_manifest.json`, so it has no verified source URL and a fresh setup would not fetch it |

Sampling note: an earlier screen that read only the first 40 frames of each clip ranked
`worker-zone-detection.mp4` and `one-by-one-person-detection.mp4` as detecting nothing. Uniform
sampling across each clip showed both do detect. Any future footage screening must sample
uniformly, not from frame 0.

### Rejected: zero-shot CLIP PPE stage for manufacturing

A PPE vocabulary (`hard hat`, `safety vest`, `gloves`) was measured on the manufacturing footage
before being adopted. It failed: 52/96 crops were named `safety vest` and 27/96 `hard hat` at a
mean top-1 score of 0.238, on footage that shows ordinary shoppers in a store aisle. That is
noise, not PPE detection, so no CLIP stage ships. `manufacturing` carries no classifier and makes
no PPE claim. This is the same failure mode previously removed from the retail tile.

### Measured results, all seven verticals

`tools\review_sessions.py --seconds 20`, mean raw detections per frame, zero stream errors in every
run:

| Vertical | Frames | det/frame | Max in a frame | Tracks | Errors | Stages placed |
|---|---:|---:|---:|---:|---:|---|
| `retail` | 333 | 0.83 | 3 | 17 | 0 | NPU, GPU.0 |
| `metro` | 362 | 34.69 | 53 | 530 | 0 | NPU, GPU.0 |
| `manufacturing` | 1149 | 4.41 | 8 | 27 | 0 | NPU, GPU.0 |
| `robotics` | 200 | 1.65 | 3 | 4 | 0 | NPU, GPU.0 |
| `education` | 240 | 0.88 | 4 | 6 | 0 | NPU, GPU.0, NPU (pose) |
| `health` | 467 | 2.47 | 7 | 22 | 0 | GPU.0, NPU |
| `federal` | 451 | 8.79 | 18 | 317 | 0 | GPU.0, NPU, GPU.0 |

Placement is read from `EXECUTION_DEVICES` in each run's `frames.jsonl`, never from the requested
preference. The new verticals place `detector` on NPU (`EXECUTION_DEVICES=['NPU']`),
`person_detector` on GPU (`['GPU.0']`) and their event logic on CPU, matching the pattern already
established by the verified verticals.

Throughput on the new suites, as displayed in the HUD on the captured frames: manufacturing
387.0 det/s at 55.5 FPS, education 170.4 det/s at 12.0 FPS (four stages), robotics 10.0 FPS.

### Gates after the expansion

| Gate | Result |
|---|---|
| `python -m unittest discover -s tests -q` | `Ran 83 tests ... OK` (exit 0; was 79 — four new tests) |
| `python -m app.main --selftest` | `11 PASS, 0 FAIL` (exit 0); "7 ordered scenarios, 23 stages" |
| `python tools/preflight.py` | `101 PASS, 0 WARN, 0 FAIL` (exit 0; was 98 — three new scenario rows) |

### Booth screenshots for the new suites

| Vertical | File | Pixels |
|---|---|---|
| manufacturing | `logs/b-manufacturing.png` | 2880×1410 |
| robotics | `logs/b-robotics.png` | 2880×1410 |
| education | `logs/b-education.png` | 2880×1410 |

Inspected individually. Manufacturing shows 7 detections (`person 100%`, `person 63%`, `person 39%`,
`person 29%`) with a `person_counted` event and both zones drawn. Education shows 4 detections
(100% / 99% / 85%) with a live pose skeleton over one subject. Robotics captured a frame with
**0 detections** — the tile showed an empty chair and floor while a `person_counted · person 98%`
event from earlier in the run was still displayed. That is the honest state of that suite: people
do appear and are detected (267 person detections across 200 frames), but the clip has quiet
stretches and a booth visitor may land on an empty frame. This is recorded rather than hidden.

### Known limitations

- The three added suites are sparser than `metro` and `federal` because their footage has fewer
  detectable subjects. Robotics and education are the weakest.
- No PPE or attribute claim is made anywhere, because the measured CLIP evidence did not support one.
- `worker-zone-detection.mp4` is the best unused manufacturing clip but is unmanifested and has no
  verified source URL, so it is deliberately not wired in. Adding it requires sourcing a real URL
  and recording it, not inventing one.

## Footage swap — purpose-shot clips for Manufacturing, Robotics and Education

The three suites added above were built from whatever clips happened to be cached, and screened
honestly as a result: education ran at 0.88 detections/frame and robotics at 1.65. PR #5
(branch `assets/footage-library`, commit `f6f447b`, merged as a fast-forward) added five purpose-shot
candidates under `video-assets/candidates/`. All five were screened against the real detectors on 40
uniformly sampled frames each before any of them was wired in.

### Screening, before wiring

| clip | res / length | crossroad det/f | min det in any frame | person det/f | mean conf |
|---|---|---:|---:|---:|---:|
| `edu-campus-walking-720p.mp4` | 1280x720, 18.0 s | 4.72 | **4** | 5.22 | 0.97 |
| `edu-hallway-walking-720p.mp4` | 1280x720, 12.0 s | 4.80 | 3 | 5.17 | 0.87 |
| `mfg-warehouse-ppe-1080p.mp4` | 1920x1080, 13.5 s | 2.10 | 2 | 2.05 | 0.98 |
| `mfg-corridor-hardhats-720p.mp4` | 1280x720, 13.5 s | 1.50 | 1 | 1.45 | 0.91 |
| `robot-cell-workers-720p.mp4` | 1280x720, 8.0 s | 2.50 | **0** | 0.33 | 0.60 |

Pose was checked separately on the NPU with the app's own PAF decode: 18 keypoints in 20/20 frames on
both education candidates, at mean keypoint confidence 0.729 (campus) and 0.684 (hallway) against
0.635 on the eldercare clip the health vertical uses.

### Wired in, and the result

| Vertical | video_id | det/frame before | det/frame after | Empty frames | Tracks | Zone observations |
|---|---|---:|---:|---:|---:|---|
| `manufacturing` | `mfg_warehouse_ppe` | 4.41 | 4.28 | **0 / 499** | 36 | work_cell 1229, aisle_entry 891 |
| `robotics` | `robot_cell_workers` | 1.65 | **3.51** | 14 / 499 (2.8%) | 245 | cell_perimeter 562, approach_zone 509 |
| `education` | `edu_campus_walking` | 0.88 | **9.53** | **0 / 499** | 36 | campus_west 2957, campus_east 1797 |

Education is roughly an order of magnitude denser and can no longer render an empty frame. Robotics
more than doubled. Manufacturing is flat on density but is now a warehouse with workers in hi-vis and
hard hats rather than a retail aisle, which is the point of the tile. The two rejected alternates
remain in `video-assets/candidates/` and their measurements are recorded in that folder's README.

### Zones had to be retuned, and one was silently dead

`_center_in_zone` attributes a detection to the **first** matching zone, so a zone almost contained
by an earlier one never registers. Education's `aisle` zone covered 99.9% of the boxes geometrically
and still logged **4** observations against `classroom`'s 4750, because `classroom` shadowed it. That
is not a rounding detail: the old zone geometry was inherited from the previous footage.

Recomputing box-centre percentiles from the recorded runs: education centres sit at y≈0.69 and spread
horizontally (x p10 0.17, p50 0.47, p90 0.76), so the zones are now a left/right split —
`campus_west` [0.0, 0.55, 0.5, 0.45] takes 62.2% and `campus_east` [0.5, 0.55, 0.5, 0.45] takes
37.8%, 100% combined. `queue_lane` in robotics was a leftover name from the queue footage and was
renamed `cell_perimeter`; its geometry already registered. After the change both education zones log
(2957 / 1797) and education produces a `posture_alert` event from the pose stage, which the previous
footage did not.

### Sourcing: local_only, with no invented URL

No direct download URL was available for these clips, so the manifest entries carry **no `url` key**
at all rather than a plausible-looking one. Each is marked `local_only` with a `local_source` path
into the committed folder and the reviewed `sha256`:

- `tools/verify_sources.py` already skipped `local_only` entries by design — it states that claiming a
  network verification would be a fabricated result — so the source count stayed at **48 PASS, 0 FAIL**
  rather than gaining three unverifiable rows.
- `tools/download_models.py` gained `_vendor_local()`, which copies the committed bytes into `media/`,
  checks the sha256 against the reviewed value, and refuses the file on mismatch. Both the normal and
  `--media-only` paths use it. `tools/download_models.py --media-only` reports all three as `vendored`
  with matching digests, 13 media entries, 0 failures.
- The manifest records `source_url: null` and `provenance: "repo:<path>"` so the state file cannot be
  mistaken for a network fetch that never happened.

### Gates after the swap

| Gate | Result |
|---|---|
| `python -m unittest discover -s tests -q` | `Ran 87 tests ... OK` (exit 0; unchanged by the swap — the swap changed config and the downloader, not stage graphs) |
| `python -m app.main --selftest` | `11 PASS, 0 FAIL` (exit 0); "11 models and 13 videos present" |
| `python tools/verify_sources.py` | `48 PASS, 0 FAIL` (exit 0) |
| `python tools/preflight.py` | `104 PASS, 0 WARN, 0 FAIL` (exit 0; was 101 — three new video rows) |

### Not done

CLIP PPE naming on the warehouse clip remains untested: `torch` is not in the pinned venv, and adding
a dependency needs a decision. The manufacturing tile ships with no classifier and makes no PPE claim.
*(Superseded — see "PPE compliance check (manufacturing)" below. The dependency was authorised, the
text tower was built in a separate dev venv, and a binary compliance question shipped after a
five-way garment vocabulary was measured and rejected.)*


## GPU-off fallback fix

### Symptom

With the GPU operator toggle off (`G`, or `--gpu-off`), the person-detecting verticals rendered an
empty tile. The fallback chain was handing `person-detection-retail-0013` to the NPU — the one device
where that model produces nothing.

### Root cause, measured

Two boundaries, measured rather than inferred:

**Policy → assignment.** With `gpu_enabled=False` in `SPREAD`, `device_for_stream(0)` returns `NPU`
for every stage preference, and `build_stage_assignments` produced `person_detector` with
`intended=GPU -> requested=NPU` for `health`, `manufacturing`, `robotics` and `education`.
`federal` reached the NPU by a different route: `perimeter_detector` is excluded from the auxiliary
branch and so inherited the round-robin result, also `NPU`.

**Model → detections.** `person-detection-retail-0013` on 30 uniformly sampled frames of
`media/medical-eldercare.mp4` at confidence ≥ 0.3:

| Device | Total detections | Mean/frame | Labels |
|---|---:|---:|---|
| NPU | **0** | 0.00 | — |
| GPU | 78 | 2.60 | person ×78 |
| CPU | 78 | 2.60 | person ×78 |

So the model compiles on the NPU and reports `EXECUTION_DEVICES=['NPU']`, but emits no boxes. GPU
and CPU are identical. `AvailabilityMatrix.supports()` answers "did it compile and land correctly",
not "does it produce anything", so the availability gate was satisfied by a device where the model is
blind. That is the defect: **device selection was driven by compilability alone.**

### Fix

The fact that a model compiles but emits nothing is measured data, so it is recorded as verified data
in `config/models.json` rather than a hardcoded set in code:

```json
"no_usable_output_on": {
  "NPU": "Compiles and reports EXECUTION_DEVICES=['NPU'], but emits no boxes. Measured 2026-10-01 ..."
}
```

`load_scenario_catalog` reads that into `StageSpec.unusable_devices`, and device selection now
applies one rule everywhere: **a model is never assigned to a device where it was measured to
produce no usable output.** Enforced in three places that could each reintroduce the bug — the
`SPLIT` GPU→NPU fallback, the `SPREAD`/auxiliary GPU→NPU fallback, the round-robin result, plus
`_fallback_for_model` so `AUTO` priority lists drop the blind device too.

### Verification

New tests (`test_gpu_off_never_parks_a_npu_blind_model_on_the_npu`, over `SPREAD` and `SPLIT` × all
five person-detecting verticals) failed on all ten subtests before the fix, plus
`test_npu_blind_fact_is_recorded_in_verified_config` and
`test_catalog_propagates_the_blind_device_to_the_stage`. Two pre-existing tests had encoded the buggy
expectation (`perimeter_detector -> NPU` with the GPU off) and were corrected with the reason
recorded.

Measured after the fix, `--gpu-off`, 25 s per vertical, from the captured HUD:

| Vertical | Placement with GPU off | Detections | Evidence |
|---|---|---:|---|
| `health` | Person: **CPU** · Pose: NPU · Events: CPU | 3 | `person 95%`, `person 31%`; `logs/gpuoff-health.png` |
| `federal` | Perimeter: **CPU** · vehicle_detector: NPU · Plate: NPU · Events: CPU | 12 | `person 86%`, `person 82%`, `person 48%`, `vehicle 38%`, `license plate 34%`; `logs/gpuoff-federal.png` |

`crossroad-1016` and `vehicle-license-plate-detection-barrier-0106` are unlisted and keep their NPU
placements. Cost of the fix: the person detector runs on the CPU when the GPU is off — 15.7 FPS on
`health` and 13.5 FPS on `federal` — but it detects, which is the point of the mode. The HUD still
shows the `GPU OFF · NPU FAILOVER` badge and the `FALLBACK` indicator, so the substitution is visible
to an operator rather than hidden.

Gates after the fix: `87 tests OK` (was 84), selftest `11 PASS / 0 FAIL`, preflight
`101 PASS / 0 WARN / 0 FAIL`, `tools/verify_sources.py` `48 PASS / 0 FAIL`.

### Known limitation, deliberately not changed

`NPU_ONLY` mode still forces every stage to the NPU, including the blind person detector, so that
tile will be empty. That is an explicit operator request to run NPU-only, and overriding it would
make the mode label untrue. It is recorded here rather than silently changed.




## PPE compliance check (manufacturing)

Authorised later, and investigated rather than assumed. `torch`/`transformers` are deliberately
absent from `requirements.txt`, so the text tower was built in a **separate** dev venv
(`torch 2.14.1+cpu`, `transformers 4.57.6` - pinned below 5.0 because the repo build tool is written
against the 4.x `get_text_features` API, which returns a tensor rather than an output object). The
production `.venv` was not modified.

### What was tried and rejected first

A five-way garment vocabulary (`safety_vest`, `hard_hat`, `work_overalls`, `safety_gloves`,
`plain_clothes`) on full-body crops:

| Clip | Truth | Result |
|---|---|---|
| `mfg-warehouse-ppe-1080p.mp4` | vest + hard hat | safety_vest 50%, work_overalls 45%, gloves 5% |
| `mfg-corridor-hardhats-720p.mp4` | overalls + hard hat | work_overalls 95%, hard_hat 5% |
| `store-aisle-detection.mp4` (control) | **no PPE at all** | **work_overalls 65%**, plain_clothes 35% |

The control decides it: shoppers wearing no PPE were read as `work_overalls` 65% of the time, so the
label was an attractor for "a person in a scene", not a clothing read.

Switching to head-and-shoulder crops (top 38% of the box, widened to 1.3x) made the five-way
vocabulary *semantically* correct on the warehouse clip - every read was a label the workers genuinely
had - but it still flipped between `hard_hat` and `safety_vest` **19 times in 60 sampled frames**. For
comparison the retail vocabulary, which does work, changes label 2 times in 20 labelled frames. That
instability is the same defect that got the ImageNet stage removed, so the five-way vocabulary was not
shipped.

### What shipped: a binary compliance question

`ppe_worn` / `no_ppe` on head/shoulder crops, 60 sampled frames per clip:

| Clip | Truth | Head crops | Full-body crops |
|---|---|---|---|
| `mfg-warehouse-ppe-1080p.mp4` | PPE worn | **60/60, 0 changes** | 59/59, 0 changes |
| `mfg-corridor-hardhats-720p.mp4` | PPE worn | **60/60, 0 changes** | 51/57, 8 changes |
| `edu-campus-walking-720p.mp4` | no PPE | **60/60, 0 changes** | 60/60, 0 changes |
| `store-aisle-detection.mp4` | no PPE | 44/49, 10 changes | 49/50, 2 changes |

A binary decision is far more stable than a five-way argmax, and it separates both ways: the
warehouse and corridor workers read `ppe_worn`, the campus students read `no_ppe`. Store shoppers are
the hardest case at 90% - indoor retail lighting, people in coats.

### In the live pipeline

25 s, `tools\review_sessions.py --scenario manufacturing`:

- 4.25 detections/frame, 40 tracks, **0 stream errors**
- **515 classifications, all `ppe_worn`, 0 `no_ppe`** - no false positives on a clip where both
  workers wear hi-vis and hard hats
- **0 classification changes** across every track
- 11 of 40 tracks were named; the rest never sustained `classify_min_frames` consecutive qualifying
  frames, so the stage stays silent rather than guessing
- CLIP stage placed on `EXECUTION_DEVICES=['GPU.0']` with `inference_count: 1` per frame
- Screenshot `logs/ppe-manufacturing.png` shows `person 100% . ppe_worn 33% . #13` on the overlay,
  118.7 det/s at 25.0 FPS

### One real code change was needed, and it was previously a latent bug

`classification_candidates()` hard-banned `person` crops for every classifier, on the reasoning that
an ImageNet label for a person crop is always noise. That reasoning is sound for ImageNet but it also
silently applied to the zero-shot stage, whose output is bounded by an operator-declared vocabulary
instead of 1000 unrelated classes. Consequences:

- `metro` declares `classify_detector_labels: ["vehicle", "person"]` and carries a `person` entry in
  its vocabulary, and **not one of its person crops was ever classified**. Nothing failed; the
  declaration was simply inert.
- The manufacturing PPE check could not run at all until this was fixed.

The ban is now scoped: `allow_person_labels` defaults to off, so the ImageNet path keeps it no matter
what a scenario asks for, and only `_run_zero_shot()` opts in. Both directions are covered by
`test_zero_shot_may_submit_person_but_imagenet_may_not`.

### Guard against the silent-wrong-vocabulary failure

The bake artifacts are gitignored, and `load_zero_shot_vocabulary()` falls back to the **default**
(retail grocery) pair when a named pair is missing. An unbaked manufacturing tile would therefore name
warehouse workers `mtn_dew`. Two checks now prevent that:

- `tools\preflight.py` gained a `phase 3 vocabulary` row per declared stage, failing when the baked
  pair is absent **or stale** relative to `tools/clip_vocabulary.py`, with the rebuild command as
  remediation. Three rows are reported.
- `tests/test_vocabulary.py::test_every_declared_scenario_vocabulary_is_baked_and_current` asserts
  the same invariant for every scenario that declares a vocabulary.

### Gates

| Gate | Result |
|---|---|
| `python -m unittest discover -s tests -q` | `Ran 90 tests ... OK` (exit 0; was 87) |
| `python -m app.main --selftest` | `11 PASS, 0 FAIL` (exit 0); "7 ordered scenarios, 24 stages" |
| `python tools/verify_sources.py` | `48 PASS, 0 FAIL` (exit 0) |
| `python tools/preflight.py` | `107 PASS, 0 WARN, 0 FAIL` (exit 0; was 104 - three vocabulary rows) |

### What this does not claim

It cannot say *which* item of PPE is worn, and it has never observed a violation, because no clip in
the library contains a non-compliant worker. The tile confirms compliance; it does not police it. Both
limits are stated in `README.md` and in the vocabulary's own comment so the claim cannot drift.
## HUD layout - video legibility at booth distance

The picture was too small to read. Three changes, all verified by measuring the
rect the image is actually drawn into rather than by looking at a screenshot.

### What moved

1. **The rotating ticker line is gone.** Its slot in the header now holds the
   measured `PIPELINE PLACEMENT - duration share` bar, which was previously buried
   below the video.
2. **The whole bottom row is hidden at density 1** - the miniature copy of the
   main stream plus the six metric tiles. It returns the moment an operator
   raises the density, because above 1 there are genuinely several streams to
   compare. Verified: `--density 2` brings both stream tiles and the metric grid
   back, and shows per-stream round-robin placement (stream 0 detector on NPU,
   stream 1 on GPU.0) at 1089 det/s.
3. **On-video descriptors are much smaller.** A new `font_detection_label` token
   (13) replaces the 22pt overlay font for detection boxes and zone names, and
   `overlay_label_height` drops 30 -> 18 so rows pack instead of smearing. The
   header badge and event strip use `font_overlay`, reduced 22 -> 18.

### Measured result

`FrameCanvas.video_geometry()` reports the drawn rect, and a 16:9 clip inside this
canvas is height-limited, so reclaiming vertical space is the only thing that
enlarges it:

| | canvas | video | picture area |
|---|---|---|---|
| before | 1101x230 | 409x230 | 94,044 px2 |
| after | 1101x482 | **857x482** | **413,020 px2** |

**4.4x the picture area**, from the same 1101px width. Width was never the
constraint; height was, which is why the bottom row had to go rather than being
narrowed.

Two measurement mistakes are worth recording, because both produced a wrong
number first: judging the size by eye from a screenshot, and then trying to
automate it with a brightness threshold over the left panel - which measured the
bright *placement bar* as footage and reported the change as negative. The
baseline above was produced by adding only the instrumentation to the committed
layout and running it for real.

### Gates

| Gate | Result |
|---|---|
| `python -m unittest discover -s tests -q` | `Ran 94 tests ... OK` (exit 0; +4 new layout tests) |
| `python -m app.main --selftest` | `11 PASS, 0 FAIL` (exit 0) |
| `python tools/verify_sources.py` | `48 PASS, 0 FAIL` (exit 0) |
| `python tools/preflight.py` | `107 PASS, 0 WARN, 0 FAIL` (exit 0) |

`tests/test_layout.py` pins the fitted-size arithmetic, the new font tokens and
the seven-scenario key map, so the picture cannot silently shrink again. It
asserts the pure function rather than instantiating `FrameCanvas`, because
building a QWidget needs a QApplication and no other test in the suite does that.
### Header stat strip replaces the LIVE badge and the clock

The status panel's second row carried a `LIVE` badge and a wall clock. Neither told a booth visitor
anything they could not already see, and they occupied the most valuable corner of the header. They
are replaced by three measured readouts, ordered to answer three questions in sequence:

| Position | Stat | Question it answers | Source |
|---|---|---|---|
| left | `FPS` | Is it live? | stream 0 processing frames/s over the rolling 10 s window |
| centre | `DET/s` | Is it doing real work? | detections/s summed across all active streams |
| right | `EVENTS` | Does it produce a business answer? | business events in the scenario's rolling event window |

`EVENTS` is the one most demos omit. Detection counts alone only show that a model fired; the event
count is the number a retail, transit or safety lead would actually be asked for, and it is counted on
the CPU from tracked detections, which is what makes it a result rather than a raw inference.

All three read the same values the metric tiles use, so they cannot disagree. Verified at
`--density 2`: the header read `FPS 24.7 / DET/s 276.9 / EVENTS 50` while the tile grid read
`REAL-TIME 24.7 FPS / DET/s 276.9 / EVENTS 50` on the same frame. At density 1 the strip is the only
place these three appear, since the tile grid is hidden to give the video its height.

A missing measurement renders as an em dash rather than a zero, because a zero reads as "measured, and
the answer was nothing" when the truth is usually "not measured yet".

Two related notes on what was removed. The `LIVE`/`ATTRACT` badge went because attract mode replaces
the entire page, so the page itself is the indicator. The clock's timer stayed: it is the UI thread's
C-level heartbeat for `hangwatch`, and removing the widget did not remove `mark_ui_tick()` or
`arm_dump_later()` - only the text update went.

Gates after this change: `95 tests OK` (one new typography test),
selftest `11 PASS / 0 FAIL`, preflight `107 PASS / 0 WARN / 0 FAIL`.
# Engine Lab — current state

**Read this first.** It is the current truth about the project. `docs/SPEC.md` is the original
build contract (what was asked for, and why) and `AGENTS.md` is the working rules. Where this file
and `docs/SPEC.md` disagree, **this file wins** — the spec has been amended in places, and those
amendments are listed at the bottom.

Last updated: 2026-09-30, after a full review, five fix commits and a complete acceptance run on
the demo machine.

---

## 1. What this is

A Windows booth demo for Intel Core Ultra laptops (NPU + iGPU + CPU). It plays a video through a
real OpenVINO pipeline and shows which engine is doing the work, with live toggles that disable
the NPU and/or GPU so the audience watches the load move to the CPU. Four verticals: retail,
smart city, medical, government/defense.

The point of the demo is the *platform* story: three engines, one SoC, real measured numbers.

## 2. Demo machine

| | |
|---|---|
| CPU | Intel Core Ultra 9 288V (Lunar Lake, 8P + 8L cores) |
| GPU | Intel Arc 140V, driver 32.0.101.6737 |
| NPU | Intel AI Boost, driver 32.0.100.5540 (OpenVINO 2026.4 wants 5540+) |
| OS | Windows 11 24H2+ |
| Repo | `C:\dev\engine-lab` |

Script execution is blocked on this machine. Never depend on running a `.ps1` or `.bat`: issue the
commands directly. `setup-demo-machine.cmd` exists for the bootstrap.

## 3. Branch and PR state

| PR | Branch | Base | What it is |
|---:|---|---|---|
| #1 | `fix/demo-killers` | `main` | stream survival, honest NPU attribution, two ctypes bugs, Phase 0 bootstrap |
| #2 | `fix/benchmark-evidence` | `main` | the acceptance report must not cite diagnostics a run did not produce |
| #3 | `fix/verticals` | `fix/demo-killers` | footage that matches each vertical + hang capture |

`fix/verticals` contains everything in #1 as well, so it is the branch to pull for a full test.
Commits are authored `Hermes Agent <agent@hermes.local>`.

## 4. Verified on the demo machine

These are real numbers from the machine, not estimates.

- **Preflight: 95 PASS, 0 WARN, 0 FAIL** (95 rows), including 33/33 model × device compiles with
  `EXECUTION_DEVICES` placement assertions.
- **Unit tests: 48 PASS** on the branch the machine tested (`fix/demo-killers`). `+5` hangwatch
  tests on `fix/verticals`, `+5` benchmark-reporting tests on `fix/benchmark-evidence`.
- **NPU gauge works.** It reads real PDH values (16-18% while a scenario runs) and the counter
  finder enumerates 9 NPU/Neural instances and names the device `Intel(R) AI Boost`. Before the
  `_get_registry_string` fix the name came back as one character and the NPU was dropped.
- **The toggle behaves.** Pressing `N` moved CPU 23% → 36% and GPU 46% → 70%, the video kept
  playing (track ids kept incrementing), and no stream died.
- **Full acceptance matrix ran** (300s × 4 scenarios at density 1, 60s × 4 at density 4, 600s
  attract): **8 PASS, 1 FAIL**.
- **No memory leak.** Attract at density 4: RSS 861 MiB at start → 2638 MiB by ~70s → flat.
  From the 2-minute checkpoint to the end (8.5 minutes, 50 scenario switches) growth is
  **+17.9 MiB (0.68%)**. The app settles around **2.6 GB** — know that for the booth laptop.

## 5. Open defects, in priority order

**A. gov_defense at density 4 can hang (~1 in 3).** The app stops responding to Windows messages
and Windows kills it: event log `Application Hang` / `AppHangB1`, no traceback, no diagnostic
file, and the session log stops right after the availability-cache line. It passes standalone
(76s for a 60s run) and it is the only scenario with **two GPU detector stages**. Leading suspect:
the async callback state in `app/engine/runner.py`, where the OpenVINO callback thread writes
instance attributes that the worker thread then reads after `wait_all()` with no synchronisation —
the previous review listed "does `wait_all()` guarantee the callback has completed" as unverified.
**Not fixed.** `app/hangwatch.py` (new) now dumps every thread's stack into the session log if the
UI thread stalls for 30s, so the next occurrence should name its own line. This is a sign-off
blocker: the spec says the app must never crash in front of a customer.

**B. Numbers on screen that can mislead.** All still open:
- the 60s NPU sparkline can blend system-counter samples with app-measured samples in one line;
- the sampler's exception path publishes `process_rss_bytes=0`, rendered as "0 MiB";
- the stage breakdown double-counts preprocess + inference inside the classifier/CLIP stages, so a
  stage's cost can exceed the end-to-end latency shown beside it;
- `inferences/s` divides by one frame's busy time, not wall clock, so it is an upper bound;
- the placement bar colours segments by position, not by the device that ran them;
- the `FALLBACK` chip is suppressed for AUTO placements;
- with both toggles off the header reads `OFF · NPU off · GPU off`, dropping the real mode name.

**C. Presentation.** The engine numeral renders at 40pt/56pt while `--selftest` asserts an unused
72pt theme token (a false pass). At 3440×1440 the app takes its *compact* layout, which caps the
video panel at 430px and splits the row 6:4 instead of ~7:3 — so the video is small and
letterboxed, and at density 1 the tile canvas is hidden, leaving the largest panel showing text.
There is also nothing on screen that proves a classification happened: no strip of detected item
crops with labels. **This is the next piece of work.**

**D. Scenario content.** `app/scenarios/smart_city.py` and `app/scenarios/gov_defense.py` are
imported nowhere — three of the four verticals are the same graph with different labels and
footage. Either give them real per-vertical logic or delete them. Also `_run_classifier` is
unreachable dead code that would `KeyError` if the spec's `classifier` stage were restored.

**E. Verification tooling that verifies less than it appears to.**
- `tools/preflight.py` "phase 3 honesty / delivery" rows PASS from source-text greps.
- the availability row passes whenever the probe enumerated every model × device, regardless of
  how many compiles succeeded.
- the config rows pass vacuously (an empty `models.json` reports PASS).
- `tools/preflight.py` raises a traceback instead of printing a FAIL row when
  `config/telemetry_map.json` is missing.
- no test covers the toggle contract, which the spec explicitly asks for ("assert on the device
  list, not on the gauge").

## 6. Not verified anywhere

- whether `AsyncInferQueue.wait_all()` guarantees the Python callback finished before it returns;
- whether `Core().available_devices` returns `GPU` or `GPU.0` on this machine (exact-match
  membership would make a whole engine disappear from the UI);
- whether the driver-version probe succeeds (it decides whether the availability cache can ever
  invalidate after a driver update);
- plate legibility in the government clip — so that scenario's copy no longer promises plate reads;
- whether the new clips look right in the app.

## 7. How to verify anything

```cmd
.venv\Scripts\python.exe tools\verify_sources.py     # URLs checked by content, not status code
.venv\Scripts\python.exe tools\download_models.py    # models, labels, media
.venv\Scripts\python.exe tools\probe_telemetry.py    # -> config\telemetry_map.json
.venv\Scripts\python.exe tools\preflight.py          # PASS/FAIL table
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tools\benchmark_matrix.py   # full acceptance, ~36 minutes
```

`tools/benchmark_matrix.py --quick` is a smoke test only. Its output must never be presented as
acceptance evidence: a 55-second attract run measures the startup ramp (which is where the old
"159.61% RSS growth" number came from) instead of steady-state behaviour.

## 8. Scenario media

Each scenario points at one clip that was fetched and frame-checked before being committed. The
old clips are kept as named alternates.

| scenario | clip | verified content | licence |
|---|---|---|---|
| retail | Pexels 32658425 | overhead self-checkout; Ruffles bag legible through most of it, Mountain Dew around 13-18s | Pexels (commercial, no attribution) |
| smart_city | Mixkit 1755 | top-down multi-lane intersection with cars | Mixkit (commercial, no attribution) |
| medical | Mixkit 5559 | indoor care scene, caregiver helping an elderly person stand and walk | Mixkit |
| gov_defense | Mixkit 35890 | ground-level vehicle entry point with cars and people | Mixkit |

The retail clip is the same one Intel's own retail loss-prevention reference downloads
(`configs/camera_to_workload.json` → `pexels.com/download/video/32658425`).

**Media filenames changed**, so `tools/download_models.py` must run before the app will start on
a machine that has the old files. Until it does, `tests/test_scenarios.py` fails on the missing
media — that is expected, not a regression.

## 9. Rules that are easy to get wrong

1. **Never zero a gauge for a switched-off engine.** The spec requires the last measured value to
   stay visible but greyed. Zeroing claims the engine is idle, which is a different and false
   statement. What is missing is a freshness cue ("last measured 12s ago"), not a zero.
2. **Static shapes only.** The NPU rejects dynamic shapes.
3. **`AUTO` + device priorities, never `HETERO`**, for splitting work on the NPU.
4. **`EXECUTION_DEVICES` is the only proof of placement.** Never derive a badge from the string
   that was requested.
5. **No fabricated values.** A missing counter renders as "no counter" or an em dash, never 0.
6. **The app runs as a standard user.** Elevation is for installs only.
7. **Never present `--quick` output as acceptance evidence.**

## 10. Amendments to docs/SPEC.md

- **Retail's classifier is CLIP zero-shot** (`clip-vision-patch32`), not the spec's
  `efficientnet-b0-fp16` + `product-detection-0001`. The CLIP stage was spec'd as Phase 4(a)
  ("keep it out of the default path") and is now the default. This is a deliberate product call —
  zero-shot naming from a declared store vocabulary is a better story than ImageNet words — but it
  needs a pinned model hash and a precision check: `tools/convert_clip_onnx.py` defaults to FP32
  while `config/models.json` declares FP16.
- **`config/models.json` was extended** beyond the originally verified set (now 11 models, 9 media
  entries) and the CLIP entry has no downloadable IR, so it is built locally. It is the only entry
  with `source: "converted"`.
- **`tools/download_models.py` treats `source: "converted"` as "build locally"**, not a failure.
- **The spec's Phase-3 durations are the real ones**: 300s at density 1, 60s at density 4, 600s
  attract. Anything shorter is a smoke test.

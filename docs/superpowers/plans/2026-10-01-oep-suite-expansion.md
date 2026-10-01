# Open Edge Platform suite expansion — plan and outcome

Date: 2026-10-01
Status: implemented and verified

## Why

The booth demo presented four verticals under its own naming (retail, smart city, medical /
eldercare, government / defense). Intel's [Open Edge Platform](https://github.com/open-edge-platform)
defines seven AI suites, and four of ours mapped onto them almost 1:1 while three were missing
entirely. Adopting the OEP taxonomy means the demo is described in Intel's own vertical language,
and it makes the gaps explicit rather than leaving the demo looking arbitrarily partial.

The mapping:

| OEP suite | Before | After |
|---|---|---|
| Retail | `retail` | `retail` (unchanged) |
| Metro | `smart_city` | `metro` |
| Manufacturing | — | `manufacturing` (new) |
| Robotics | — | `robotics` (new) |
| Education | — | `education` (new) |
| Health and Life Sciences | `medical` | `health` |
| Federal and Aerospace | `gov_defense` | `federal` |

## Hard constraints carried into the design

1. **Every vertical must exercise NPU, GPU and CPU.** Enforced by
   `tests/test_scenarios.py::test_every_vertical_uses_all_three_engines` and by a preflight row per
   scenario.
2. **`person-detection-retail-0013` returns zero detections on the NPU** on this footage, regardless
   of blob-cache state (measured during the earlier `gov_defense` fix). It is therefore GPU-preferred
   in every scenario that uses it. The NPU hosts `person-vehicle-bike-detection-crossroad-1016` or
   `human-pose-estimation-0001` instead, both of which work there.
3. **Footage must match the detector, and that is checked before wiring, not after.** The earlier
   `smart_city` failure was a clip problem, not a model problem.
4. **No new model downloads.** Every model used was already in `config/models.json`.
5. **No unsourced numbers.** Anything displayed or documented must come from a run whose output is
   recorded.

## Step 1 — rename (commit `120964c`)

`smart_city`→`metro`, `medical`→`health`, `gov_defense`→`federal`, across `config/scenarios.json`,
`app/scenarios/*.py`, `app/engine/pipelines.py`, `app/hud.py`, `app/main.py`, tests, tools, `README.md`
and `bench_report.md`. The `scenario.id == "gov_defense"` branch in `app/engine/pipelines.py` was
re-keyed to `scenario.id == "federal"` and left in place: it encodes a measured hardware fact about
that stage graph, not a statement about government semantics.

Verified behaviour-neutral by per-frame comparison against pre-rename baselines for all four
scenarios. Historical plan/spec documents were not rewritten; they carry a superseded-naming note.

## Step 2 — footage screening, before any code

All 13 cached clips were screened with uniform sampling of 40 frames each, running
`person-detection-retail-0013` @0.3 and `person-vehicle-bike-detection-crossroad-1016` @0.2 on GPU.
Full table in `bench_report.md`.

**Method note that matters:** the first screen read only the *first* 40 frames of each clip and
reported `worker-zone-detection.mp4` and `one-by-one-person-detection.mp4` as detecting nothing.
Re-sampling uniformly across each clip showed both detect at confidence 1.00. Sampling from frame 0
is not a valid screen for looping clips. Anyone repeating this must sample uniformly.

**Exclusion:** `worker-zone-detection.mp4` is present in `media/` but has no
`models/download_manifest.json` entry and no known source URL. It is the best unused
manufacturing-style clip (0.9 person/frame, conf 1.00) but wiring it in would mean the demo depends
on a file a fresh setup cannot fetch, and recording a source URL would mean inventing one. Excluded.
The right fix is to source a real URL and record it, not to hardcode a path.

## Step 3 — the three new verticals

| Vertical | `video_id` / file | Stages |
|---|---|---|
| `manufacturing` | `store_aisle_alt` / `store-aisle-detection.mp4` | crossroad-1016 NPU → person-detection GPU → zone-occupancy CPU |
| `robotics` | `queue_alt` / `one-by-one-person-detection.mp4` | crossroad-1016 NPU → person-detection GPU → approach-zone CPU |
| `education` | `walking_people` / `face-demographics-walking.mp4` | crossroad-1016 NPU → person-detection GPU → pose NPU → attendance CPU |

The HUD grew from keys 1–4 to 1–7. Note the key mapping changed meaning: key 3 was `health` and is
now `manufacturing`; `health` moved to 6 and `federal` to 7.

## Rejected: CLIP PPE stage for manufacturing

A zero-shot PPE vocabulary (`hard hat`, `safety vest`, `gloves`) was the intended differentiator for
manufacturing, since naming PPE from a declared vocabulary is a genuine strength of this pipeline.
It was measured before adoption and rejected: 52/96 crops named `safety vest`, 27/96 `hard hat`, mean
top-1 0.238, on footage of ordinary shoppers in a store aisle. Shipping it would have put a false
claim on a booth screen. `manufacturing` therefore ships with no classifier, and a test asserts the
absence of the CLIP stage so it cannot be reintroduced silently.

## Verification actually performed

| Gate | Result |
|---|---|
| `unittest discover -s tests -q` | `Ran 83 tests ... OK`, exit 0 (was 79) |
| `python -m app.main --selftest` | `11 PASS, 0 FAIL`, exit 0, "7 ordered scenarios, 23 stages" |
| `python tools/preflight.py` | `101 PASS, 0 WARN, 0 FAIL`, exit 0 (was 98) |
| `review_sessions.py` × 3 new verticals | 0 stream errors; densities and track counts in `bench_report.md` |
| `EXECUTION_DEVICES` per stage | detector `['NPU']`, person_detector `['GPU.0']`, education pose `['NPU']` |
| Screenshots | `logs/b-manufacturing.png`, `logs/b-robotics.png`, `logs/b-education.png`, each opened and inspected |

## Outcome, stated honestly

The three new suites work and are correctly placed across NPU/GPU/CPU, but they are the weakest part
of the demo. Manufacturing (4.41 det/frame, 27 tracks, 387 det/s at 55.5 FPS) is genuinely
presentable. Robotics (1.65 det/frame) and education (0.88 det/frame) are sparse, and robotics has
quiet stretches — the captured booth screenshot caught a frame with zero detections while a
`person_counted` event from earlier in the run was still on screen.

The cause is footage, and the fix is better footage, not a different model. The candidate clips for
these three suites were chosen from what happened to be cached; a purpose-shot factory floor, a
robot cell and a classroom would each be denser and more on-message. That is the highest-value
follow-up if these three tiles are going to carry a booth slot.

## Follow-ups not done

- Source and manifest a replacement for `worker-zone-detection.mp4`, then reconsider manufacturing
  footage (it is thematically a better fit than a store aisle).
- GPU-off mode still detects no people: `person-detection-retail-0013` round-robins back to the NPU
  via `device_for_stream`, which is the NPU combination measured to return zero detections. This
  predates this work and is untouched by it.

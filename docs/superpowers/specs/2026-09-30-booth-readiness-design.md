# Booth-readiness rework — design

> **Superseded naming:** This design was written under the original scenario ids. The four verticals have since been renamed to follow the Open Edge Platform suite taxonomy: `smart_city` → `metro`, `medical` → `health`, `gov_defense` → `federal`. The `retail` scenario is unchanged. The analysis and decisions described here were actually made under the old names; this note is added for traceability and the design's history is intentionally left as written.

Date: 2026-09-30
Status: design approved in conversation; awaiting spec review
Branch: `fix/verticals` (builds on `fix/demo-killers`)

## Purpose

Make Engine Lab read as a sales demo on its actual target hardware — a 16:9 1080p
display — for all four verticals. Four problems were reported from the demo machine:

1. **Layout is wrong for 16:9.** The app is being driven on a 2293×960 ultrawide, but the
   show runs at 1080p. The full layout never appears on the target screen.
2. **Not every vertical shows NPU + GPU + CPU.** Two verticals only exercise two engines.
3. **The workloads are not legible.** The HUD shows model identifiers
   (`YOLO11n`, `clip-vision-patch32`) rather than what each stage is doing.
4. **Classification is poor.** CLIP is naming groceries with a kitchen vocabulary.

Success: on a 1920×1080 screen, each of the four verticals visibly exercises NPU, GPU and
CPU with a legible task description per stage, and the retail classification names things
that are actually in the frame.

## Root causes found

| # | Root cause | Evidence |
|---|---|---|
| 1 | `hud.py:1086` — `self.compact_layout = available_height < 1600`. A 1080p screen (height 1080) trips the threshold, so the full layout only runs above 1600px. | code read |
| 2 | `smart_city` has no GPU stage; `gov_defense` has no NPU stage. | `config/scenarios.json` |
| 3 | The tile strip renders model ids and the generic words `detect` / `process`. | screenshots |
| 4 | The baked CLIP vocabulary is dishware (`pot`, `mixing bowl`, `storage container`) while the retail clip is an overhead self-checkout of groceries. The vocabulary is global, so it cannot differ per vertical. | `tools/clip_vocabulary.py`, contact sheet of `retail-checkout.mp4` |

## Workstreams

Ordered as E → B → A/D → C so that visible wins land early.

### E. 16:9 layout

- Replace the height threshold with a decision keyed to the demo target. The rule is
  concrete and testable: **full layout when the window is at least 1600×900; compact
  layout below that.** A 1080p screen (1920×1080), a 4K screen and the operator's
  2293×960 ultrawide all take the full layout; only a genuinely small window goes compact.
  The decision moves into a small pure function so it can be unit-tested.
- The video panel is constrained to 16:9 and fills its column, instead of being capped at
  430px in the compact layout.
- Acceptance: render at 1920×1080 and screenshot all four verticals; the video is the
  largest panel and is not letterboxed into a small tile.

### B. Stage graphs — every vertical uses all three engines

| vertical | NPU | GPU | CPU |
|---|---|---|---|
| retail | YOLO detect | CLIP classify | events |
| smart_city | traffic detect | CLIP vehicle typing (new) | zone count |
| medical | pose | person detect | posture events |
| gov_defense | person detect (moved from GPU) | vehicle detect (new) + plate detect | track events |

- **smart_city** gains a GPU CLIP classification stage. This is real work and is what
  `docs/SPEC.md` §4 specified all along (`detect → classify → count per lane zone`).
- **gov_defense** moves `perimeter_detector` from GPU to NPU and gains a continuous GPU
  vehicle detector. Moving rather than duplicating, because the vertical already has a
  person detector; a second would be redundant. The vehicle detector keeps the GPU busy on
  frames where no plate is legible, which was the reason gov_defense's GPU looked idle.
- Acceptance: a unit test asserts every scenario declares at least one stage on each of
  NPU, GPU and CPU.

### A. Per-scenario CLIP vocabularies

- `tools/build_clip_zero_shot.py` gains `--scenario`, baking
  `text_embeddings_<scenario>.npy` and `vocabulary_<scenario>.json` into the CLIP model
  directory (one pair per vertical that uses CLIP).
- `config/scenarios.json` names its vocabulary via a new `vocabulary` field on the
  classifier stage. A stage without one falls back to the existing default file, so
  nothing existing breaks if a vocabulary is missing.
- `tools/clip_vocabulary.py` holds one declared vocabulary per vertical.
- The retail vocabulary is rebuilt to match the checkout footage: bananas, apple,
  packaged snack, bottle, canned good, and similar grocery categories. This was verified by
  hand — CLIP reads this footage correctly and called the loaf `bread`, the bunch
  `bananas` and the tins `jar` at 40–49% confidence.
- The smart_city vocabulary is vehicle/person categories so its new GPU stage has something
  meaningful to say.
- Acceptance: retail on the checkout clip names grocery items, not kitchenware.

### D. Workload descriptions

- `config/models.json` gains a human-readable `task` per model: *Object detection*,
  *Zero-shot classification*, *Pose estimation*, *License plate detection*,
  *Vehicle + pedestrian detection*.
- The tile strip and the pipeline placement bar render `model — task → device` instead of
  `YOLO11n: detect · clip-vision-patch32: process`.
- A compact per-stage legend lists each stage's task, model and device, so a passer-by can
  read what each engine is doing.
- Acceptance: a screenshot shows a plain-English task next to every stage.

### C. Honest idle states

`EngineGauge` currently has three states. It gains a fourth, and the distinction matters
for a sales floor:

| state | when | renders |
|---|---|---|
| `ACTIVE` | measured load now | live percentage |
| `IDLE` | this vertical uses the engine, momentarily quiet | `IDLE · last measured Ns ago` |
| `NOT USED BY THIS VERTICAL` | no stage on that engine | label + tooltip |
| `OFF BY OPERATOR` | toggled off | greyed last value, never zero (existing rule) |

This fixes a real defect from the screenshots: retail's GPU read `0% MEASURED IDLE` between
detections, which reads as a broken demo rather than a quiet engine.

## Out of scope

- The government/defense title wording (raised, then withdrawn by the operator).
- The pre-existing hang fixes already in the working tree (`app/hangwatch.py`,
  `app/engine/runner.py`, `app/hud.py`, `tests/test_hangwatch.py`) — those are a separate
  change set.
- Anything requiring new model downloads beyond what `config/models.json` already lists.
- Speaker/attract-mode polish beyond what layout work requires.

## Testing

- Unit: per-scenario vocabulary resolution; **every vertical declares NPU + GPU + CPU**;
  the layout-decision function at 1920×1080, 3440×1440 and 3840×2160.
- Manual: screenshots of all four verticals at 1920×1080, reviewed for layout, engine
  activity and task legibility.
- Existing gates must stay green: `unittest`, `--selftest`, `tools/preflight.py`,
  `tools/verify_sources.py`.

## Risks

- Rebuilding each CLIP vocabulary needs the dev venv with `torch`/`transformers`, which is
  deliberately outside production `requirements.txt`. One build per vertical.
- Adding stages costs frames. gov_defense at density 4 was already the scenario that
  exposed the hang; its frame budget must be re-measured after the vehicle detector is
  added, not assumed.
- A per-scenario vocabulary is a second thing to keep in sync with the footage. If the
  footage changes, the vocabulary must be rebuilt; the README should say so.

# Booth-Readiness Implementation Plan

> **Superseded naming:** This plan was written under the original scenario ids. The four verticals have since been renamed to follow the Open Edge Platform suite taxonomy: `smart_city` → `metro`, `medical` → `health`, `gov_defense` → `federal`. The `retail` scenario is unchanged. The work described here was actually done under the old names; this note is added for traceability and the plan's history is intentionally left as written.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Engine Lab read as a sales demo on a 16:9 1080p display, with every vertical visibly exercising NPU + GPU + CPU and a legible task label per stage.

**Architecture:** Four independent workstreams against the existing Qt HUD and OpenVINO pipeline. A pure layout-decision function replaces an inline height threshold; scenario stage graphs are extended so each vertical has a stage on all three engines; CLIP text embeddings become per-scenario so each vertical's vocabulary matches its footage; the HUD renders a human task label instead of a model id.

**Tech Stack:** Python 3.12, PySide6 6.11.2, OpenVINO 2026.4.0, numpy, opencv.

**Spec:** `docs/superpowers/specs/2026-09-30-booth-readiness-design.md`

## Global Constraints

- Static shapes only; the NPU rejects dynamic shapes.
- `AUTO` + `ov::device::priorities`, never `HETERO`.
- `EXECUTION_DEVICES` is the only proof of placement.
- No fabricated values. A missing counter renders as "no counter" or an em dash, never 0.
- Never zero a gauge for an engine that is switched off.
- Peak specs are labelled `peak` and carry a source.
- Do not change pinned dependency versions.
- Work on branch `fix/verticals`; commit after each task.

## Review Focus

- **A window smaller than 1600×900** — must take the compact layout and not clip; a 1280×720 laptop is a plausible booth backup.
- **A vertical with no stage on an engine** — the gauge must say `NOT USED BY THIS VERTICAL`, never `0%`, which reads as broken.
- **A missing per-scenario vocabulary file** — must fall back to the default vocabulary rather than crashing at startup.
- **CLIP naming something not in the footage** — retail must name groceries, not kitchenware; a wrong-but-confident label is worse than no label.
- **The added gov_defense vehicle detector at density 4** — this is the scenario that exposed the hang; frame budget must be measured, not assumed.

---

### Task 1: Layout decision function

**Files:**
- Create: `app/layout.py`
- Modify: `app/hud.py` (the `compact_layout` assignment near line 1086)
- Test: `tests/test_layout.py`

**Interfaces:**
- Produces: `use_compact_layout(width: int, height: int) -> bool` — True only when the window is smaller than 1600×900 in either dimension.

- [ ] **Step 1: Write the failing test**

```python
def test_1080p_uses_full_layout():
    assert use_compact_layout(1920, 1080) is False

def test_ultrawide_uses_full_layout():
    assert use_compact_layout(2293, 960) is False

def test_4k_uses_full_layout():
    assert use_compact_layout(3840, 2160) is False

def test_small_window_goes_compact():
    assert use_compact_layout(1280, 720) is True

def test_exactly_1600x900_is_full():
    assert use_compact_layout(1600, 900) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_layout -v`
Expected: FAIL, `ModuleNotFoundError: app.layout`

- [ ] **Step 3: Implement in `app/layout.py`**

```python
def use_compact_layout(width: int, height: int) -> bool:
    """True only when the window is smaller than the full layout's minimum."""
```

Rule: compact iff `width < 1600 or height < 900`. Constants `FULL_LAYOUT_MIN_WIDTH = 1600`, `FULL_LAYOUT_MIN_HEIGHT = 900`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_layout -v` → PASS

- [ ] **Step 5: Use it in `app/hud.py`**

Replace `self.compact_layout = available_height < 1600` with a call passing the screen's available width and height.

- [ ] **Step 6: Commit** — `feat: choose the layout for the demo's 16:9 target`

---

### Task 2: Every vertical uses NPU, GPU and CPU

**Files:**
- Modify: `config/scenarios.json`
- Modify: `app/scenarios/smart_city.py` (zone vocab if needed)
- Test: `tests/test_scenarios.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: every scenario in `catalog` has ≥1 stage on each of `NPU`, `GPU`, `CPU`.

- [ ] **Step 1: Write the failing test**

```python
def test_every_vertical_uses_all_three_engines(self):
    for scenario in self.catalog.values():
        devices = {s.device_pref for s in scenario.stages}
        self.assertTrue({"NPU", "GPU", "CPU"} <= devices, f"{scenario.id}: {devices}")
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL naming `smart_city` and `gov_defense`.

- [ ] **Step 3: Extend the stage graphs**

- `smart_city`: add a `product_classifier` stage, `model_id: clip-vision-patch32`, `device_pref: GPU`, plus `classify_detector_labels: ["vehicle"]` and `zero_shot_min`.
- `gov_defense`: change `perimeter_detector` `device_pref` GPU → NPU. Add a `vehicle_detector` stage, `model_id: person-vehicle-bike-detection-crossroad-1016`, `device_pref: GPU`.

- [ ] **Step 4: Run test to verify it passes** → PASS, and confirm no other `test_scenarios` case regresses.

- [ ] **Step 5: Measure gov_defense at density 4** (the added stage costs frames)

Run: `.venv\Scripts\python.exe -m app.main --scenario gov_defense --density 4 --mode spread --exit-after 60 --diagnostic-output logs\task2-gov.json`
Expected: exit 0, and record e2e p50. If p50 rises above ~35 ms, note it rather than hiding it.

- [ ] **Step 6: Commit** — `feat: give smart_city a GPU stage and gov_defense an NPU stage`

---

### Task 3: Per-scenario CLIP vocabularies

**Files:**
- Modify: `tools/clip_vocabulary.py`
- Modify: `tools/build_clip_zero_shot.py`
- Modify: `app/engine/pipelines.py` (`_zero_shot_vocabulary`)
- Modify: `app/engine/stages.py` (`load_zero_shot_vocabulary`)
- Modify: `config/scenarios.json` (vocabulary field)
- Test: `tests/test_vocabulary.py`

**Interfaces:**
- Produces: `VOCABULARIES: dict[str, dict[str, tuple[str, ...]]]` keyed by scenario id.
- Produces: `load_zero_shot_vocabulary(path: Path, name: str | None = None)` — falls back to the default file when `name` is None or its files are absent.

- [ ] **Step 1: Write the failing tests**

```python
def test_retail_vocabulary_names_groceries(self):
    labels = set(VOCABULARIES["retail"])
    self.assertIn("bananas", labels)
    self.assertNotIn("mixing bowl", labels)

def test_missing_vocabulary_falls_back(self):
    embeddings, labels, counts = load_zero_shot_vocabulary(path, name="does-not-exist")
    self.assertTrue(labels)   # default used, no exception
```

- [ ] **Step 2: Run tests to verify they fail.**

- [ ] **Step 3: Implement**

- `tools/clip_vocabulary.py`: `VOCABULARIES` keyed by scenario id; retail becomes groceries matching the checkout clip (bananas, apple, packaged snack, bottle, canned good, carton, bread, bag of chips); smart_city becomes vehicle/person categories. Keep `RETAIL_VOCABULARY` as an alias for retails's entry so existing imports keep working.
- `tools/build_clip_zero_shot.py`: `--scenario NAME` writes `text_embeddings_<name>.npy` and `vocabulary_<name>.json`.
- `_zero_shot_vocabulary` / `load_zero_shot_vocabulary`: take the stage's `vocabulary` name and fall back to the default.

- [ ] **Step 4: Rebuild the embeddings**

Run with the dev venv, once per vertical that uses CLIP:
```
<dev-venv>\Scripts\python.exe tools\build_clip_zero_shot.py --scenario retail
<dev-venv>\Scripts\python.exe tools\build_clip_zero_shot.py --scenario smart_city
```

- [ ] **Step 5: Verify retail on the real footage**

Run: `.venv\Scripts\python.exe tools\review_sessions.py --scenario retail --seconds 40 --output logs\review-task3`
Expected: surfaced labels are grocery categories; no `pot` / `mixing bowl` / `storage container`.

- [ ] **Step 6: Commit** — `feat: give each vertical its own CLIP vocabulary`

---

### Task 4: Human task labels

**Files:**
- Modify: `config/models.json` (add `task_label` per model)
- Modify: `app/hud.py` (tile strip and pipeline bar)
- Test: `tests/test_scenarios.py`

**Interfaces:**
- Produces: every model in `config/models.json` has a non-empty `task_label`.

- [ ] **Step 1: Write the failing test**

```python
def test_every_model_has_a_task_label(self):
    for model in models:
        self.assertTrue(model.get("task_label"), model["id"])
```

- [ ] **Step 2: Run test to verify it fails.**

- [ ] **Step 3: Add `task_label` to every model**

Values: `yolo11n-*` → "Object detection"; `clip-vision-patch32` → "Zero-shot classification"; `efficientnet-*` → "Image classification"; `person-detection-retail-0013` → "Person detection"; `person-vehicle-bike-detection-crossroad-1016` → "Vehicle + pedestrian detection"; `vehicle-license-plate-detection-barrier-0106` → "License plate detection"; `human-pose-estimation-0001` → "Pose estimation"; `age-gender-recognition-retail-0013` → "Age + gender estimation"; `product-detection-0001` → "Product detection".

- [ ] **Step 4: Render `model — task → device`** in the stream tile strip and the pipeline placement bar, replacing the current `model: detect · model: process` text.

- [ ] **Step 5: Run test to verify it passes.**

- [ ] **Step 6: Screenshot and read it back** to confirm a task label is visible next to every stage.

- [ ] **Step 7: Commit** — `feat: label each stage with what it is actually doing`

---

### Task 5: Honest idle states

**Files:**
- Modify: `app/hud.py` (`EngineGauge`, `_on_telemetry`)
- Test: `tests/test_gauge_state.py`

**Interfaces:**
- Produces: `gauge_state(engine: str, scenario_devices: set[str], disabled: bool, value: float | None, age_s: float | None) -> str` returning one of `ACTIVE`, `IDLE`, `NOT USED BY THIS VERTICAL`, `OFF BY OPERATOR`.

- [ ] **Step 1: Write the failing tests**

```python
def test_engine_with_no_stage_is_not_used(self):
    self.assertEqual(gauge_state("GPU", {"NPU", "CPU"}, False, None, None), "NOT USED BY THIS VERTICAL")

def test_used_but_quiet_engine_is_idle(self):
    self.assertEqual(gauge_state("GPU", {"NPU", "GPU"}, False, 0.0, 12.0), "IDLE")

def test_live_engine_is_active(self):
    self.assertEqual(gauge_state("NPU", {"NPU"}, False, 18.0, 0.2), "ACTIVE")

def test_operator_off_beats_everything(self):
    self.assertEqual(gauge_state("NPU", {"NPU"}, True, 18.0, 0.2), "OFF BY OPERATOR")
```

- [ ] **Step 2: Run tests to verify they fail.**

- [ ] **Step 3: Implement `gauge_state`** with the precedence: operator-off, then not-used, then active (recent and non-zero), else idle.

- [ ] **Step 4: Run tests to verify they pass.**

- [ ] **Step 5: Render the state in `EngineGauge`** — `IDLE` shows `IDLE · last measured Ns ago`; `NOT USED BY THIS VERTICAL` shows that text with a tooltip; `OFF BY OPERATOR` keeps the existing greyed last value and is never zeroed.

- [ ] **Step 6: Commit** — `feat: distinguish a quiet engine from an unused one`

---

### Task 6: Final verification

- [ ] **Step 1:** `.venv\Scripts\python.exe -m unittest discover -s tests -q` → all pass
- [ ] **Step 2:** `.venv\Scripts\python.exe -m app.main --selftest` → 11 PASS, 0 FAIL
- [ ] **Step 3:** `.venv\Scripts\python.exe tools\preflight.py` → 0 FAIL
- [ ] **Step 4:** `.venv\Scripts\python.exe tools\verify_sources.py` → 0 FAIL
- [ ] **Step 5:** Screenshot all four verticals at 1920×1080 and confirm: full layout, a task label per stage, and no engine reading a bare `0%`.
- [ ] **Step 6:** gov_defense density 4 × 6 runs → no hang
- [ ] **Step 7: Commit** — `chore: booth-readiness verification`

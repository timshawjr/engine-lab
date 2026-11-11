"""Reproduce the reported sequence: N, then G, then both back on.

qa_sync.py toggles NPU and GPU in the same instant, which is not what an operator
does at a booth. The reported symptom is that pressing N, then G, then turning
them back on leaves the wireframe overlay out of step with the video -- on health,
and possibly on every vertical.

The overlay is drawn from the worker's detections over the display clock's current
frame, so "out of sync" means the two captures are at different file positions.
This records, per phase, the drift (clock position minus worker position) and how
many frames the workers have resynced so far, so a phase with drift but no change
in seeks is a correction that never fired.

Usage: python tools/qa_toggle_sequence.py [scenario]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.engine.pipelines import ScenarioModelRegistry, load_scenario_catalog
from app.hud import MainWindow
from app.main import (
    REQUIRED_CONFIG,
    REQUIRED_PROFILES,
    REQUIRED_RAG_CONFIG,
    REQUIRED_SCENARIOS,
    _run_availability_gate,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "health"

app = QApplication(sys.argv)

clock = {"position": -1}
_orig = MainWindow._on_video_frame


def spy(self, image, frame_index):  # noqa: ANN001
    clock["position"] = getattr(self.video_clock, "last_position", -1)
    return _orig(self, image, frame_index)


MainWindow._on_video_frame = spy

window = MainWindow(
    catalog=load_scenario_catalog(REQUIRED_SCENARIOS, REQUIRED_CONFIG, ROOT / "media"),
    registry=ScenarioModelRegistry(REQUIRED_CONFIG, ROOT / "models", ROOT / "cache"),
    initial_scenario=SCENARIO,
    camera_index=None,
    cache_dir=ROOT / "cache",
    telemetry_map_path=REQUIRED_PROFILES,
    profiles_path=REQUIRED_PROFILES,
    availability=_run_availability_gate(force=False),
    prewarm_fallbacks=[],
    fullscreen=False,
    initial_density=1,
    rag_config_path=REQUIRED_RAG_CONFIG,
)
window.start()

phase = {"name": "0 baseline"}
samples: dict[str, list[int]] = {}
seeks: dict[str, tuple[int, int]] = {}   # phase -> (first, last) resynced_frames total
last_clock = {"value": -1}
notes: list[str] = []


def total_seeks() -> int:
    return sum(getattr(w, "resynced_frames", 0) for w in window.stream_workers.values())


def set_phase(name: str) -> None:
    phase["name"] = name
    samples.setdefault(name, [])
    seeks.setdefault(name, (total_seeks(), total_seeks()))
    notes.append(f"{time.strftime('%H:%M:%S')}  {name}")


def sample() -> None:
    name = phase["name"]
    samples.setdefault(name, [])
    seeks.setdefault(name, (total_seeks(), total_seeks()))
    first, _ = seeks[name]
    seeks[name] = (first, total_seeks())

    position = clock["position"]
    if position < 0:
        return
    # A lap change: the clock jumped backwards by more than a few frames.
    if last_clock["value"] >= 0 and position < last_clock["value"] - 5:
        last_clock["value"] = position
        return
    last_clock["value"] = position

    for worker in window.stream_workers.values():
        worker_position = getattr(worker, "last_position", -1)
        if worker_position >= 0:
            drift = position - worker_position
            if -5 <= drift <= 400:
                samples[name].append(drift)


timer = QTimer()
timer.setInterval(100)
timer.timeout.connect(sample)
timer.start()


def report_overall() -> None:
    print()
    print("  phase                n     drift p50   p95   max    seeks fired")
    print("  " + "-" * 70)
    for name in list(dict.fromkeys(samples)):
        rows = sorted(samples.get(name, []))
        first, last = seeks.get(name, (0, 0))
        if not rows:
            print(f"  {name:20s}   0            (none)         seeks +{last - first}")
            continue
        n = len(rows)
        print(
            f"  {name:20s} {n:4d}      {rows[n // 2]:4d}  {rows[int(n * 0.95)]:4d}  "
            f"{rows[-1]:4d}    +{last - first}"
        )
    print()
    print("  operator timeline:")
    for line in notes:
        print(f"    {line}")


def finalise() -> None:
    timer.stop()
    report_overall()
    window.shutdown()
    app.quit()


# The operator's sequence, with pauses long enough to press the keys by hand.
QTimer.singleShot(200, lambda: set_phase("0 baseline"))
QTimer.singleShot(12000, lambda: set_phase("1 wait, before any key"))
QTimer.singleShot(14000, lambda: (window.toggle_npu(), set_phase("2 NPU off")))
QTimer.singleShot(18000, lambda: (window.toggle_gpu(), set_phase("3 NPU+GPU off")))
QTimer.singleShot(28000, lambda: (window.toggle_npu(), set_phase("4 NPU back on")))
QTimer.singleShot(32000, lambda: (window.toggle_gpu(), set_phase("5 GPU back on")))
QTimer.singleShot(34000, lambda: set_phase("6 after restore 0-8s"))
QTimer.singleShot(42000, lambda: set_phase("7 after restore 8-16s"))
QTimer.singleShot(50000, lambda: set_phase("8 after restore 16-30s"))
QTimer.singleShot(64000, finalise)

QTimer.singleShot(120000, app.quit)
app.exec()

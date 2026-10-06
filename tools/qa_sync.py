"""Measure detection alignment across a device toggle, without being fooled.

Two traps this avoids, both of which produced wrong answers earlier:

1. Comparing the clock's CUMULATIVE frame counter to anything: it climbs past
   the file length because the clips loop.
2. Comparing raw file positions across two captures that loop INDEPENDENTLY:
   when one wraps and the other has not, the difference is meaningless and
   swings by a whole file length. Measured that way, a healthy run reported
   drift anywhere from 20 to 900 frames.

So this samples both file positions and only counts a drift reading when the two
captures are in the same lap. A lap change is detected by a large negative jump
in either position (the wrap), and the comparison restarts from there.

Usage: python tools/qa_sync.py [scenario] [--toggle]
"""
from __future__ import annotations

import statistics
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
    REQUIRED_SCENARIOS,
    _run_availability_gate,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "metro"
WANT_TOGGLE = "--toggle" in sys.argv

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
)
window.start()

last_clock = {"value": -1}
samples: list[tuple[str, int]] = []
phase = {"name": "1 normal"}


def sample() -> None:
    position = clock["position"]
    if position < 0:
        return
    # A lap change: the clock position jumped backwards by more than a few frames.
    if last_clock["value"] >= 0 and position < last_clock["value"] - 5:
        # Restart the comparison from this lap; do not record a bogus drift.
        last_clock["value"] = position
        return
    last_clock["value"] = position
    for worker in window.stream_workers.values():
        if getattr(worker, "last_position", -1) >= 0:
            drift = position - worker.last_position
            # Only meaningful within one lap. A drift beyond a couple of seconds
            # means the two are in different laps; skip rather than report it.
            if -5 <= drift <= 240:
                samples.append((phase["name"], drift))


timer = QTimer()
timer.setInterval(100)
timer.timeout.connect(sample)
timer.start()


def report(name: str) -> None:
    rows = [d for p, d in samples if p == name]
    if not rows:
        print(f"  {name:22s} (no samples)")
        return
    ordered = sorted(rows)
    n = len(ordered)
    seeks = sum(w.resynced_frames for w in window.stream_workers.values())
    print(
        f"  {name:22s} n={n:4d}  drift p50={ordered[n // 2]:3d}  "
        f"p95={ordered[int(n * 0.95)]:3d}  max={ordered[-1]:3d} frames"
        f"   seeks={seeks}"
    )


def phase_to(name: str) -> None:
    phase["name"] = name


def finalise() -> None:
    timer.stop()
    for label in ("1 normal", "2 on CPU", "3 recovered", "4 settled", "2 steady", "3 steady"):
        report(label)
    window.shutdown()
    app.quit()


if WANT_TOGGLE:
    QTimer.singleShot(10000, lambda: report("1 normal"))
    QTimer.singleShot(10050, lambda: phase_to("2 on CPU"))
    QTimer.singleShot(11000, lambda: (window.toggle_npu(), window.toggle_gpu()))
    QTimer.singleShot(30000, lambda: report("2 on CPU"))
    QTimer.singleShot(30050, lambda: phase_to("3 recovered"))
    QTimer.singleShot(31000, lambda: (window.toggle_npu(), window.toggle_gpu()))
    QTimer.singleShot(39000, lambda: report("3 recovered"))
    QTimer.singleShot(39050, lambda: phase_to("4 settled"))
    QTimer.singleShot(55000, finalise)
else:
    QTimer.singleShot(10000, lambda: report("1 normal"))
    QTimer.singleShot(10050, lambda: phase_to("2 steady"))
    QTimer.singleShot(48000, finalise)


QTimer.singleShot(90000, app.quit)
app.exec()
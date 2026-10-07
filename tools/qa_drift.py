"""Does alignment drift accumulate over a long run?

The one-shot resync fires at startup and on a device change. If the stream
worker is even marginally slower than the display clock, the offset grows
unbounded between those events.

Samples drift every 5 s for the requested duration, resetting the baseline
whenever either capture wraps (the clips loop, so a raw position difference is
meaningless across a wrap).

Usage: python tools/qa_drift.py [minutes] [scenario]
"""
from __future__ import annotations

import sys
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
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
SCENARIO = sys.argv[2] if len(sys.argv) > 2 else "metro"

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

state = {"last_clock": -1, "elapsed": 0}


def sample() -> None:
    position = clock["position"]
    if position < 0:
        return
    if state["last_clock"] >= 0 and position < state["last_clock"] - 5:
        state["last_clock"] = position  # lap change; skip this reading
        return
    state["last_clock"] = position
    drifts = [
        position - w.last_position
        for w in window.stream_workers.values()
        if getattr(w, "last_position", -1) >= 0
    ]
    if not drifts:
        return
    drift = max(drifts)
    if not (-5 <= drift <= 240):
        return
    seeks = sum(w.resynced_frames for w in window.stream_workers.values())
    print(f"  t={state['elapsed']:4d}s   drift={drift:4d}f   seeks={seeks}", flush=True)


timer = QTimer()
timer.setInterval(5000)
timer.timeout.connect(sample)
timer.start()


def tick_clock() -> None:
    state["elapsed"] += 5


clock_timer = QTimer()
clock_timer.setInterval(5000)
clock_timer.timeout.connect(tick_clock)
clock_timer.start()


def stop() -> None:
    timer.stop()
    clock_timer.stop()
    window.shutdown()
    app.quit()


QTimer.singleShot(int(MINUTES * 60_000), stop)
QTimer.singleShot(int(MINUTES * 60_000) + 30_000, app.quit)
app.exec()
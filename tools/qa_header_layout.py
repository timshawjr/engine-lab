"""Measure header jitter: does the stat strip reflow as the numbers change?

The header holds three live readouts (FPS, DET/s, EVENTS). If their boxes are
sized to their text, the whole strip -- and the panels beside it -- move every
time a digit changes, which at 5 Hz reads as a bounce.

Samples the geometry of each stat and of the neighbouring title panel for a
while and reports how much any of it moved.

Usage: python tools/qa_header_layout.py [seconds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.engine.pipelines import ScenarioModelRegistry, load_scenario_catalog
from app.hud import HeaderStat, MainWindow
from app.main import (
    REQUIRED_CONFIG,
    REQUIRED_PROFILES,
    REQUIRED_SCENARIOS,
    _run_availability_gate,
)

ROOT = Path(__file__).resolve().parents[1]
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 25

app = QApplication(sys.argv)

window = MainWindow(
    catalog=load_scenario_catalog(REQUIRED_SCENARIOS, REQUIRED_CONFIG, ROOT / "media"),
    registry=ScenarioModelRegistry(REQUIRED_CONFIG, ROOT / "models", ROOT / "cache"),
    initial_scenario="metro",
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

stats: list[HeaderStat] = window.findChildren(HeaderStat)
print(f"  found {len(stats)} header stats", flush=True)

samples: dict[str, list[tuple[int, int]]] = {}


def sample() -> None:
    for widget in stats:
        key = widget.value.text().split()[0] if widget.value.text() else "?"
        geo = (widget.width(), widget.x())
        samples.setdefault(f"stat:{id(widget) % 1000}", []).append(geo)
    # Also watch a panel next to the strip, to catch the reflow it causes.
    if window.title_label is not None:
        samples.setdefault(
            "title_label", []
        ).append((window.title_label.width(), window.title_label.x()))


timer = QTimer()
timer.setInterval(200)
timer.timeout.connect(sample)
timer.start()


def report() -> None:
    timer.stop()
    print("  widget                 distinct widths   width range   x range")
    print("  " + "-" * 68)
    moved = False
    for name, rows in samples.items():
        widths = sorted({w for w, _ in rows})
        xs = sorted({x for _, x in rows})
        if len(widths) > 1 or len(xs) > 1:
            moved = True
        print(
            f"  {name:22s} {len(widths):>6d}          "
            f"{widths[0]:4d}-{widths[-1]:<4d}     {xs[0]:4d}-{xs[-1]:<4d}"
        )
    print()
    print("  VERDICT:", "JITTERS - layout is moving" if moved else "STATIC - nothing moved")
    window.shutdown()
    app.quit()


QTimer.singleShot(SECONDS * 1000, report)
QTimer.singleShot(SECONDS * 1000 + 30_000, app.quit)
app.exec()
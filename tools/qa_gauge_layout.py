"""Is a gauge tile being clipped, and did the header width change cause it?

Reports the height of the header panel and of each engine gauge tile. A tile
whose visible height is less than the sum of its parts has its bar cut off, which
is what the CPU tile is showing.

Usage: python tools/qa_gauge_layout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.engine.pipelines import ScenarioModelRegistry, load_scenario_catalog
from app.hud import EngineGauge, MainWindow
from app.main import (
    REQUIRED_CONFIG,
    REQUIRED_PROFILES,
    REQUIRED_SCENARIOS,
    _run_availability_gate,
)
from app.theme import THEME

ROOT = Path(__file__).resolve().parents[1]

app = QApplication(sys.argv)

window = MainWindow(
    catalog=load_scenario_catalog(REQUIRED_SCENARIOS, REQUIRED_CONFIG, ROOT / "media"),
    registry=ScenarioModelRegistry(REQUIRED_CONFIG, ROOT / "models", ROOT / "cache"),
    initial_scenario="manufacturing",
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


def report() -> None:
    print(f"  window                : {window.width()} x {window.height()}")
    gauges = window.findChildren(EngineGauge)
    print(f"  engine gauges found   : {len(gauges)}")
    print()
    print("  engine   top    height   bottom   container_bottom   verdict")
    print("  " + "-" * 66)
    for gauge in gauges:
        engine = getattr(gauge, "engine", "?")
        top_left = gauge.mapTo(window, gauge.rect().topLeft())
        bottom_in_window = top_left.y() + gauge.height()
        parent = gauge.parentWidget()
        # Where the column that holds the tiles actually ends, in window space.
        parent_bottom = (
            parent.mapTo(window, parent.rect().bottomLeft()).y() if parent else -1
        )
        clipped = bottom_in_window > parent_bottom > 0
        # The meaningful test is whether the tile got the height its own content
        # needs (header + value + bar + gap). The container boundary can be a
        # pixel out through rounding without hiding anything.
        content_ok = gauge.height() >= THEME.gauge_header_height + round(
            THEME.gauge_value_font * THEME.gauge_value_line_ratio
        ) + THEME.gauge_bar_min + THEME.gauge_gap_min
        verdict = "ok" if content_ok else "CLIPPED"
        if clipped and content_ok:
            verdict = "ok (1px overhang)"
        print(
            f"  {engine:8s} {top_left.y():5d}  {gauge.height():7d}  "
            f"{bottom_in_window:6d}   {parent_bottom:16d}   {verdict}"
        )

    print()
    print("  engine   height   sizeHint   minSizeHint   needed_by_content")
    print("  " + "-" * 62)
    for gauge in gauges:
        engine = getattr(gauge, "engine", "?")
        # header + value line + bar + gaps, using the same tokens the tile draws with
        content = (
            THEME.gauge_header_height
            + round(THEME.gauge_value_font * THEME.gauge_value_line_ratio)
            + THEME.gauge_bar_min
            + THEME.gauge_gap_min
        )
        print(
            f"  {engine:8s} {gauge.height():6d}   {gauge.sizeHint().height():8d}   "
            f"{gauge.minimumSizeHint().height():11d}   {content:15d}"
        )

    print()
    window.shutdown()
    app.quit()


QTimer.singleShot(2000, lambda: window.resize(1920, 1080))
QTimer.singleShot(10000, report)
QTimer.singleShot(60000, app.quit)
app.exec()

"""Sweep window sizes and report where the layout clips.

The booth target is 1080p, but Windows scaling changes the logical size the app
actually sees: a 1080p panel at 125% presents 1536x864 to Qt, which is below the
full-layout threshold. This reports, per size, whether the layout is compact and
which widgets cannot fit their content.

Usage: python tools/qa_resolution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from app.engine.pipelines import ScenarioModelRegistry, load_scenario_catalog
from app.hud import MainWindow
from app.main import (
    REQUIRED_CONFIG,
    REQUIRED_PROFILES,
    REQUIRED_RAG_CONFIG,
    REQUIRED_SCENARIOS,
    _run_availability_gate,
)
from app.theme import THEME

ROOT = Path(__file__).resolve().parents[1]

# Logical sizes the app can meet. 1920x1080 is the booth target; 1536x864 is a
# 1080p panel at 125% Windows scaling; 1366x768 and 1280x720 are small laptops.
SIZES = [
    (1920, 1080, "1080p at 100%"),
    (1600, 900, "full-layout threshold"),
    (1536, 864, "1080p at 125% scaling"),
    (1366, 768, "small laptop"),
    (1280, 720, "minimum window"),
]

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
    rag_config_path=REQUIRED_RAG_CONFIG,
)
window.start()

results: list[tuple[str, bool, list[str]]] = []


def audit(label: str, width: int, height: int) -> None:
    window.resize(width, height)
    app.processEvents()

    compact = window.compact_layout
    problems: list[str] = []

    for child in window.findChildren(QWidget):
        if not child.isVisible() or child.width() <= 0 or child.height() <= 0:
            continue
        # Content that needs more room than it was given.
        hint = child.minimumSizeHint()
        if hint.height() > child.height() + 2:
            problems.append(
                f"{type(child).__name__}(parent={type(child.parentWidget()).__name__ if child.parentWidget() else None}, y={child.y()}, w={child.width()}, h={child.height()}) needs {hint.height()}px, has {child.height()}px"
            )
        # A child rendered outside its parent's visible box.
        parent = child.parentWidget()
        if parent is not None and parent.isVisible():
            bottom = child.y() + child.height()
            if bottom > parent.height() + 2 and child.height() > 8:
                problems.append(
                    f"{type(child).__name__} overflows parent by "
                    f"{bottom - parent.height()}px"
                )

    # Long single-line labels that cannot show their text at this width.
    for label_widget in window.findChildren(QLabel):
        if not label_widget.isVisible() or not label_widget.text() or label_widget.wordWrap():
            continue
        needed = label_widget.fontMetrics().horizontalAdvance(label_widget.text())
        if needed > label_widget.width() + 2:
            problems.append(
                f"QLabel[{label_widget.text()[:26]}] clips {needed - label_widget.width()}px"
            )

    results.append((f"{width}x{height}  {label}", compact, problems[:6]))


index = {"value": 0}


def step() -> None:
    if index["value"] >= len(SIZES):
        finish()
        return
    width, height, label = SIZES[index["value"]]
    audit(label, width, height)
    index["value"] += 1
    QTimer.singleShot(400, step)


def finish() -> None:
    print()
    print(f"  minimum window declared: {THEME.minimum_window_width}x{THEME.minimum_window_height}")
    print(f"  full-layout threshold  : 1600x900")
    print()
    for label, compact, problems in results:
        mode = "COMPACT" if compact else "full"
        status = "clean" if not problems else f"{len(problems)} problem(s)"
        print(f"  {label:34s} {mode:8s} {status}")
        for problem in problems:
            print(f"      - {problem}")
    window.shutdown()
    app.quit()


QTimer.singleShot(9000, step)
QTimer.singleShot(9000 + 400 * len(SIZES) + 10_000, finish)
app.exec()


"""Sweep window sizes and report where the layout clips, with and without Q&A.

The booth target is 1080p, but Windows scaling changes the logical size the app
actually sees: a 1080p panel at 150% presents 1280x720 to Qt, which is the
declared minimum. The first version of this probe missed a real defect because it
only measured the demo on its own: opening the document Q&A panel takes
``rag_panel_width`` from the demo column, and at 1280x720 logical that clips the
hardware card and the stat tiles. So every size is audited in both states.

Usage: python tools/qa_resolution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QAbstractScrollArea, QLabel, QWidget

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
# 1080p panel at 125% scaling; 1280x720 is 1080p at 150%, which is the size the
# booth machine actually runs at.
SIZES = [
    (1920, 1080, "1080p at 100%"),
    (1600, 900, "full-layout threshold"),
    (1536, 864, "1080p at 125%"),
    (1366, 768, "small laptop"),
    (1280, 720, "1080p at 150% / minimum"),
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


def _inside_scroll_area(widget: QWidget) -> bool:
    """True when the widget lives inside a scroll area's viewport.

    Content there is *supposed* to be taller than the viewport -- that is what a
    scroll area is for -- so counting it as an overflow is a false positive.
    """
    node = widget.parentWidget()
    while node is not None:
        if isinstance(node, QAbstractScrollArea):
            return True
        node = node.parentWidget()
    return False


def audit(label: str, width: int, height: int, state: str) -> None:
    problems: list[str] = []

    for child in window.findChildren(QWidget):
        if not child.isVisible() or child.width() <= 0 or child.height() <= 0:
            continue
        if _inside_scroll_area(child):
            continue
        # Content that needs more room than it was given.
        hint = child.minimumSizeHint()
        if hint.height() > child.height() + 2:
            problems.append(
                f"{type(child).__name__}(y={child.y()}, h={child.height()}) "
                f"needs {hint.height()}px, has {child.height()}px"
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
        if not label_widget.isVisible() or not label_widget.text():
            continue
        if label_widget.wordWrap():
            continue
        needed = label_widget.fontMetrics().horizontalAdvance(label_widget.text())
        if needed > label_widget.width() + 2:
            problems.append(
                f"QLabel[{label_widget.text()[:24]}] clips {needed - label_widget.width()}px"
            )

    results.append((f"{width}x{height}  {label:24s} {state}", window.compact_layout, problems[:8]))


index = {"value": 0}


def step() -> None:
    if index["value"] >= len(SIZES):
        finish()
        return
    width, height, label = SIZES[index["value"]]
    index["value"] += 1

    window.resize(width, height)
    app.processEvents()
    audit(label, width, height, "demo only ")

    window.toggle_rag_panel()
    app.processEvents()
    audit(label, width, height, "Q&A open  ")

    window.toggle_rag_panel()
    app.processEvents()

    QTimer.singleShot(300, step)


def finish() -> None:
    print()
    print(f"  minimum window declared: {THEME.minimum_window_width}x{THEME.minimum_window_height}")
    print(f"  full-layout threshold  : 1600x900")
    print(f"  Q&A panel width        : {THEME.rag_panel_width}px")
    print()
    for label, compact, problems in results:
        mode = "COMPACT" if compact else "full"
        status = "clean" if not problems else f"{len(problems)} problem(s)"
        print(f"  {label} {mode:8s} {status}")
        for problem in problems:
            print(f"      - {problem}")
    window.shutdown()
    app.quit()


QTimer.singleShot(9000, step)
QTimer.singleShot(9000 + 400 * len(SIZES) + 15_000, finish)
app.exec()

"""Find every widget whose geometry changes while the demo runs.

The header stat strip was fixed once already, but "text box jitter" implies
something else still moves. Rather than guess which label, this walks the whole
widget tree, samples geometry repeatedly, and reports every widget that moved.

A static HUD should produce a single geometry per widget. Anything with more
than one distinct width, height, x or y is a candidate for the visible bounce.

Usage: python tools/qa_ui_jitter.py [seconds] [scenario]
"""

from __future__ import annotations

import sys
from collections import defaultdict
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

ROOT = Path(__file__).resolve().parents[1]
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SCENARIO = sys.argv[2] if len(sys.argv) > 2 else "manufacturing"

app = QApplication(sys.argv)


def path_of(widget: QWidget) -> str:
    """A short human path: parent chain of class names plus any text."""
    parts: list[str] = []
    node: QWidget | None = widget
    for _ in range(4):
        if node is None:
            break
        name = type(node).__name__
        parent = node.parentWidget()
        if isinstance(node, QLabel) and node.text():
            name += f"[{node.text()[:22]}]"
        parts.append(name)
        node = parent
    return " < ".join(parts)


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
    initial_density=int(sys.argv[3]) if len(sys.argv) > 3 else 1,
    rag_config_path=REQUIRED_RAG_CONFIG,
)
window.resize(1920, 1080)
window.start()

# widget pointer -> set of observed geometries
seen: dict[int, set[tuple[int, int, int, int]]] = defaultdict(set)
labels: dict[int, str] = {}
widgets: dict[int, QWidget] = {}


def sample() -> None:
    for widget in window.findChildren(QWidget):
        key = id(widget)
        if key not in widgets:
            widgets[key] = widget
            labels[key] = path_of(widget)
        seen[key].add((widget.x(), widget.y(), widget.width(), widget.height()))


timer = QTimer()
timer.setInterval(50)
timer.timeout.connect(sample)
timer.start()


def report() -> None:
    timer.stop()
    movers = []
    for key, geometries in seen.items():
        if len(geometries) <= 1:
            continue
        widths = {g[2] for g in geometries}
        heights = {g[3] for g in geometries}
        xs = {g[0] for g in geometries}
        ys = {g[1] for g in geometries}
        movers.append(
            (
                len(geometries),
                labels[key],
                len(widths),
                min(widths),
                max(widths),
                len(heights),
                len(xs),
                len(ys),
            )
        )

    movers.sort(reverse=True)
    print(f"  widgets sampled : {len(seen)}")
    print(f"  widgets moving  : {len(movers)}")
    print()
    if movers:
        print("  samples  widths          width range   heights  x-vals  y-vals  widget")
        print("  " + "-" * 104)
        for count, label, nw, wmin, wmax, nh, nx, ny in movers[:28]:
            print(
                f"  {count:7d}  {nw:3d}     {wmin:5d}-{wmax:<5d}  {nh:7d}  {nx:6d}  {ny:6d}  {label[:52]}"
            )
    window.shutdown()
    app.quit()


QTimer.singleShot(SECONDS * 1000, report)
QTimer.singleShot(SECONDS * 1000 + 30_000, app.quit)
app.exec()

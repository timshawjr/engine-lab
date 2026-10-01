"""Federal and Aerospace perimeter helpers."""

from __future__ import annotations

from app.engine.stages import Detection


def is_plate(detection: Detection) -> bool:
    return detection.label == "license plate"

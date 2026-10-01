"""Metro classification and counting helpers."""

from __future__ import annotations

from app.engine.stages import Detection


def is_vehicle(detection: Detection) -> bool:
    return detection.label in {"vehicle", "car", "bus", "truck"}


def is_person(detection: Detection) -> bool:
    return detection.label == "person"

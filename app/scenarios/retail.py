"""Retail-specific event helpers."""

from __future__ import annotations

from app.engine.stages import Detection


def classification_candidates(
    detections: tuple[Detection, ...],
    confidence_min: float,
    top_k: int,
) -> tuple[Detection, ...]:
    eligible = tuple(
        detection
        for detection in detections
        if detection.confidence >= confidence_min
        and detection.width >= 1.0
        and detection.height >= 1.0
    )
    return tuple(sorted(eligible, key=lambda item: item.confidence, reverse=True)[:top_k])

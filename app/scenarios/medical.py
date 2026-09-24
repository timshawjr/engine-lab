"""Medical posture helpers."""

from __future__ import annotations

from collections.abc import Iterable

from app.engine.events import torso_angle_degrees
from app.engine.stages import Keypoint


def confident_pose_angle(
    keypoints: Iterable[Keypoint],
    confidence_min: float,
) -> float | None:
    values = {keypoint.name: keypoint for keypoint in keypoints}
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(
        name not in values or values[name].confidence < confidence_min
        for name in required
    ):
        return None
    return torso_angle_degrees(values.values())

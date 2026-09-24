"""Measured business-event generation for the four scenario graphs."""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterable

from app.engine.stages import Detection

if TYPE_CHECKING:
    from app.engine.pipelines import ScenarioConfig, Zone


@dataclass(frozen=True)
class BusinessEvent:
    ts: float
    type: str
    label: str
    confidence: float
    zone: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Track:
    track_id: int
    label: str
    box: tuple[float, float, float, float]
    first_seen: float
    last_seen: float
    started_in_shelf: bool
    shelf_name: str
    outside_since: float | None = None
    inside_zone: str = ""
    inside_since: float | None = None
    picked_emitted: bool = False
    classified_at: float = 0.0
    plate_emitted: bool = False


def _iou(left: Detection, right: Detection) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_in_zone(detection: Detection, zone: "Zone") -> bool:
    center_x = detection.x1 + detection.width / 2.0
    center_y = detection.y1 + detection.height / 2.0
    x, y, width, height = zone.roi
    return x <= center_x <= x + width and y <= center_y <= y + height


def _center(detection: Detection) -> tuple[float, float]:
    return detection.x1 + detection.width / 2.0, detection.y1 + detection.height / 2.0


class EventTracker:
    def __init__(self, scenario: "ScenarioConfig", stream_index: int) -> None:
        self.scenario = scenario
        self.stream_index = stream_index
        self._next_track_id = 1
        self._tracks: dict[int, _Track] = {}
        self.events: deque[BusinessEvent] = deque(maxlen=600)
        self._batch_events: list[BusinessEvent] = []
        self._last_posture_event = 0.0
        self._posture_since: float | None = None

    def reset(self, scenario: "ScenarioConfig") -> None:
        self.scenario = scenario
        self._tracks.clear()
        self._batch_events.clear()
        self._next_track_id = 1
        self._last_posture_event = 0.0
        self._posture_since = None

    def _new_track(self, detection: Detection, now: float) -> _Track:
        shelf = next(
            (
                zone
                for zone in self.scenario.zones
                if zone.kind == "shelf" and _center_in_zone(detection, zone)
            ),
            None,
        )
        track = _Track(
            track_id=self._next_track_id,
            label=detection.label,
            box=(detection.x1, detection.y1, detection.x2, detection.y2),
            first_seen=now,
            last_seen=now,
            started_in_shelf=shelf is not None,
            shelf_name=shelf.name if shelf is not None else "",
        )
        self._next_track_id += 1
        self._tracks[track.track_id] = track
        return track

    def _assign_tracks(
        self,
        detections: tuple[Detection, ...],
        now: float,
    ) -> list[tuple[Detection, _Track]]:
        rules = self.scenario.event_rules
        iou_threshold = float(rules.get("tracking_iou_threshold", rules["iou_threshold"]))
        persist_seconds = float(rules["track_persist_s"])
        unmatched = set(self._tracks)
        assignments: list[tuple[Detection, _Track]] = []
        for detection in detections:
            candidates = sorted(
                (
                    (_iou(detection, Detection(*track.box, track.label, 1.0)), track)
                    for track_id, track in self._tracks.items()
                    if track_id in unmatched
                    and track.label == detection.label
                    and now - track.last_seen <= persist_seconds
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if candidates and candidates[0][0] >= iou_threshold:
                _, track = candidates[0]
                unmatched.remove(track.track_id)
            else:
                track = self._new_track(detection, now)
            track.box = (detection.x1, detection.y1, detection.x2, detection.y2)
            track.last_seen = now
            assignments.append((detection.with_track(track.track_id), track))
        for track_id in unmatched:
            track = self._tracks[track_id]
            if now - track.last_seen > persist_seconds:
                self._tracks.pop(track_id, None)
        return assignments

    def _emit(
        self,
        event_type: str,
        detection: Detection,
        zone: str,
        now: float,
    ) -> BusinessEvent:
        is_classification = event_type == "object_classified"
        event = BusinessEvent(
            ts=now,
            type=event_type,
            label=detection.classification if is_classification else detection.label,
            confidence=float(
                detection.classification_confidence
                if is_classification
                else detection.confidence
            ),
            zone=zone,
        )
        self.events.append(event)
        self._batch_events.append(event)
        return event

    def update(
        self,
        detections: tuple[Detection, ...],
        *,
        now: float | None = None,
        posture_angles: tuple[float, ...] = (),
    ) -> tuple[tuple[Detection, ...], tuple[BusinessEvent, ...]]:
        timestamp = time.time() if now is None else now
        self._batch_events.clear()
        rules = self.scenario.event_rules
        assignments = self._assign_tracks(detections, timestamp)
        tracked: list[Detection] = []
        for detection, track in assignments:
            if detection.classification:
                if timestamp - track.classified_at >= float(
                    rules["event_cooldown_s"]
                ):
                    self._emit("object_classified", detection, track.inside_zone, timestamp)
                    track.classified_at = timestamp

            inside_zones = [
                zone for zone in self.scenario.zones
                if _center_in_zone(detection, zone)
            ]
            if inside_zones:
                zone = inside_zones[0]
                if track.inside_zone != zone.name:
                    track.inside_since = timestamp
                dwell = timestamp - (track.inside_since or timestamp)
                if dwell >= float(rules["count_dwell_s"]):
                    if zone.kind == "person_entry" and detection.label == "person":
                        self._emit("person_counted", detection, zone.name, timestamp)
                    elif zone.kind in {"vehicle_entry", "vehicle_exit"} and detection.label in {
                        "vehicle",
                        "car",
                        "bus",
                        "truck",
                    }:
                        self._emit("vehicle_counted", detection, zone.name, timestamp)
                track.inside_zone = zone.name
            else:
                track.inside_zone = ""
                track.inside_since = None

            if (
                track.started_in_shelf
                and not any(
                    zone.kind == "shelf" and _center_in_zone(detection, zone)
                    for zone in self.scenario.zones
                )
            ):
                if track.outside_since is None:
                    track.outside_since = timestamp
                if (
                    not track.picked_emitted
                    and timestamp - track.outside_since
                    >= float(rules.get("picked_up_dwell_s", 0.0))
                ):
                    self._emit("object_picked_up", detection, track.shelf_name, timestamp)
                    track.picked_emitted = True
            else:
                track.outside_since = None

            if (
                detection.label == "license plate"
                and not track.plate_emitted
                and detection.confidence
                >= float(rules.get("plate_confidence_min", detection.confidence))
            ):
                self._emit("plate_detected", detection, "perimeter", timestamp)
                track.plate_emitted = True
            tracked.append(detection)

        if posture_angles:
            fall_angle = float(rules.get("fall_angle_deg", 180.0))
            cooldown = float(rules["event_cooldown_s"])
            if max(posture_angles) >= fall_angle:
                if self._posture_since is None:
                    self._posture_since = timestamp
                if (
                    timestamp - self._posture_since
                    >= float(rules.get("posture_dwell_s", 0.0))
                    and timestamp - self._last_posture_event >= cooldown
                    and assignments
                ):
                    detection = assignments[0][0]
                    self._emit("posture_alert", detection, "monitoring_zone", timestamp)
                    self._last_posture_event = timestamp
            else:
                self._posture_since = None

        new_events = tuple(self._batch_events)
        self._batch_events.clear()
        return tuple(tracked), new_events

    def rolling_counts(
        self,
        now: float | None = None,
        window_seconds: float | None = None,
    ) -> dict[str, int]:
        timestamp = time.time() if now is None else now
        if window_seconds is None:
            window_seconds = float(
                self.scenario.event_rules["event_window_s"]
            )
        return dict(
            Counter(
                event.type
                for event in self.events
                if timestamp - event.ts <= window_seconds
            )
        )

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        return {
            "stream_index": self.stream_index,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "label": track.label,
                    "first_seen": track.first_seen,
                    "last_seen": track.last_seen,
                }
                for track in self._tracks.values()
            ],
            "rolling_60s": self.rolling_counts(now),
            "events": [event.to_dict() for event in self.events],
        }


def torso_angle_degrees(keypoints: Iterable[Any]) -> float | None:
    values = {keypoint.name: keypoint for keypoint in keypoints}
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(values.get(name) is None for name in required):
        return None
    if any(values[name].confidence < 0.0 for name in required):
        return None
    shoulder_x = (values["left_shoulder"].x + values["right_shoulder"].x) / 2.0
    shoulder_y = (values["left_shoulder"].y + values["right_shoulder"].y) / 2.0
    hip_x = (values["left_hip"].x + values["right_hip"].x) / 2.0
    hip_y = (values["left_hip"].y + values["right_hip"].y) / 2.0
    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y
    if math.hypot(dx, dy) <= 1e-6:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))

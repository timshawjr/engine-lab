"""Measured business-event generation for the four scenario graphs."""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
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
    classification_label: str = ""
    classification_streak: int = 0
    emitted_classifications: set[str] = field(default_factory=set)
    counted_zones: set[str] = field(default_factory=set)
    plate_emitted: bool = False
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    hits: int = 0
    misses: int = 0


def _iou(left: Detection, right: Detection) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_in_zone(
    detection: Detection,
    zone: "Zone",
    source_size: tuple[int, int],
) -> bool:
    """Test a detection centre against a normalized ROI.

    ``zone.roi`` is normalized to 0..1 but detections are in source pixels, so
    the centre has to be normalized before it is compared. Skipping that
    division silently makes every zone test fail, which previously left
    ``person_counted``, ``vehicle_counted`` and ``object_picked_up`` unable to
    ever fire. ``tools/review_sessions.py`` always did the conversion, which is
    why the bug survived so long: the review harness reported zone activity the
    live tracker was not actually seeing.
    """

    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        return False
    center_x = _center(detection)[0] / source_width
    center_y = _center(detection)[1] / source_height
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
        # Detections arrive in source pixels while zone ROIs are normalized, so
        # the tracker needs the frame size to test membership. It is set from
        # the pipeline metrics on every update.
        self._source_size: tuple[int, int] = (0, 0)

    def reset(self, scenario: "ScenarioConfig") -> None:
        self.scenario = scenario
        self._tracks.clear()
        self._batch_events.clear()
        self._next_track_id = 1
        self._last_posture_event = 0.0
        self._posture_since = None
        self._source_size = (0, 0)

    def _may_raise_business_event(self, detection: Detection) -> bool:
        """Gate business events on the scenario's declared detector classes.

        YOLO hallucinates objects in a retail scene: on the measured store-aisle
        footage it reported shelf hardware as ``microwave``, ``oven`` and ``tv``,
        and it also reports the shopper. Without this gate the tracker could
        dutifully log ``object_picked_up person``, which is not a statement worth
        making. A scenario opts in with ``business_event_labels``; an empty or
        missing list means unrestricted, which is the pre-existing behaviour for
        the other graphs.
        """

        allowed = self.scenario.event_rules.get("business_event_labels")
        if not allowed:
            return True
        return detection.label.lower() in {str(v).strip().lower() for v in allowed}

    def _in_zone(self, detection: Detection, zone: "Zone") -> bool:
        return _center_in_zone(detection, zone, self._source_size)

    def _new_track(self, detection: Detection, now: float) -> _Track:
        shelf = next(
            (
                zone
                for zone in self.scenario.zones
                if zone.kind == "shelf" and self._in_zone(detection, zone)
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
            hits=1,
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
        min_iou = float(rules.get("tracking_min_iou", 0.05))
        center_threshold = float(rules.get("tracking_center_distance", 1.0))
        persist_seconds = float(rules["track_persist_s"])
        max_misses = int(rules.get("max_track_misses", 2))
        existing_track_ids = set(self._tracks)
        assigned_tracks: set[int] = set()
        assignments: list[tuple[Detection, _Track]] = []

        # Keep the original per-detection IoU priority. Motion is only a
        # fallback for a detector dropout or a moving box; global greedy
        # pairing can swap two same-label retail objects in a crowded frame.
        for detection in detections:
            detection_center = _center(detection)
            candidates: list[tuple[float, int]] = []
            fallback_candidates: list[tuple[float, int]] = []
            for track_id, track in self._tracks.items():
                if (
                    track_id in assigned_tracks
                    or track.label != detection.label
                    or now - track.last_seen > persist_seconds
                ):
                    continue
                track_detection = Detection(*track.box, track.label, 1.0)
                current_iou = _iou(detection, track_detection)
                predicted_box = (
                    track.box[0] + track.velocity_x,
                    track.box[1] + track.velocity_y,
                    track.box[2] + track.velocity_x,
                    track.box[3] + track.velocity_y,
                )
                predicted_iou = _iou(
                    detection,
                    Detection(*predicted_box, track.label, 1.0),
                )
                track_center = _center(track_detection)
                predicted_center = (
                    track_center[0] + track.velocity_x,
                    track_center[1] + track.velocity_y,
                )
                scale = max(1.0, track_detection.width, track_detection.height)
                center_distance = math.hypot(
                    detection_center[0] - predicted_center[0],
                    detection_center[1] - predicted_center[1],
                ) / scale
                center_score = max(0.0, 1.0 - center_distance / center_threshold)
                if current_iou >= iou_threshold:
                    # Preserve the original IoU-first decision whenever a
                    # real overlap exists; this avoids same-label ID swaps.
                    candidates.append((current_iou, track_id))
                elif predicted_iou >= min_iou and center_distance <= center_threshold:
                    fallback_candidates.append((predicted_iou + 0.10 * center_score, track_id))

            if candidates:
                _score, track_id = max(candidates)
            elif fallback_candidates:
                _score, track_id = max(fallback_candidates)
            else:
                track_id = 0
            if track_id:
                track = self._tracks[track_id]
                old_center = _center(Detection(*track.box, track.label, 1.0))
                new_center = _center(detection)
                track.velocity_x = 0.5 * track.velocity_x + 0.5 * (new_center[0] - old_center[0])
                track.velocity_y = 0.5 * track.velocity_y + 0.5 * (new_center[1] - old_center[1])
                track.box = (detection.x1, detection.y1, detection.x2, detection.y2)
                track.last_seen = now
                track.hits += 1
                track.misses = 0
                assigned_tracks.add(track_id)
            else:
                track = self._new_track(detection, now)
            assignments.append((detection.with_track(track.track_id), track))

        for track_id in existing_track_ids - assigned_tracks:
            track = self._tracks.get(track_id)
            if track is None:
                continue
            track.misses += 1
            if (
                track.misses > max_misses
                or now - track.last_seen > persist_seconds
            ):
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
        if is_classification:
            # A declared-vocabulary name tells an operator more than the COCO
            # class: "pot" is correct where the 80-class head says "bowl".
            event_label = detection.classification
        else:
            event_label = detection.label
        event = BusinessEvent(
            ts=now,
            type=event_type,
            label=event_label,
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
        source_size: tuple[int, int] = (0, 0),
    ) -> tuple[tuple[Detection, ...], tuple[BusinessEvent, ...]]:
        timestamp = time.time() if now is None else now
        self._batch_events.clear()
        if source_size != (0, 0):
            self._source_size = source_size
        rules = self.scenario.event_rules
        assignments = self._assign_tracks(detections, timestamp)
        # A single-frame ImageNet top-1 is noise. Require the same label to
        # repeat on this track before it is displayed or emitted.
        stable_frames = max(1, int(rules.get("classify_min_frames", 1)))
        tracked: list[Detection] = []
        for detection, track in assignments:
            if detection.classification:
                if detection.classification == track.classification_label:
                    track.classification_streak += 1
                else:
                    track.classification_label = detection.classification
                    track.classification_streak = 1
                if track.classification_streak < stable_frames:
                    detection = detection.with_classification("", 0.0)
            else:
                track.classification_label = ""
                track.classification_streak = 0
            if (
                detection.classification
                and detection.classification not in track.emitted_classifications
                and timestamp - track.classified_at >= float(rules["event_cooldown_s"])
            ):
                self._emit("object_classified", detection, track.inside_zone, timestamp)
                track.emitted_classifications.add(detection.classification)
                track.classified_at = timestamp

            inside_zones = [
                zone for zone in self.scenario.zones
                if self._in_zone(detection, zone)
            ]
            if inside_zones:
                zone = inside_zones[0]
                if track.inside_zone != zone.name:
                    track.inside_since = timestamp
                dwell = timestamp - (track.inside_since or timestamp)
                # Count each track once per zone. Without this guard the event
                # re-fires on every single frame the track dwells in the zone,
                # which produced 2,459 "vehicle_counted" events in 25 seconds and
                # made the rolling count meaningless.
                if (
                    dwell >= float(rules["count_dwell_s"])
                    and zone.name not in track.counted_zones
                ):
                    if zone.kind == "person_entry" and detection.label == "person":
                        self._emit("person_counted", detection, zone.name, timestamp)
                        track.counted_zones.add(zone.name)
                    elif zone.kind in {"vehicle_entry", "vehicle_exit"} and detection.label in {
                        "vehicle",
                        "car",
                        "bus",
                        "truck",
                    }:
                        self._emit("vehicle_counted", detection, zone.name, timestamp)
                        track.counted_zones.add(zone.name)
                track.inside_zone = zone.name
            else:
                track.inside_zone = ""
                track.inside_since = None

            if (
                track.started_in_shelf
                and not any(
                    zone.kind == "shelf" and self._in_zone(detection, zone)
                    for zone in self.scenario.zones
                )
            ):
                if track.outside_since is None:
                    track.outside_since = timestamp
                if (
                    not track.picked_emitted
                    and timestamp - track.outside_since
                    >= float(rules.get("picked_up_dwell_s", 0.0))
                    and self._may_raise_business_event(detection)
                ):
                    # Gated like every other business event: a shopper walking
                    # out of a shelf zone is not a product being taken, and
                    # "object_picked_up person" is not a statement worth making.
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

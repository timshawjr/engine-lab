#!/usr/bin/env python3
"""Record and summarize raw-vs-tracked scenario output for offline review.

This is a development-only tool. It uses the production pipeline but writes
frame-level evidence that the normal HUD intentionally does not retain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DEPS = Path(r"C:\Users\Intel Demo\AppData\Local\Temp\opencode\engine-lab-review-deps")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if REVIEW_DEPS.is_dir() and str(REVIEW_DEPS) not in sys.path:
    sys.path.insert(0, str(REVIEW_DEPS))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from app.engine.availability import probe_device_availability
from app.engine.device_policy import DeviceMode, DevicePolicy
from app.engine.pipelines import (
    PipelineFrame,
    ScenarioModelRegistry,
    ScenarioStreamWorker,
    build_stage_assignments,
    load_scenario_catalog,
)
from app.engine.stages import Detection
from app.telemetry.npu_fallback import NpuDutyCycle

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


TRACK_COLORS = (
    (80, 220, 255),
    (100, 240, 150),
    (255, 210, 80),
    (255, 120, 220),
    (180, 140, 255),
)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _detection_dict(detection: Detection) -> dict[str, Any]:
    return {
        "x1": detection.x1,
        "y1": detection.y1,
        "x2": detection.x2,
        "y2": detection.y2,
        "label": detection.label,
        "confidence": detection.confidence,
        "classification": detection.classification,
        "classification_confidence": detection.classification_confidence,
        "track_id": detection.track_id,
        "keypoints": [asdict(point) for point in detection.keypoints],
    }


def _color_for_track(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (255, 180, 80)
    return TRACK_COLORS[(int(track_id) - 1) % len(TRACK_COLORS)]


def _draw_zones(frame: np.ndarray, zones: tuple[Any, ...]) -> None:
    height, width = frame.shape[:2]
    for zone in zones:
        x, y, zone_width, zone_height = zone.roi
        p1 = (int(x * width), int(y * height))
        p2 = (int((x + zone_width) * width), int((y + zone_height) * height))
        cv2.rectangle(frame, p1, p2, (80, 210, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, zone.name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 210, 240), 1, cv2.LINE_AA)


def _draw_detection(frame: np.ndarray, detection: Detection, raw: bool) -> None:
    height, width = frame.shape[:2]
    p1 = (max(0, int(detection.x1)), max(0, int(detection.y1)))
    p2 = (min(width - 1, int(detection.x2)), min(height - 1, int(detection.y2)))
    color = (40, 170, 255) if raw else _color_for_track(detection.track_id)
    thickness = 1 if raw else 2
    cv2.rectangle(frame, p1, p2, color, thickness, cv2.LINE_AA)
    if not raw:
        parts = [f"{detection.label} {detection.confidence:.2f}"]
        if detection.track_id is not None:
            parts.append(f"id={detection.track_id}")
        if detection.classification:
            parts.append(f"{detection.classification} {detection.classification_confidence:.2f}")
        label = " | ".join(parts)
        font_scale = 0.42
        thickness_text = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)
        text_y = max(text_height + baseline + 2, p1[1] - 4)
        cv2.rectangle(frame, (p1[0], text_y - text_height - baseline - 2), (p1[0] + text_width + 6, text_y + 2), (7, 16, 24), -1)
        cv2.putText(frame, label, (p1[0] + 3, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 250, 252), thickness_text, cv2.LINE_AA)
    for point in detection.keypoints:
        center = (int(point.x), int(point.y))
        point_color = (255, 220, 80) if point.confidence >= 0.3 else (120, 120, 120)
        cv2.circle(frame, center, 3, point_color, -1, cv2.LINE_AA)


def _annotate(frame: np.ndarray, result: PipelineFrame) -> np.ndarray:
    annotated = frame.copy()
    _draw_zones(annotated, result.zones)
    for detection in result.raw_detections:
        _draw_detection(annotated, detection, raw=True)
    for detection in result.detections:
        _draw_detection(annotated, detection, raw=False)
    cv2.putText(annotated, f"{result.scenario_id} frame={result.metrics.frame_index} tracks={len(result.detections)}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, "orange=raw  colored=tracked", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 190, 80), 1, cv2.LINE_AA)
    return annotated


class SourceReader:
    def __init__(self, path: Path) -> None:
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open review source: {path}")
        self.frame_count = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.next_index = 0

    def read(self, frame_index: int) -> np.ndarray | None:
        source_index = max(0, (frame_index - 1) % self.frame_count)
        if source_index != self.next_index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        self.next_index = source_index + 1
        return frame

    def close(self) -> None:
        self.capture.release()


def _contact_sheet(frames: list[tuple[int, np.ndarray]], path: Path) -> None:
    if not frames or Image is None or ImageDraw is None:
        return
    thumb_width, thumb_height = 420, 260
    columns = 3
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (frame_index, frame) in enumerate(frames):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((thumb_width, thumb_height))
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + 28)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + thumb_height + 5), f"frame {frame_index}", fill=(20, 35, 45))
    sheet.save(path)


def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = load_scenario_catalog(ROOT / "config" / "scenarios.json", ROOT / "config" / "models.json", ROOT / "media")
    registry = ScenarioModelRegistry(ROOT / "config" / "models.json", ROOT / "models", ROOT / "cache")
    availability = probe_device_availability(ROOT / "config" / "models.json", ROOT / "cache")
    registry.prewarm(catalog, availability)
    scenario = catalog[args.scenario]
    policy = DevicePolicy(
        available_devices=availability.available_devices,
        mode=DeviceMode.SPREAD,
        density=1,
    )
    assignments = build_stage_assignments(policy, scenario, 0, availability=availability)
    media_path = catalog.media_path(scenario.id, camera_index=None)
    if media_path is None:
        raise RuntimeError("review currently requires a local scenario video")

    app = QApplication.instance() or QApplication([])
    source = SourceReader(media_path)
    writer_path = output / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(writer_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(source.capture.get(cv2.CAP_PROP_FPS)) or 30.0,
        (int(source.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create review video: {writer_path}")

    records: list[dict[str, Any]] = []
    samples: list[tuple[int, np.ndarray]] = []
    track_stats: dict[int, dict[str, Any]] = defaultdict(lambda: {"frames": 0, "first_frame": None, "last_frame": None, "labels": Counter(), "classifications": Counter(), "label_changes": 0, "classification_changes": 0, "last_label": None, "last_classification": None})
    track_frame_indices: dict[int, set[int]] = defaultdict(set)
    previous_boxes_by_label: dict[str, list[tuple[Detection, int]]] = {}
    last_pose_by_track: dict[int, tuple[int, dict[str, tuple[float, float]]]] = {}
    zone_observations: Counter[str] = Counter()
    pose_confidences: list[float] = []
    pose_steps_px: list[float] = []
    errors: list[str] = []
    previous_ids: set[int] = set()
    dropout_events = 0
    candidate_id_switches = 0

    def on_frame(_stream_index: int, result: PipelineFrame) -> None:
        nonlocal dropout_events, candidate_id_switches
        source_frame = source.read(result.metrics.frame_index)
        if source_frame is None:
            return
        annotated = _annotate(source_frame, result)
        writer.write(annotated)
        if len(samples) < args.contact_sheet_samples and result.metrics.frame_index % args.sample_every == 0:
            samples.append((result.metrics.frame_index, annotated.copy()))
        current_ids: set[int] = set()
        current_boxes_by_label: dict[str, list[tuple[Detection, int]]] = defaultdict(list)
        for detection in result.detections:
            if detection.track_id is None:
                continue
            track_id = int(detection.track_id)
            frame_index = result.metrics.frame_index
            current_ids.add(track_id)
            current_boxes_by_label[detection.label].append((detection, track_id))
            track_frame_indices[track_id].add(frame_index)
            stats = track_stats[track_id]
            stats["frames"] += 1
            stats["first_frame"] = frame_index if stats["first_frame"] is None else stats["first_frame"]
            stats["last_frame"] = frame_index
            stats["labels"][detection.label] += 1
            if detection.classification:
                stats["classifications"][detection.classification] += 1
                if (
                    stats["last_classification"] is not None
                    and stats["last_classification"] != detection.classification
                ):
                    stats["classification_changes"] += 1
                stats["last_classification"] = detection.classification
            if stats["last_label"] is not None and stats["last_label"] != detection.label:
                stats["label_changes"] += 1
            stats["last_label"] = detection.label

            center_x = (detection.x1 + detection.x2) / 2.0
            center_y = (detection.y1 + detection.y2) / 2.0
            for zone in result.zones:
                zone_x, zone_y, zone_width, zone_height = zone.roi
                if (
                    zone_x <= center_x / source_frame.shape[1] <= zone_x + zone_width
                    and zone_y <= center_y / source_frame.shape[0] <= zone_y + zone_height
                ):
                    zone_observations[zone.name] += 1
                    break

            if detection.keypoints:
                pose_confidences.extend(point.confidence for point in detection.keypoints)
                keypoint_map = {
                    point.name: (point.x, point.y)
                    for point in detection.keypoints
                }
                box_diagonal = float(np.hypot(detection.width, detection.height))
                previous_pose = last_pose_by_track.get(track_id)
                if previous_pose is not None:
                    _previous_frame, previous_points = previous_pose
                    shared = set(previous_points) & set(keypoint_map)
                    if shared and box_diagonal > 0.0:
                        step = float(
                            np.mean(
                                [
                                    np.hypot(
                                        keypoint_map[name][0] - previous_points[name][0],
                                        keypoint_map[name][1] - previous_points[name][1],
                                    )
                                    for name in shared
                                ]
                            )
                        )
                        pose_steps_px.append(step)
                last_pose_by_track[track_id] = (frame_index, keypoint_map)

        for label, current_entries in current_boxes_by_label.items():
            previous_entries = previous_boxes_by_label.get(label, [])
            for detection, track_id in current_entries:
                for previous_detection, previous_track_id in previous_entries:
                    if previous_track_id == track_id:
                        continue
                    scale = max(1.0, detection.width, detection.height, previous_detection.width, previous_detection.height)
                    distance = np.hypot(
                        (detection.x1 + detection.x2 - previous_detection.x1 - previous_detection.x2) / 2.0,
                        (detection.y1 + detection.y2 - previous_detection.y1 - previous_detection.y2) / 2.0,
                    ) / scale
                    if distance <= 0.35:
                        candidate_id_switches += 1
                        break
        if previous_ids - current_ids:
            dropout_events += len(previous_ids - current_ids)
        previous_boxes_by_label.clear()
        previous_boxes_by_label.update(current_boxes_by_label)
        record = {
            "scenario": result.scenario_id,
            "frame_index": result.metrics.frame_index,
            "captured_at": result.metrics.captured_at,
            "processing_fps": result.metrics.processing_fps,
            "inferences_per_second": result.metrics.inferences_per_second,
            "raw_detections": [_detection_dict(item) for item in result.raw_detections],
            "tracked_detections": [_detection_dict(item) for item in result.detections],
            "events": [event.to_dict() for event in result.events],
            "stages": [asdict(item) for item in result.metrics.stages],
            "placements": {stage: asdict(info) for stage, info in result.placements.items()},
        }
        records.append(record)
        if current_ids - previous_ids and previous_ids:
            record["new_track_ids"] = sorted(current_ids - previous_ids)
        previous_ids.clear()
        previous_ids.update(current_ids)

    def on_failed(_stream_index: int, error: str) -> None:
        errors.append(error)

    worker = ScenarioStreamWorker(
        stream_index=0,
        scenario=scenario,
        catalog=catalog,
        camera_index=None,
        registry=registry,
        assignments=assignments,
        policy_sequence=policy.snapshot().sequence,
        npu_duty_cycle=NpuDutyCycle(),
    )
    worker.frame_ready.connect(on_frame)
    worker.failed.connect(on_failed)
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    QTimer.singleShot(round(args.seconds * 1000.0), worker.requestInterruption)
    started = time.perf_counter()
    worker.start()
    loop.exec()
    worker.wait(10000)
    elapsed = time.perf_counter() - started
    source.close()
    writer.release()
    _contact_sheet(samples, output / "contact-sheet.png")

    short_lived_tracks = sum(
        1
        for track_id, stats in track_stats.items()
        if stats["frames"] <= 3
    )
    summary = {
        "scenario": scenario.id,
        "requested_seconds": args.seconds,
        "elapsed_seconds": elapsed,
        "frames_reviewed": len(records),
        "errors": errors,
        "track_count": len(track_stats),
        "short_lived_tracks": short_lived_tracks,
        "track_fragmentation_rate": short_lived_tracks / max(1, len(track_stats)),
        "new_track_events": sum(1 for record in records if record.get("new_track_ids")),
        "detection_dropout_events": dropout_events,
        "candidate_id_switches": candidate_id_switches,
        "zone_observations": dict(zone_observations),
        "pose_tracks": len(last_pose_by_track),
        "pose_keypoint_observations": len(pose_confidences),
        "pose_mean_keypoint_confidence": float(np.mean(pose_confidences)) if pose_confidences else None,
        "pose_mean_step_px": float(np.mean(pose_steps_px)) if pose_steps_px else None,
        "pose_max_step_px": float(np.max(pose_steps_px)) if pose_steps_px else None,
        "track_stats": {
            str(track_id): {
                "frames": stats["frames"],
                "first_frame": stats["first_frame"],
                "last_frame": stats["last_frame"],
                "lifetime_frames": (stats["last_frame"] - stats["first_frame"] + 1) if stats["first_frame"] is not None and stats["last_frame"] is not None else 0,
                "missing_frames_in_lifetime": (stats["last_frame"] - stats["first_frame"] + 1 - len(track_frame_indices[track_id])) if stats["first_frame"] is not None and stats["last_frame"] is not None else 0,
                "labels": dict(stats["labels"]),
                "classifications": dict(stats["classifications"]),
                "label_changes": stats["label_changes"],
                "classification_changes": stats["classification_changes"],
            }
            for track_id, stats in sorted(track_stats.items())
        },
    }
    with (output / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_default) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=(
            "retail",
            "metro",
            "manufacturing",
            "education",
            "health",
            "federal",
        ),
        required=True,
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--contact-sheet-samples", type=int, default=12)
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

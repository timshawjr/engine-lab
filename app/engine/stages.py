"""CPU decode, preprocess, inference postprocess, and overlay primitives."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import openvino as ov


@dataclass(frozen=True)
class LetterboxTransform:
    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int
    model_width: int
    model_height: int
    scale_x: float | None = None
    scale_y: float | None = None

    @property
    def effective_scale_x(self) -> float:
        return self.scale_x or self.scale

    @property
    def effective_scale_y(self) -> float:
        return self.scale_y or self.scale

    def point_to_source(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.pad_x) / self.effective_scale_x,
            (y - self.pad_y) / self.effective_scale_y,
        )


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    classification: str = ""
    classification_confidence: float = 0.0
    track_id: int | None = None
    keypoints: tuple[Keypoint, ...] = ()

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def with_classification(self, label: str, confidence: float) -> "Detection":
        return Detection(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.label,
            self.confidence,
            label,
            confidence,
            self.track_id,
            self.keypoints,
        )

    def with_track(self, track_id: int) -> "Detection":
        return Detection(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.label,
            self.confidence,
            self.classification,
            self.classification_confidence,
            track_id,
            self.keypoints,
        )

    def with_keypoints(self, keypoints: tuple[Keypoint, ...]) -> "Detection":
        return Detection(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.label,
            self.confidence,
            self.classification,
            self.classification_confidence,
            self.track_id,
            keypoints,
        )


@dataclass(frozen=True)
class PreprocessResult:
    tensor: np.ndarray
    transform: LetterboxTransform
    duration_ms: float


@dataclass(frozen=True)
class ModelInputSpec:
    shape: tuple[int, ...]
    layout: str
    height: int
    width: int


def load_labels(path: Path) -> tuple[str, ...]:
    labels = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not labels:
        raise ValueError(f"label file is empty: {path}")
    return labels


def load_preprocess_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    preprocess = payload.get("preprocess", payload)
    if not isinstance(preprocess, dict):
        raise ValueError(f"invalid preprocess metadata in {path}")
    return preprocess


def _nchw_static_shape(model: ov.Model) -> tuple[int, int]:
    if len(model.inputs) != 1:
        raise ValueError(f"Phase 1 runner requires one model input, found {len(model.inputs)}")
    shape = model.inputs[0].get_partial_shape()
    dimensions = [dimension.get_length() if dimension.is_static else 0 for dimension in shape]
    if len(dimensions) != 4 or dimensions[0] != 1 or dimensions[1] != 3:
        raise ValueError(f"Phase 1 runner requires static NCHW [1,3,H,W], found {dimensions}")
    if dimensions[2] <= 0 or dimensions[3] <= 0:
        raise ValueError(f"dynamic model shape is forbidden: {dimensions}")
    return dimensions[2], dimensions[3]


def model_input_size(model: ov.Model) -> tuple[int, int]:
    """Return H,W from the actual IR."""

    return _nchw_static_shape(model)


def inspect_model_input(model: ov.Model) -> ModelInputSpec:
    if len(model.inputs) != 1:
        raise ValueError(f"runner requires one model input, found {len(model.inputs)}")
    shape = tuple(
        int(dimension.get_length())
        for dimension in model.inputs[0].get_partial_shape()
        if dimension.is_static
    )
    dimensions = [
        int(dimension.get_length()) if dimension.is_static else 0
        for dimension in model.inputs[0].get_partial_shape()
    ]
    if len(shape) != 4 or len(dimensions) != 4 or dimensions[0] != 1:
        raise ValueError(f"static batch-1 image input required, found {dimensions}")
    if dimensions[1] in {1, 3, 4}:
        return ModelInputSpec(dimensions, "NCHW", dimensions[2], dimensions[3])
    if dimensions[3] in {1, 3, 4}:
        return ModelInputSpec(dimensions, "NHWC", dimensions[1], dimensions[2])
    raise ValueError(f"unsupported static image layout: {dimensions}")


def preprocess_omz_image(
    frame: np.ndarray,
    input_spec: ModelInputSpec,
) -> PreprocessResult:
    """Direct resize with the raw BGR 0..255 convention used by these OMZ IRs."""

    started = time.perf_counter()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected BGR frame, found shape {frame.shape}")
    original_height, original_width = frame.shape[:2]
    resized = cv2.resize(
        frame,
        (input_spec.width, input_spec.height),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = resized.astype(np.float32)
    if input_spec.layout == "NCHW":
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    else:
        tensor = tensor[None, ...]
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    return PreprocessResult(
        tensor=tensor,
        transform=LetterboxTransform(
            scale=1.0,
            pad_x=0.0,
            pad_y=0.0,
            original_width=original_width,
            original_height=original_height,
            model_width=input_spec.width,
            model_height=input_spec.height,
            scale_x=input_spec.width / original_width,
            scale_y=input_spec.height / original_height,
        ),
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def preprocess_classification(
    frame: np.ndarray,
    input_spec: ModelInputSpec,
    metadata: dict[str, Any],
) -> PreprocessResult:
    started = time.perf_counter()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected BGR frame, found shape {frame.shape}")
    original_height, original_width = frame.shape[:2]
    resized = cv2.resize(
        frame,
        (input_spec.width, input_spec.height),
        interpolation=cv2.INTER_LINEAR,
    )
    if str(metadata.get("reverse_input_channels", "YES")).upper() in {
        "YES",
        "TRUE",
        "1",
    }:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    mean_values = np.asarray(
        [float(value) for value in str(metadata["mean_values"]).split()],
        dtype=np.float32,
    )
    scale_values = np.asarray(
        [float(value) for value in str(metadata["scale_values"]).split()],
        dtype=np.float32,
    )
    tensor = resized.astype(np.float32)
    tensor = (tensor - mean_values) / scale_values
    if input_spec.layout == "NCHW":
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    else:
        tensor = tensor[None, ...]
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    return PreprocessResult(
        tensor=tensor,
        transform=LetterboxTransform(
            scale=1.0,
            pad_x=0.0,
            pad_y=0.0,
            original_width=original_width,
            original_height=original_height,
            model_width=input_spec.width,
            model_height=input_spec.height,
            scale_x=input_spec.width / original_width,
            scale_y=input_spec.height / original_height,
        ),
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def preprocess_yolo(
    frame: np.ndarray,
    model_width: int,
    model_height: int,
    metadata: dict[str, Any],
) -> PreprocessResult:
    started = time.perf_counter()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected BGR frame, found shape {frame.shape}")
    original_height, original_width = frame.shape[:2]
    scale = min(model_width / original_width, model_height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (model_width - resized_width) / 2.0
    pad_y = (model_height - resized_height) / 2.0
    left = round(pad_x)
    top = round(pad_y)
    canvas = np.full((model_height, model_width, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized

    scale_value = float(metadata.get("scale_values", 255.0))
    if scale_value <= 0:
        raise ValueError(f"invalid scale_values metadata: {scale_value}")
    tensor = canvas.astype(np.float32) / scale_value
    reverse_channels = str(metadata.get("reverse_input_channels", "YES")).upper()
    if reverse_channels in {"YES", "TRUE", "1"}:
        tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    return PreprocessResult(
        tensor=tensor,
        transform=LetterboxTransform(
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            original_width=original_width,
            original_height=original_height,
            model_width=model_width,
            model_height=model_height,
        ),
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0.0)
    intersection_area = intersection[:, 0] * intersection[:, 1]
    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    return intersection_area / np.maximum(box_area + boxes_area - intersection_area, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        overlaps = _box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][overlaps <= threshold]
    return keep


def postprocess_yolo(
    output: np.ndarray,
    labels: Sequence[str],
    transform: LetterboxTransform,
    confidence_threshold: float = 0.30,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> tuple[Detection, ...]:
    predictions = np.asarray(output)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ValueError(f"unexpected YOLO output shape: {predictions.shape}")
    if predictions.shape[0] == 4 + len(labels) or predictions.shape[1] == 4 + len(labels):
        predictions = predictions.T
    if predictions.shape[1] != 4 + len(labels):
        raise ValueError(
            f"YOLO output {predictions.shape} does not match {len(labels)} labels"
        )
    if not 0.0 < confidence_threshold <= 1.0 or not 0.0 < iou_threshold <= 1.0:
        raise ValueError("postprocess thresholds must be in (0, 1]")

    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    candidates = np.flatnonzero(confidences >= confidence_threshold)
    if candidates.size == 0:
        return ()

    center_x = predictions[candidates, 0]
    center_y = predictions[candidates, 1]
    width = predictions[candidates, 2]
    height = predictions[candidates, 3]
    boxes = np.column_stack(
        (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
    )
    candidate_scores = confidences[candidates]
    keep = _nms(boxes, candidate_scores, iou_threshold)[:max_detections]

    detections: list[Detection] = []
    for index in keep:
        x1, y1 = transform.point_to_source(float(boxes[index, 0]), float(boxes[index, 1]))
        x2, y2 = transform.point_to_source(float(boxes[index, 2]), float(boxes[index, 3]))
        x1 = float(np.clip(x1, 0.0, transform.original_width))
        y1 = float(np.clip(y1, 0.0, transform.original_height))
        x2 = float(np.clip(x2, 0.0, transform.original_width))
        y2 = float(np.clip(y2, 0.0, transform.original_height))
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(class_ids[candidates[index]])
        detections.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=labels[class_id],
                confidence=float(candidate_scores[index]),
            )
        )
    return tuple(detections)


def postprocess_ssd(
    output: np.ndarray,
    labels: dict[int, str],
    transform: LetterboxTransform,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> tuple[Detection, ...]:
    predictions = np.asarray(output)
    if predictions.ndim == 4:
        predictions = predictions[0]
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim != 2 or predictions.shape[1] != 7:
        raise ValueError(f"unexpected SSD output shape: {predictions.shape}")
    candidates = predictions[
        (predictions[:, 0] >= 0)
        & (predictions[:, 2] >= confidence_threshold)
    ]
    if candidates.size == 0:
        return ()
    boxes_model = np.column_stack(
        (
            candidates[:, 3],
            candidates[:, 4],
            candidates[:, 5],
            candidates[:, 6],
        )
    )
    boxes_source = np.empty_like(boxes_model)
    for index, box in enumerate(boxes_model):
        x1, y1 = transform.point_to_source(float(box[0]), float(box[1]))
        x2, y2 = transform.point_to_source(float(box[2]), float(box[3]))
        boxes_source[index] = (
            np.clip(x1, 0.0, transform.original_width),
            np.clip(y1, 0.0, transform.original_height),
            np.clip(x2, 0.0, transform.original_width),
            np.clip(y2, 0.0, transform.original_height),
        )
    scores = candidates[:, 2]
    keep = _nms(boxes_source, scores, iou_threshold)[:max_detections]
    detections: list[Detection] = []
    for index in keep:
        row = candidates[index]
        x1, y1, x2, y2 = boxes_source[index]
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(row[1])
        detections.append(
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                label=labels.get(class_id, f"class-{class_id}"),
                confidence=float(row[2]),
            )
        )
    return tuple(detections)


def postprocess_classification(
    output: np.ndarray,
    labels: Sequence[str],
    top_k: int,
) -> tuple[tuple[str, float], ...]:
    logits = np.asarray(output).reshape(-1)
    if logits.size != len(labels):
        raise ValueError(
            f"classification output {logits.size} does not match {len(labels)} labels"
        )
    if not np.all(np.isfinite(logits)):
        raise ValueError("classification output contains non-finite values")
    if float(np.min(logits)) < 0.0 or not np.isclose(
        float(np.sum(logits)), 1.0, atol=1e-3
    ):
        shifted = logits - float(np.max(logits))
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
    else:
        probabilities = logits
    count = min(max(1, top_k), len(labels))
    indices = np.argsort(probabilities)[::-1][:count]
    return tuple((labels[int(index)], float(probabilities[int(index)])) for index in indices)


def decode_pose(
    heatmap_output: np.ndarray,
    paf_output: np.ndarray,
    keypoint_names: Sequence[str],
    transform: LetterboxTransform,
    confidence_threshold: float,
) -> tuple[Keypoint, ...]:
    heatmaps = np.asarray(heatmap_output)
    pafs = np.asarray(paf_output)
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]
    if pafs.ndim == 4:
        pafs = pafs[0]
    if heatmaps.shape[0] < len(keypoint_names):
        raise ValueError(
            f"pose heatmap {heatmaps.shape} has fewer than {len(keypoint_names)} keypoints"
        )
    if pafs.shape[0] < (len(keypoint_names) - 1) * 2:
        raise ValueError(
            f"pose PAF {pafs.shape} does not cover {len(keypoint_names)} keypoints"
        )
    keypoints: list[Keypoint] = []
    for keypoint_index, name in enumerate(keypoint_names):
        heatmap = heatmaps[keypoint_index]
        flat_index = int(np.argmax(heatmap))
        y_index, x_index = np.unravel_index(flat_index, heatmap.shape)
        confidence = float(heatmap[y_index, x_index])
        model_x = (x_index + 0.5) * (transform.model_width / heatmap.shape[1])
        model_y = (y_index + 0.5) * (transform.model_height / heatmap.shape[0])
        source_x, source_y = transform.point_to_source(model_x, model_y)
        keypoints.append(
            Keypoint(
                name=name,
                x=float(
                    np.clip(source_x, 0.0, transform.original_width)
                ),
                y=float(
                    np.clip(source_y, 0.0, transform.original_height)
                ),
                confidence=confidence,
            )
        )
    return tuple(keypoints)


def crop_detection(frame: np.ndarray, detection: Detection) -> np.ndarray:
    x1 = max(0, min(frame.shape[1] - 1, int(np.floor(detection.x1))))
    y1 = max(0, min(frame.shape[0] - 1, int(np.floor(detection.y1))))
    x2 = max(x1 + 1, min(frame.shape[1], int(np.ceil(detection.x2))))
    y2 = max(y1 + 1, min(frame.shape[0], int(np.ceil(detection.y2))))
    return frame[y1:y2, x1:x2]

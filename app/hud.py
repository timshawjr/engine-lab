"""Phase 3 Qt booth HUD: scenarios, events, operator overlay, and attract mode."""

from __future__ import annotations

import gc
import json
import logging
import time
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QApplication,
    QLayout,
)

from app.engine.availability import AvailabilityMatrix
from app.engine.device_policy import DENSITIES, DeviceMode, DevicePolicy, PolicySnapshot
from app.engine.events import BusinessEvent
from app.engine.pipelines import (
    PipelineFrame,
    PipelineFrameMetrics,
    ScenarioCatalog,
    ScenarioConfig,
    ScenarioModelRegistry,
    ScenarioStreamWorker,
    StageAssignment,
    StageMetric,
    build_stage_assignments,
)
from app.engine.runner import RetailVideoClock, RunnerInfo
from app.engine.stages import Detection, Keypoint
from app.layout import use_compact_layout
from app.overlay import source_to_display_rect
from app import hangwatch
from app.telemetry.devices_win import (
    enumerate_compute_accelerators,
    enumerate_display_adapters,
    is_intel,
    is_npu,
)
from app.telemetry.npu_fallback import NpuDutyCycle
from app.telemetry.sampler import EngineMetric, TelemetryFrame, TelemetrySampler
from app.theme import THEME


LOGGER = logging.getLogger("engine_lab")
SCENARIO_ORDER = ("retail", "smart_city", "medical", "gov_defense")
POSE_SKELETON = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
    (2, 16),
    (5, 17),
)


def _label(
    text: str,
    *,
    size: int = THEME.font_regular,
    color: str = THEME.text,
    bold: bool = False,
) -> QLabel:
    label = QLabel(text)
    font = QFont(THEME.font_family, size)
    font.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
    label.setFont(font)
    label.setStyleSheet(f"color: {color}; background: transparent;")
    label.setWordWrap(True)
    return label


def _panel() -> QFrame:
    panel = QFrame()
    panel.setStyleSheet(
        f"background: {THEME.panel}; border: 1px solid {THEME.border}; "
        f"border-radius: {THEME.radius_panel}px;"
    )
    return panel


def cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
    except OSError:
        return "Intel CPU"


def _hardware_description() -> dict[str, Any]:
    gpu_name = "Intel GPU"
    gpu_driver = "unknown"
    try:
        device = next(
            (item for item in enumerate_display_adapters() if is_intel(item)),
            None,
        )
        if device is not None:
            gpu_name = device.friendly_name or device.description
            gpu_driver = device.driver_version or "unknown"
    except Exception:
        pass
    npu_name = "Intel NPU"
    npu_driver = "unknown"
    try:
        device = next(
            (
                item
                for item in enumerate_compute_accelerators()
                if is_intel(item) and is_npu(item)
            ),
            None,
        )
        if device is not None:
            npu_name = device.friendly_name or device.description
            npu_driver = device.driver_version or "unknown"
    except Exception:
        pass
    return {
        "cpu": cpu_name(),
        "gpu": gpu_name,
        "gpu_driver": gpu_driver,
        "npu": npu_name,
        "npu_driver": npu_driver,
        "physical_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cores": psutil.cpu_count(logical=True) or 0,
    }


def _platform_peak(
    cpu: str,
    profiles_path: Path,
) -> tuple[str, str]:
    try:
        payload = json.loads(profiles_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Peak TOPS unavailable", ""
    lowered = cpu.lower()
    for profile in payload.get("profiles", []):
        if any(str(match).lower() in lowered for match in profile.get("match", [])):
            peaks = profile.get("peak_tops", {})
            values = []
            for engine in ("npu", "gpu", "platform"):
                value = peaks.get(engine)
                if value is not None:
                    values.append(f"{engine.upper()} peak {value} TOPS")
            if not values:
                return "Peak TOPS unavailable", str(profile.get("source", ""))
            return " · ".join(values), str(profile.get("source", ""))
    return "Peak TOPS unavailable", ""


class FrameCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._detections: tuple[Detection, ...] = ()
        self._zones: tuple[Any, ...] = ()
        self._source_size = (0, 0)
        self._event_text = ""
        self.placeholder = "Starting measured pipeline…"
        self.header_text = "Retail"
        self.show_header = True
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(
        self,
        image: QImage,
        detections: tuple[Detection, ...],
        zones: tuple[Any, ...],
        source_size: tuple[int, int],
        event_text: str,
    ) -> None:
        self._image = image
        self._detections = detections
        self._zones = zones
        self._source_size = source_size
        self._event_text = event_text
        self.update()

    def _video_rect(self) -> QRectF:
        if self._image is None or self._image.isNull():
            return QRectF()
        scale = min(self.width() / self._image.width(), self.height() / self._image.height())
        width = self._image.width() * scale
        height = self._image.height() * scale
        return QRectF(
            (self.width() - width) / 2.0,
            (self.height() - height) / 2.0,
            width,
            height,
        )

    def _draw_zone(self, painter: QPainter, video_rect: QRectF, zone: Any) -> None:
        x, y, width, height = zone.roi
        rect = QRectF(
            video_rect.left() + x * video_rect.width(),
            video_rect.top() + y * video_rect.height(),
            width * video_rect.width(),
            height * video_rect.height(),
        )
        pen = QPen(QColor(THEME.gpu), THEME.overlay_border)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        painter.fillRect(
            QRectF(rect.left(), rect.top(), rect.width(), THEME.overlay_label_height),
            QColor(THEME.overlay_fill),
        )
        painter.setPen(QPen(QColor(THEME.overlay_text)))
        painter.setFont(QFont(THEME.font_family, THEME.font_regular))
        painter.drawText(
            QRectF(rect.left(), rect.top(), rect.width(), THEME.overlay_label_height),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            zone.name,
        )

    def _draw_pose(
        self,
        painter: QPainter,
        video_rect: QRectF,
        keypoints: tuple[Keypoint, ...],
    ) -> None:
        if not keypoints:
            return
        by_name = {keypoint.name: keypoint for keypoint in keypoints}
        names = (
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "neck",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        )
        painter.setPen(QPen(QColor(THEME.cpu), THEME.overlay_border))
        for start_name, end_name in POSE_SKELETON:
            if start_name >= len(names) or end_name >= len(names):
                continue
            start = by_name.get(names[start_name])
            end = by_name.get(names[end_name])
            if start is None or end is None:
                continue
            start_point = QPointF(
                video_rect.left() + start.x / self._source_size[0] * video_rect.width(),
                video_rect.top() + start.y / self._source_size[1] * video_rect.height(),
            )
            end_point = QPointF(
                video_rect.left() + end.x / self._source_size[0] * video_rect.width(),
                video_rect.top() + end.y / self._source_size[1] * video_rect.height(),
            )
            painter.drawLine(start_point, end_point)
        painter.setBrush(QColor(THEME.npu))
        painter.setPen(QPen(QColor(THEME.overlay_text), 1))
        for keypoint in keypoints:
            point = QRectF(
                video_rect.left() + keypoint.x / self._source_size[0] * video_rect.width() - 4,
                video_rect.top() + keypoint.y / self._source_size[1] * video_rect.height() - 4,
                8,
                8,
            )
            painter.drawEllipse(point)

    def _draw_detections(
        self,
        painter: QPainter,
        video_rect: QRectF,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(THEME.font_family, THEME.font_overlay)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        # Labels are drawn above their box, but boxes near the top edge have no
        # room, so every such label lands on the same row and they overprint
        # each other into an unreadable smear. Track the rows already taken and
        # push a colliding label below them instead.
        occupied_rows: list[tuple[float, float]] = []
        for detection in self._detections:
            rect = source_to_display_rect(
                detection,
                video_rect,
                self._source_size[0],
                self._source_size[1],
            )
            if rect.isEmpty():
                continue
            painter.setPen(QPen(QColor(THEME.npu), THEME.overlay_border))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
            parts = [f"{detection.label} {detection.confidence:.0%}"]
            if detection.classification:
                # The label is a member of the scenario's declared vocabulary,
                # not a free-form guess, so it is shown as the product it is.
                parts.append(f"{detection.classification} {detection.classification_confidence:.0%}")
            if detection.track_id is not None:
                parts.append(f"#{detection.track_id}")
            label = " · ".join(parts)
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(label) + THEME.spacing_sm * 2
            top = rect.top() - THEME.overlay_label_height
            if top < video_rect.top():
                # No room above the box: sit just inside it instead.
                top = rect.top()
            left = rect.left()
            # Nudge down until this label does not overlap an earlier one.
            for _ in range(len(occupied_rows) + 1):
                if not any(
                    top < row_top + THEME.overlay_label_height
                    and row_top < top + THEME.overlay_label_height
                    and left < row_left + row_width
                    and row_left < left + text_width
                    for row_top, row_left, row_width in occupied_rows
                ):
                    break
                top += THEME.overlay_label_height
            occupied_rows.append((top, left, text_width))
            label_rect = QRectF(
                rect.left(),
                top,
                text_width,
                THEME.overlay_label_height,
            )
            painter.fillRect(label_rect, QColor(THEME.overlay_fill))
            painter.setPen(QPen(QColor(THEME.overlay_text)))
            painter.drawText(
                label_rect.adjusted(THEME.spacing_sm, 0, -THEME.spacing_sm, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
        for detection in self._detections:
            self._draw_pose(painter, video_rect, detection.keypoints)
        painter.restore()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#03080D"))
        if self._image is None or self._image.isNull():
            painter.setPen(QPen(QColor(THEME.text_muted)))
            painter.setFont(QFont(THEME.font_family, THEME.font_title))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            return
        video_rect = self._video_rect()
        painter.drawImage(video_rect, self._image)
        for zone in self._zones:
            self._draw_zone(painter, video_rect, zone)
        self._draw_detections(painter, video_rect)
        if self.show_header:
            badge = f"{self.header_text} · {len(self._detections)} detection(s)"
            painter.fillRect(
                QRectF(
                    video_rect.left(),
                    video_rect.top(),
                    THEME.overlay_badge_width,
                    THEME.overlay_label_height,
                ),
                QColor(THEME.overlay_fill),
            )
            painter.setPen(QPen(QColor(THEME.overlay_text)))
            painter.setFont(QFont(THEME.font_family, THEME.font_overlay))
            painter.drawText(
                QRectF(
                    video_rect.left(),
                    video_rect.top(),
                    THEME.overlay_badge_width,
                    THEME.overlay_label_height,
                ),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                badge,
            )
        if self._event_text:
            ticker_rect = QRectF(
                video_rect.left(),
                video_rect.bottom() - THEME.overlay_label_height,
                video_rect.width(),
                THEME.overlay_label_height,
            )
            painter.fillRect(ticker_rect, QColor(THEME.overlay_fill))
            painter.setPen(QPen(QColor(THEME.overlay_text)))
            painter.setFont(QFont(THEME.font_family, THEME.font_regular))
            painter.drawText(
                ticker_rect.adjusted(THEME.spacing_sm, 0, -THEME.spacing_sm, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._event_text,
            )


class StreamTile(QFrame):
    def __init__(
        self,
        stream_index: int,
        parent: QWidget | None = None,
        *,
        compact: bool = False,
    ) -> None:
        super().__init__(parent)
        self.stream_index = stream_index
        self.compact = compact
        self.setStyleSheet(
            f"background: {THEME.panel_alt}; border: 1px solid {THEME.border}; "
            f"border-radius: {THEME.radius_small}px;"
        )
        self.setMinimumSize(260, 120 if compact else 110)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            THEME.spacing_xs,
            THEME.spacing_xs,
            THEME.spacing_xs,
            THEME.spacing_xs,
        )
        self.badge = _label(
            f"STREAM {stream_index} · starting",
            size=14 if compact else 18,
            bold=True,
        )
        self.canvas = FrameCanvas()
        self.canvas.setMinimumSize(80, 70 if compact else 45)
        self.canvas.show_header = False
        self.canvas.placeholder = "starting"
        self._thumbnail_visible = True
        self.status = _label(
            "LIVE · main video view",
            size=14 if compact else THEME.font_regular,
            color=THEME.text_muted,
        )
        self.metrics = _label(
            "FPS — · inference — ms",
            size=14 if compact else THEME.font_regular,
            color=THEME.text_muted,
        )
        layout.addWidget(self.badge)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.status)
        layout.addWidget(self.metrics)

    def set_thumbnail_visible(self, visible: bool) -> None:
        self._thumbnail_visible = visible
        self.canvas.setVisible(visible)
        self.status.setVisible(not visible)

    def update_stream(
        self,
        image: QImage,
        detections: tuple[Detection, ...],
        zones: tuple[Any, ...],
        source_size: tuple[int, int],
        placements: dict[str, RunnerInfo],
        model_ids: dict[str, str],
        metrics: PipelineFrameMetrics | None,
        event_text: str,
    ) -> None:
        if placements:
            stage_names = {
                "detector": "Detector",
                "classifier": "Classifier",
                "person_detector": "Person",
                "pose": "Pose",
                "perimeter_detector": "Perimeter",
                "plate_detector": "Plate",
            }
            placement_parts = [
                f"{stage_names.get(stage, stage)}: "
                f"{','.join(info.execution_devices) or 'pending'}"
                for stage, info in placements.items()
            ]
            placement_parts.append("Events: CPU")
            self.badge.setText(
                f"STREAM {self.stream_index} · " + " · ".join(placement_parts)
            )
            tooltip = "\n".join(
                f"{stage}: requested {info.requested_device}; "
                f"EXECUTION_DEVICES={','.join(info.execution_devices)}; "
                f"hint={info.performance_hint}"
                for stage, info in placements.items()
            )
            self.setToolTip(tooltip + "\nCPU: logical event processing")
        if metrics is not None:
            self.metrics.setText(
                f"infer {metrics.inference_ms:.2f} ms · "
                f"{metrics.detection_count} det"
            )
        self.status.setText(
            _stream_activity_text(
                placements,
                model_ids,
                len(detections),
            )
        )
        if self._thumbnail_visible:
            self.canvas.set_frame(
                image,
                detections,
                zones,
                source_size,
                "" if self.compact else event_text,
            )


class EngineGauge(QWidget):
    def __init__(
        self,
        engine: str,
        parent: QWidget | None = None,
        *,
        compact: bool = False,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.compact = compact
        self.engine_metric: EngineMetric | None = None
        self.disabled = False
        self._history: deque[tuple[float, float]] = deque()
        self.setMinimumHeight(115 if compact else 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_metric(self, metric: EngineMetric) -> None:
        if self.disabled:
            return
        self.engine_metric = metric
        if metric.value_percent is not None:
            self._history.append((time.time(), metric.value_percent))
        cutoff = time.time() - THEME.sparkline_window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        self.setToolTip(f"{metric.source}\n{metric.detail}")
        self.update()

    def set_disabled(self, disabled: bool) -> None:
        self.disabled = disabled
        self.update()

    def _draw_sparkline(
        self,
        painter: QPainter,
        rect: QRectF,
        accent: str,
    ) -> None:
        painter.fillRect(rect, QColor(THEME.border))
        if len(self._history) < 2:
            return
        now = time.time()
        points: list[QPointF] = []
        for timestamp, value in self._history:
            x = rect.left() + (timestamp - (now - THEME.sparkline_window_seconds)) / THEME.sparkline_window_seconds * rect.width()
            x = max(rect.left(), min(rect.right(), x))
            y = rect.bottom() - max(0.0, min(THEME.gauge_max_percent, value)) / THEME.gauge_max_percent * rect.height()
            points.append(QPointF(x, y))
        painter.setPen(QPen(QColor(accent), 2))
        painter.drawPolyline(points)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        normal_accent = {"NPU": THEME.npu, "GPU": THEME.gpu, "CPU": THEME.cpu}[self.engine]
        accent = THEME.text_muted if self.disabled else normal_accent
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(THEME.border), 1))
        painter.setBrush(QColor(THEME.panel_alt))
        painter.drawRoundedRect(rect, THEME.radius_panel, THEME.radius_panel)
        x = THEME.spacing_lg
        y = THEME.spacing_xs
        width = self.width() - THEME.spacing_lg * 2
        source = None if self.engine_metric is None else self.engine_metric.source
        provider = "—"
        if self.disabled:
            provider = "OFF"
        elif source is not None:
            if "app-measured" in source:
                provider = "APP"
            elif "PDH" in source:
                provider = "PDH"
            elif "psutil" in source:
                provider = "psutil"
        painter.setFont(QFont(THEME.font_family, 16 if self.compact else THEME.font_semibold, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(accent)))
        painter.drawText(
            QRectF(x, y, width * 0.5, 28),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{self.engine} · {provider}",
        )
        state = "OFF BY OPERATOR" if self.disabled else (
            "WAITING" if self.engine_metric is None else self.engine_metric.state
        )
        painter.setFont(QFont(THEME.font_family, 14 if self.compact else THEME.font_badge, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(x + width * 0.35, y, width * 0.65, 28),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            state,
        )
        value = None if self.engine_metric is None else self.engine_metric.value_percent
        value_text = "—" if value is None else f"{value:.0f}%"
        value_font = QFont(THEME.font_family, 40 if self.compact else 56, QFont.Weight.Bold)
        painter.setFont(value_font)
        painter.setPen(QPen(QColor(THEME.text_muted if self.disabled else THEME.text)))
        value_height = max(56 if self.compact else 72, painter.fontMetrics().height() + 10)
        value_rect = QRectF(
            x,
            y + (16 if self.compact else 20),
            width * 0.62,
            value_height,
        )
        painter.drawText(
            value_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            value_text,
        )
        spark_rect = QRectF(
            x + width * 0.64,
            y + (24 if self.compact else 28),
            width * 0.36,
            60 if self.compact else 70,
        )
        self._draw_sparkline(painter, spark_rect, accent)
        bar = QRectF(
            x,
            value_rect.bottom() + 8 if self.compact else y + 124,
            width,
            12 if self.compact else THEME.gauge_height,
        )
        painter.fillRect(bar, QColor(THEME.border))
        if value is not None:
            fraction = max(0.0, min(1.0, value / THEME.gauge_max_percent))
            painter.fillRect(
                QRectF(bar.left(), bar.top(), bar.width() * fraction, bar.height()),
                QColor(accent),
            )


class MetricTile(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"background: {THEME.panel_alt}; border: 1px solid {THEME.border}; "
            f"border-radius: {THEME.radius_small}px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            THEME.spacing_sm,
            THEME.spacing_xs,
            THEME.spacing_sm,
            THEME.spacing_xs,
        )
        layout.setSpacing(0)
        layout.addWidget(_label(title, color=THEME.text_muted, bold=True))
        self.value = _label("—", size=THEME.font_metric_compact, bold=True)
        layout.addWidget(self.value)

    def set_value(self, value: str) -> None:
        self.value.setText(value)


def _short_model_name(model_id: str, stage: str = "") -> str:
    if model_id == "cpu":
        return {
            "zone_event": "zone/events",
            "posture_event": "posture/events",
            "track_event": "track/events",
        }.get(stage, "events")
    replacements = (
        ("yolo11n", "YOLO11n"),
        ("efficientnet-b0", "EfficientNet-B0"),
        ("person-detection-retail-0013", "person-detector"),
        ("vehicle-license-plate-detection-barrier-0106", "plate-detector"),
        ("human-pose-estimation-0001", "pose"),
        ("person-vehicle-bike-detection-crossroad-1016", "traffic-detector"),
    )
    for source, replacement in replacements:
        if source in model_id:
            return replacement
    return model_id.replace("-fp16", "").replace("-int8", "")


def _stream_activity_text(
    placements: dict[str, RunnerInfo],
    model_ids: dict[str, str],
    detection_count: int,
) -> str:
    if not placements:
        return "waiting for pipeline…"
    verbs = {
        "detector": "detect",
        "classifier": "classify",
        "person_detector": "detect",
        "pose": "estimate",
        "perimeter_detector": "detect",
        "plate_detector": "detect",
    }
    parts = [
        f"{_short_model_name(model_ids.get(stage, stage))}: {verbs.get(stage, 'process')}"
        for stage in placements
    ]
    parts.append("CPU: event logic")
    parts.append(f"{detection_count} detections")
    return " · ".join(parts) if parts else "waiting for pipeline…"


class StageBreakdown(QFrame):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        compact: bool = False,
    ) -> None:
        super().__init__(parent)
        self.compact = compact
        self._metrics: tuple[StageMetric, ...] = ()
        self.setMinimumHeight(60 if compact else 70)
        self.setStyleSheet(
            f"background: {THEME.panel_alt}; border: 1px solid {THEME.border}; "
            f"border-radius: {THEME.radius_small}px;"
        )

    def set_metrics(self, metrics: tuple[StageMetric, ...]) -> None:
        self._metrics = metrics
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont(THEME.font_family, 14 if self.compact else THEME.font_regular))
        painter.setPen(QPen(QColor(THEME.text_muted)))
        painter.drawText(
            QRectF(THEME.spacing_sm, 2, self.width() * 0.55, 26),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "PIPELINE PLACEMENT · duration share",
        )
        cpu_metric = next(
            (metric for metric in self._metrics if metric.model_id == "cpu"),
            None,
        )
        if cpu_metric is not None:
            painter.setFont(
                QFont(
                    THEME.font_family,
                    10 if self.compact else 12,
                    QFont.Weight.DemiBold,
                )
            )
            painter.setPen(QPen(QColor(THEME.cpu)))
            painter.drawText(
                QRectF(self.width() * 0.55, 2, self.width() * 0.43 - 10, 26),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"CPU workload: {_short_model_name(cpu_metric.model_id, cpu_metric.stage)}",
            )
        durations = [metric.inference_ms + metric.postprocess_ms for metric in self._metrics]
        total = sum(durations)
        bar = QRectF(
            THEME.spacing_sm,
            26 if self.compact else 30,
            self.width() - 20,
            22 if self.compact else THEME.gauge_height,
        )
        painter.fillRect(bar, QColor(THEME.border))
        if total > 0.0 and self._metrics:
            cursor = bar.left()
            colors = (THEME.npu, THEME.gpu, THEME.cpu, THEME.warning)
            for index, (metric, duration) in enumerate(zip(self._metrics, durations)):
                width = bar.width() * duration / total
                segment = QRectF(cursor, bar.top(), width, bar.height())
                painter.fillRect(segment, QColor(colors[index % len(colors)]))
                if width >= 60:
                    device = metric.execution_devices[0] if metric.execution_devices else "pending"
                    label = f"{_short_model_name(metric.model_id, metric.stage)} → {device}"
                    painter.setFont(
                        QFont(
                            THEME.font_family,
                            10 if self.compact else 12,
                            QFont.Weight.DemiBold,
                        )
                    )
                    painter.setPen(QPen(QColor(THEME.overlay_fill)))
                    painter.drawText(
                        segment,
                        Qt.AlignmentFlag.AlignCenter,
                        painter.fontMetrics().elidedText(
                            label,
                            Qt.TextElideMode.ElideRight,
                            int(segment.width() - 8),
                        ),
                    )
                cursor += width


class OperatorOverlay(QDialog):
    def __init__(
        self,
        *,
        availability: AvailabilityMatrix,
        catalog: ScenarioCatalog,
        scenario: ScenarioConfig,
        registry: ScenarioModelRegistry,
        telemetry_map_path: Path,
        profiles_path: Path,
        cache_dir: Path,
        placements: dict[str, dict[str, RunnerInfo]],
        events: tuple[BusinessEvent, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Engine Lab — Operator overlay")
        self.resize(1400, 850)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        availability_table = QTableWidget(
            sum(len(model.devices) for model in availability.models.values()),
            6,
        )
        availability_table.setHorizontalHeaderLabels(
            ["Model", "Device", "Compile", "EXECUTION_DEVICES", "Error", "Source"]
        )
        row = 0
        for model_id, model in availability.models.items():
            bundle = registry.bundle(model_id)
            for device, result in model.devices.items():
                values = (
                    model_id,
                    device,
                    "PASS" if result.success else "FAIL",
                    ",".join(result.execution_devices),
                    result.error or "",
                    " · ".join(bundle.source_urls),
                )
                for column, value in enumerate(values):
                    availability_table.setItem(row, column, QTableWidgetItem(value))
                row += 1
        availability_table.resizeColumnsToContents()
        tabs.addTab(availability_table, "Models × devices")

        placement_table = QTableWidget(0, 6)
        placement_table.setHorizontalHeaderLabels(
            ["Stream", "Stage", "Model", "Requested", "EXECUTION_DEVICES", "Hint"]
        )
        stage_models = {stage.stage: stage.model_id for stage in scenario.stages}
        for stream_index, stages in sorted(placements.items()):
            for stage, info in sorted(stages.items()):
                row = placement_table.rowCount()
                placement_table.insertRow(row)
                values = (
                    str(stream_index),
                    stage,
                    stage_models.get(stage, "logical"),
                    info.requested_device,
                    ",".join(info.execution_devices),
                    info.performance_hint,
                )
                for column, value in enumerate(values):
                    placement_table.setItem(row, column, QTableWidgetItem(value))
        placement_table.resizeColumnsToContents()
        tabs.addTab(placement_table, "Live placement")

        telemetry = QPlainTextEdit()
        telemetry.setReadOnly(True)
        telemetry.setPlainText(telemetry_map_path.read_text(encoding="utf-8"))
        tabs.addTab(telemetry, "Raw telemetry map")

        hardware = _hardware_description()
        profile_text, profile_source = _platform_peak(hardware["cpu"], profiles_path)
        system_text = json.dumps(
            {
                **hardware,
                "profile_peak": profile_text,
                "profile_source": profile_source,
                "cache_dir": str(cache_dir),
                "scenario_ids": [scenario.id for scenario in catalog.values()],
            },
            indent=2,
        )
        system_view = QPlainTextEdit()
        system_view.setReadOnly(True)
        system_view.setPlainText(system_text)
        tabs.addTab(system_view, "System")

        event_view = QPlainTextEdit()
        event_view.setReadOnly(True)
        event_view.setPlainText(
            "\n".join(
                json.dumps(event.to_dict(), sort_keys=True)
                for event in events[-THEME.operator_event_lines:]
            )
        )
        tabs.addTab(event_view, "Events")

        log_path = max(
            cache_dir.parent.joinpath("logs").glob("session-*.log"),
            key=lambda path: path.stat().st_mtime,
            default=None,
        )
        log_view = QPlainTextEdit()
        log_view.setReadOnly(True)
        if log_path is not None:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            log_view.setPlainText("\n".join(lines[-THEME.operator_log_lines:]))
        tabs.addTab(log_view, f"Last {THEME.operator_log_lines} log lines")

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        catalog: ScenarioCatalog,
        registry: ScenarioModelRegistry,
        initial_scenario: str,
        camera_index: int | None,
        cache_dir: Path,
        telemetry_map_path: Path,
        profiles_path: Path,
        availability: AvailabilityMatrix,
        prewarm_fallbacks: list[str],
        fullscreen: bool,
        initial_mode: DeviceMode | None = None,
        initial_density: int = 1,
        npu_enabled: bool = True,
        gpu_enabled: bool = True,
        attract_mode: bool = False,
        startup_origin_perf: float | None = None,
    ) -> None:
        super().__init__()
        self.catalog = catalog
        self.registry = registry
        self.scenario = catalog[initial_scenario]
        self.camera_index = camera_index
        self.cache_dir = cache_dir
        self.telemetry_map_path = telemetry_map_path
        self.profiles_path = profiles_path
        self.availability = availability
        self.prewarm_fallbacks = tuple(prewarm_fallbacks)
        self.started_at = time.time()
        self.started_perf = (
            time.perf_counter() if startup_origin_perf is None else startup_origin_perf
        )
        self._shutdown = False
        self.attract_mode = attract_mode
        self.npu_duty_cycle = NpuDutyCycle()
        scenario_model_ids = {
            stage.model_id
            for scenario in catalog.values()
            for stage in scenario.inference_stages
        }
        available_scenario_devices = tuple(
            device
            for device in availability.available_devices
            if any(
                availability.supports(model_id, device)
                for model_id in scenario_model_ids
            )
        )
        self.policy = DevicePolicy(
            available_devices=available_scenario_devices,
            mode=initial_mode or DeviceMode.SPREAD,
            density=initial_density,
        )
        if not npu_enabled:
            self.policy.toggle_npu()
        if not gpu_enabled:
            self.policy.toggle_gpu()
        self.policy_snapshot = self.policy.snapshot()
        self.runtime_sequence = self.policy_snapshot.sequence
        self.video_clock = RetailVideoClock(
            catalog.media_path(self.scenario.id, camera_index),
            camera_index,
            parent=self,
        )
        self.telemetry_sampler = TelemetrySampler(
            telemetry_map_path,
            self.npu_duty_cycle,
            interval_seconds=THEME.telemetry_interval_seconds,
            parent=self,
        )
        self.stream_workers: dict[int, ScenarioStreamWorker] = {}
        self.stream_placements: dict[int, dict[str, RunnerInfo]] = {}
        self.stream_detections: dict[int, tuple[Detection, ...]] = {}
        self.stream_metrics: dict[int, PipelineFrameMetrics] = {}
        self.stream_event_text: dict[int, str] = {}
        self.frame_history: deque[PipelineFrameMetrics] = deque(maxlen=6000)
        self.telemetry_history: deque[TelemetryFrame] = deque(maxlen=5000)
        self.rss_timeline: deque[dict[str, float]] = deque(maxlen=200)
        self.events: deque[BusinessEvent] = deque(maxlen=1000)
        self.latest_telemetry: TelemetryFrame | None = None
        self.latest_image: QImage | None = None
        self.latest_image_size = (0, 0)
        self.pending_acks: dict[int, set[int]] = {}
        self.policy_requested_at: dict[int, float] = {}
        self.policy_requested_monotonic: dict[int, float] = {}
        self.policy_transitions: list[dict[str, Any]] = []
        self.scenario_history: list[dict[str, Any]] = []
        self.fallback_history: set[str] = set(self.prewarm_fallbacks)
        self.stream_fallbacks: dict[int, set[str]] = {}
        self.fallbacks: set[str] = set(self.prewarm_fallbacks)
        self.last_error: str | None = None
        self.ticker_index = 0
        self.attract_index = SCENARIO_ORDER.index(self.scenario.id)
        self._last_rss_sample = 0.0
        self._build_ui()
        self.video_clock.frame_ready.connect(self._on_video_frame)
        self.video_clock.failed.connect(self._on_video_failed)
        self.telemetry_sampler.frame_ready.connect(self._on_telemetry)
        self.ticker_timer = QTimer(self)
        self.ticker_timer.setInterval(THEME.ticker_seconds * 1000)
        self.ticker_timer.timeout.connect(self._rotate_ticker)
        self.ticker_timer.start()
        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(THEME.clock_interval_ms)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start()
        self.attract_timer = QTimer(self)
        self.attract_timer.setInterval(THEME.attract_scenario_seconds * 1000)
        self.attract_timer.timeout.connect(self._cycle_attract_scenario)
        if self.attract_mode:
            self.attract_timer.start()
        if fullscreen:
            self.showFullScreen()
        else:
            self.show()
        self.startup_elapsed_s = time.perf_counter() - self.started_perf

    def _build_ui(self) -> None:
        self.setWindowTitle("Engine Lab — Phase 3 multi-scenario booth")
        self.setMinimumSize(THEME.minimum_window_width, THEME.minimum_window_height)
        self.resize(THEME.design_width, THEME.design_height)
        self.setStyleSheet(f"background: {THEME.background}; color: {THEME.text};")
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        screen = QApplication.primaryScreen()
        available_width = screen.availableGeometry().width() if screen is not None else THEME.design_width
        available_height = screen.availableGeometry().height() if screen is not None else THEME.design_height
        self.compact_layout = use_compact_layout(available_width, available_height)
        self.demo_page = QWidget()
        self.attract_page = QWidget()
        self.stack.addWidget(self.demo_page)
        self.stack.addWidget(self.attract_page)
        self._build_demo_page()
        self._build_attract_page()
        self.stack.setCurrentIndex(1 if self.attract_mode else 0)

    def _build_demo_page(self) -> None:
        root = QVBoxLayout(self.demo_page)
        root.setContentsMargins(
            THEME.spacing_sm if self.compact_layout else THEME.spacing_md,
            THEME.spacing_sm if self.compact_layout else THEME.spacing_md,
            THEME.spacing_sm if self.compact_layout else THEME.spacing_md,
            THEME.spacing_sm if self.compact_layout else THEME.spacing_md,
        )
        root.setSpacing(THEME.spacing_xs if self.compact_layout else THEME.spacing_sm)
        hardware = _hardware_description()
        peak_text, peak_source = _platform_peak(hardware["cpu"], self.profiles_path)
        header = QHBoxLayout()
        platform_panel = _panel()
        platform_layout = QVBoxLayout(platform_panel)
        platform_layout.setContentsMargins(
            THEME.spacing_md,
            THEME.spacing_sm,
            THEME.spacing_md,
            THEME.spacing_sm,
        )
        platform_layout.addWidget(
            _label(
                hardware["cpu"],
                size=14 if self.compact_layout else THEME.font_regular,
                bold=True,
            )
        )
        platform_layout.addWidget(
            _label(
                f"iGPU · {hardware['gpu']} · {hardware['gpu_driver']}",
                size=14 if self.compact_layout else THEME.font_regular,
                color=THEME.text_muted,
            )
        )
        platform_layout.addWidget(
            _label(
                f"NPU · {hardware['npu']} · {hardware['npu_driver']} · "
                f"{hardware['physical_cores']}P/{hardware['logical_cores']}L",
                size=14 if self.compact_layout else THEME.font_regular,
                color=THEME.text_muted,
            )
        )
        peak_label = _label(
            peak_text,
            size=14 if self.compact_layout else THEME.font_regular,
            color=THEME.text_muted,
        )
        peak_label.setToolTip(peak_source)
        self.peak_base_text = peak_text
        self.peak_label = peak_label
        platform_layout.addWidget(peak_label)
        header.addWidget(platform_panel, 3)

        title_panel = _panel()
        title_layout = QVBoxLayout(title_panel)
        title_layout.setContentsMargins(
            THEME.spacing_md,
            THEME.spacing_sm,
            THEME.spacing_md,
            THEME.spacing_sm,
        )
        self.title_label = _label(
            self.scenario.title,
            size=28 if self.compact_layout else THEME.font_title,
            bold=True,
        )
        self.business_label = _label(
            self.scenario.business_line,
            size=16 if self.compact_layout else THEME.font_regular,
            color=THEME.text_muted,
        )
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.business_label)
        header.addWidget(title_panel, 5)

        status_panel = _panel()
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(
            THEME.spacing_md,
            THEME.spacing_sm,
            THEME.spacing_md,
            THEME.spacing_sm,
        )
        self.mode_label = _label(
            self.policy_snapshot.label,
            size=THEME.font_semibold,
            color=THEME.npu,
            bold=True,
        )
        status_row = QHBoxLayout()
        self.live_badge = _label("ATTRACT" if self.attract_mode else "LIVE", bold=True)
        self.clock_label = _label("", color=THEME.text_muted)
        status_row.addWidget(self.live_badge)
        status_row.addStretch(1)
        status_row.addWidget(self.clock_label)
        self.fallback_label = _label("FALLBACK", color=THEME.danger, bold=True)
        self.fallback_label.hide()
        self.failover_label = _label("", color=THEME.warning, bold=True)
        self.failover_label.hide()
        status_layout.addWidget(self.mode_label)
        status_layout.addLayout(status_row)
        status_layout.addWidget(self.failover_label)
        status_layout.addWidget(self.fallback_label)
        header.addWidget(status_panel, 3)
        root.addLayout(header)

        self.ticker_label = _label(
            self.scenario.ticker[0],
            size=16 if self.compact_layout else THEME.font_regular,
            color=THEME.gpu,
            bold=True,
        )
        root.addWidget(self.ticker_label)

        main_row = QHBoxLayout()
        main_row.setSpacing(THEME.spacing_md)
        video_panel = _panel()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(
            THEME.spacing_sm,
            THEME.spacing_sm,
            THEME.spacing_sm,
            THEME.spacing_sm,
        )
        self.main_canvas = FrameCanvas()
        self.main_canvas.header_text = self.scenario.title
        video_layout.addWidget(self.main_canvas)
        main_row.addWidget(video_panel, 6)

        gauge_panel = QFrame()
        gauge_layout = QVBoxLayout(gauge_panel)
        gauge_layout.setContentsMargins(0, 0, 0, 0)
        gauge_layout.setSpacing(THEME.spacing_sm)
        self.gauges: dict[str, EngineGauge] = {}
        for engine in ("NPU", "GPU", "CPU"):
            gauge = EngineGauge(engine, compact=self.compact_layout)
            self.gauges[engine] = gauge
            gauge_layout.addWidget(gauge, 1)
        main_row.addWidget(gauge_panel, 4)
        if self.compact_layout:
            video_panel.setMaximumHeight(430)
            gauge_panel.setMaximumHeight(430)
        main_container = QWidget(self.demo_page)
        main_container_layout = QVBoxLayout(main_container)
        main_container_layout.setContentsMargins(0, 0, 0, 0)
        main_container_layout.setSpacing(0)
        main_container_layout.addLayout(main_row)
        if self.compact_layout:
            main_container.setMaximumHeight(430)
        root.addWidget(main_container, 1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(THEME.spacing_md)
        tile_panel = _panel()
        tile_layout = QVBoxLayout(tile_panel)
        tile_layout.setContentsMargins(
            THEME.spacing_sm,
            THEME.spacing_sm,
            THEME.spacing_sm,
            THEME.spacing_sm,
        )
        tile_layout.addWidget(_label("LIVE STREAM TILES", bold=True))
        self.stage_breakdown = StageBreakdown(compact=self.compact_layout)
        tile_layout.addWidget(self.stage_breakdown)
        self.tile_grid = QGridLayout()
        self.tile_grid.setSpacing(THEME.spacing_sm)
        tile_layout.addLayout(self.tile_grid, 1)
        self.tiles = [
            StreamTile(index, compact=self.compact_layout)
            for index in range(max(DENSITIES))
        ]
        bottom_row.addWidget(tile_panel, 7)

        metrics_panel = QFrame()
        metrics_layout = QGridLayout(metrics_panel)
        metrics_layout.setSpacing(THEME.spacing_sm)
        self.latency_tile = MetricTile("E2E ms")
        self.latency_tile.setToolTip(
            f"End-to-end p50 / p95 over the measured {THEME.metric_window_seconds:g}-second window"
        )
        self.inference_rate_tile = MetricTile("INFER/s")
        self.detection_rate_tile = MetricTile("DET/s")
        self.real_time_tile = MetricTile("REAL-TIME")
        self.event_tile = MetricTile("EVENTS")
        self.event_tile.setToolTip(
            "Business events in the rolling "
            f"{float(self.scenario.event_rules['event_window_s']):g}-second window"
        )
        self.rss_tile = MetricTile("RSS")
        for index, tile in enumerate(
            (
                self.latency_tile,
                self.inference_rate_tile,
                self.detection_rate_tile,
                self.real_time_tile,
                self.event_tile,
                self.rss_tile,
            )
        ):
            metrics_layout.addWidget(tile, index // 3, index % 3)
        bottom_row.addWidget(metrics_panel, 3)
        self.bottom_panels = (tile_panel, metrics_panel)
        root.addLayout(bottom_row, 0)
        bottom_row.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self.status_label = _label(
            "1-4 scenario · N/G toggle · C mode · +/- density · A attract · F1 operator · F11 fullscreen · Q quit",
            color=THEME.text_muted,
        )
        root.addWidget(self.status_label)
        self._update_policy_header()
        self._update_clock()

    def _build_attract_page(self) -> None:
        layout = QVBoxLayout(self.attract_page)
        layout.setContentsMargins(
            THEME.spacing_xl,
            THEME.spacing_xl,
            THEME.spacing_xl,
            THEME.spacing_xl,
        )
        layout.addStretch(2)
        self.attract_title = _label(
            self.scenario.title,
            size=THEME.attract_title_font,
            color=THEME.npu,
            bold=True,
        )
        self.attract_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.attract_title)
        self.attract_vertical = _label(
            self.scenario.vertical,
            size=THEME.attract_subtitle_font,
            color=THEME.text_muted,
            bold=True,
        )
        self.attract_vertical.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.attract_vertical)
        self.attract_business = _label(
            self.scenario.business_line,
            size=THEME.attract_business_font,
        )
        self.attract_business.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.attract_business)
        self.attract_hint = _label(
            "Press any key to return to the live measured demo",
            color=THEME.text_muted,
        )
        self.attract_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.attract_hint)
        layout.addStretch(3)

    def start(self) -> None:
        self.telemetry_sampler.start()
        self.video_clock.start()
        self.scenario_history.append(
            {"scenario": self.scenario.id, "at": time.time(), "reason": "startup"}
        )
        self._start_all_scenario_pools()

    def _update_runtime_gops_label(self) -> None:
        values: dict[str, float] = {}
        for stages in self.stream_placements.values():
            for info in stages.values():
                if info.device_gops is None:
                    continue
                for execution_device in info.execution_devices:
                    values[execution_device.split(".", 1)[0]] = info.device_gops
        if values:
            runtime = " · ".join(
                f"runtime {device} {value:.2f} TOPS"
                for device, value in sorted(values.items())
            )
            self.peak_label.setText(f"{self.peak_base_text} · {runtime}")
            self.peak_label.setToolTip(
                "Peak values come from platform_profiles.json; runtime values come from "
                "OpenVINO DEVICE_GOPS."
            )
        else:
            self.peak_label.setText(self.peak_base_text)
            self.peak_label.setToolTip("Peak values come from platform_profiles.json.")

    def _refresh_fallback_display(self) -> None:
        self.fallbacks = set(self.prewarm_fallbacks)
        for values in self.stream_fallbacks.values():
            self.fallbacks.update(values)
        self.fallback_history.update(self.fallbacks)
        self.fallback_label.setVisible(bool(self.fallbacks))
        if self.fallbacks:
            self.fallback_label.setToolTip("\n".join(sorted(self.fallbacks)))

    def _update_policy_header(self) -> None:
        self.policy_snapshot = self.policy.snapshot()
        self.mode_label.setText(self.policy_snapshot.label)
        gpu_failover = (
            self.policy_snapshot.mode in {DeviceMode.SPREAD, DeviceMode.SPLIT}
            and not self.policy_snapshot.gpu_enabled
            and self.policy_snapshot.npu_enabled
        )
        npu_failover = (
            self.policy_snapshot.mode in {DeviceMode.SPREAD, DeviceMode.SPLIT}
            and not self.policy_snapshot.npu_enabled
            and self.policy_snapshot.gpu_enabled
        )
        cpu_fallback = (
            self.policy_snapshot.mode in {DeviceMode.SPREAD, DeviceMode.SPLIT}
            and not self.policy_snapshot.gpu_enabled
            and not self.policy_snapshot.npu_enabled
        )
        self.failover_label.setText(
            "GPU OFF · NPU FAILOVER"
            if gpu_failover
            else "NPU OFF · GPU FAILOVER"
            if npu_failover
            else "NPU/GPU OFF · CPU FALLBACK"
            if cpu_fallback
            else ""
        )
        self.failover_label.setVisible(bool(self.failover_label.text()))
        self.video_clock.set_target_fps(
            THEME.tile_display_fps if self.policy_snapshot.density >= 4 else 0.0
        )
        self.gauges["NPU"].set_disabled(self.policy.is_disabled("NPU"))
        self.gauges["GPU"].set_disabled(self.policy.is_disabled("GPU"))
        self.gauges["CPU"].set_disabled(False)
        self._refresh_fallback_display()
        self._layout_tiles()

    def _layout_tiles(self) -> None:
        density = self.policy_snapshot.density
        columns = min(density, 4)
        rows = (density + columns - 1) // columns
        maximum_height = (
            (270 if self.compact_layout else THEME.single_tile_row_max_height)
            if rows == 1
            else (390 if self.compact_layout else THEME.multi_tile_row_max_height)
        )
        for panel in self.bottom_panels:
            panel.setMaximumHeight(maximum_height)
        for tile in self.tiles:
            tile.hide()
            self.tile_grid.removeWidget(tile)
        for index in range(density):
            self.tile_grid.addWidget(self.tiles[index], index // columns, index % columns)
            self.tiles[index].show()
            self.tiles[index].set_thumbnail_visible(self.policy_snapshot.density > 1)
        for row in range(rows):
            self.tile_grid.setRowStretch(row, 1)
        for column in range(columns):
            self.tile_grid.setColumnStretch(column, 1)

    def _assignments(
        self,
        scenario: ScenarioConfig,
        index: int,
    ) -> tuple[StageAssignment, ...]:
        return build_stage_assignments(
            self.policy,
            scenario,
            index,
            availability=self.availability,
        )

    def _create_worker(
        self,
        index: int,
        sequence: int,
    ) -> ScenarioStreamWorker:
        worker = ScenarioStreamWorker(
            stream_index=index,
            scenario=self.scenario,
            catalog=self.catalog,
            camera_index=self.camera_index,
            registry=self.registry,
            assignments=self._assignments(self.scenario, index),
            policy_sequence=sequence,
            npu_duty_cycle=self.npu_duty_cycle,
            parent=self,
        )
        worker.frame_ready.connect(self._on_pipeline_frame)
        worker.placement_ready.connect(self._on_placement_ready)
        worker.failed.connect(self._on_stream_failed)
        return worker

    def _stop_scenario_workers(
        self,
        workers: list[ScenarioStreamWorker],
        *,
        delete_objects: bool,
    ) -> None:
        for worker in workers:
            worker.requestInterruption()
        for worker in workers:
            if not worker.wait(7000):
                LOGGER.error("Scenario stream did not stop within seven seconds")
        if delete_objects:
            for worker in workers:
                worker.setParent(None)
                worker.deleteLater()
            gc.collect()

    def _start_all_scenario_pools(self) -> None:
        sequence = self.policy_snapshot.sequence
        for index in range(self.policy_snapshot.density):
            worker = self._create_worker(index, sequence)
            self.stream_workers[index] = worker
            worker.start()

    def _sync_all_scenario_pools(
        self,
        snapshot: PolicySnapshot,
        requested_at: float,
        *,
        initial: bool,
    ) -> None:
        sequence = self.runtime_sequence
        active_indices = set(range(snapshot.density))
        if not initial:
            self.pending_acks[sequence] = set(active_indices)
            self.policy_requested_at[sequence] = time.time()
            self.policy_requested_monotonic[sequence] = requested_at
        removed = [
            self.stream_workers.pop(index)
            for index in list(self.stream_workers)
            if index not in active_indices
        ]
        if removed:
            self._stop_scenario_workers(removed, delete_objects=True)
        for index in sorted(active_indices):
            if index not in self.stream_workers:
                worker = self._create_worker(index, sequence)
                self.stream_workers[index] = worker
                worker.start()
            else:
                self.stream_workers[index].set_policy(
                    sequence,
                    self._assignments(self.scenario, index),
                    requested_at,
                )
        self._update_policy_header()

    def _apply_policy_change(self, changed: bool) -> None:
        snapshot = self.policy.snapshot()
        self.runtime_sequence = max(self.runtime_sequence, snapshot.sequence)
        if changed:
            self._sync_all_scenario_pools(
                snapshot,
                time.monotonic(),
                initial=False,
            )
        else:
            self._update_policy_header()

    def toggle_npu(self) -> None:
        before = self.policy.snapshot().sequence
        self.policy.toggle_npu()
        self._apply_policy_change(self.policy.snapshot().sequence != before)

    def toggle_gpu(self) -> None:
        before = self.policy.snapshot().sequence
        self.policy.toggle_gpu()
        self._apply_policy_change(self.policy.snapshot().sequence != before)

    def cycle_mode(self) -> None:
        before = self.policy.snapshot().sequence
        self.policy.cycle_mode()
        self._apply_policy_change(self.policy.snapshot().sequence != before)

    def change_density(self, direction: int) -> None:
        before = self.policy.snapshot().sequence
        self.policy.cycle_density(direction)
        self._apply_policy_change(self.policy.snapshot().sequence != before)

    def switch_scenario(self, scenario_id: str, *, reason: str = "keyboard") -> None:
        if scenario_id not in SCENARIO_ORDER or scenario_id == self.scenario.id:
            return
        self.runtime_sequence += 1
        self.scenario = self.catalog[scenario_id]
        self.video_clock.set_video_path(
            self.catalog.media_path(self.scenario.id, self.camera_index)
        )
        self.title_label.setText(self.scenario.title)
        self.main_canvas.header_text = self.scenario.title
        self.business_label.setText(self.scenario.business_line)
        self.attract_title.setText(self.scenario.title)
        self.attract_vertical.setText(self.scenario.vertical)
        self.attract_business.setText(self.scenario.business_line)
        self.ticker_index = 0
        self.ticker_label.setText(self.scenario.ticker[0])
        self.scenario_history.append(
            {"scenario": self.scenario.id, "at": time.time(), "reason": reason}
        )
        self.stream_placements.clear()
        self._update_runtime_gops_label()
        self.stream_detections.clear()
        self.stream_metrics.clear()
        self.stream_event_text.clear()
        self.stream_fallbacks.clear()
        requested_at = time.monotonic()
        active_indices = set(range(self.policy_snapshot.density))
        self.pending_acks[self.runtime_sequence] = set(active_indices)
        self.policy_requested_at[self.runtime_sequence] = time.time()
        self.policy_requested_monotonic[self.runtime_sequence] = requested_at
        for index, worker in self.stream_workers.items():
            worker.set_runtime(
                self.runtime_sequence,
                self.scenario,
                self._assignments(self.scenario, index),
                requested_at,
            )
        self._update_policy_header()
        self.status_label.setText(
            f"Scenario switched to {self.scenario.title} · no process reload"
        )

    def set_attract_mode(self, enabled: bool) -> None:
        self.attract_mode = enabled
        self.stack.setCurrentIndex(1 if enabled else 0)
        self.live_badge.setText("ATTRACT" if enabled else "LIVE")
        if enabled:
            self.attract_index = SCENARIO_ORDER.index(self.scenario.id)
            self.attract_timer.start()
        else:
            self.attract_timer.stop()

    def _cycle_attract_scenario(self) -> None:
        self.attract_index = (self.attract_index + 1) % len(SCENARIO_ORDER)
        self.switch_scenario(
            SCENARIO_ORDER[self.attract_index],
            reason="attract",
        )

    def _rotate_ticker(self) -> None:
        if not self.scenario.ticker:
            return
        self.ticker_index = (self.ticker_index + 1) % len(self.scenario.ticker)
        self.ticker_label.setText(self.scenario.ticker[self.ticker_index])

    def _update_clock(self) -> None:
        # This timer is the UI thread's heartbeat for the hang watchdog: if it stops firing, the
        # event loop is blocked and hangwatch dumps every thread's stack into the session log.
        hangwatch.mark_ui_tick()
        # Re-arm the C-level timer on every tick. A Python watchdog thread cannot run while the
        # main thread is blocked in a C call that holds the GIL, which is the case that actually
        # hangs; this timer is C-level and still fires, so the stack is captured either way.
        hangwatch.arm_dump_later()
        self.clock_label.setText(datetime.now().strftime("%H:%M:%S"))

    def _on_video_frame(self, image: QImage, frame_index: int) -> None:
        self.latest_image = image
        self.latest_image_size = (image.width(), image.height())
        detections = self.stream_detections.get(0, ())
        self.main_canvas.set_frame(
            image,
            detections,
            self.scenario.zones,
            self.latest_image_size,
            self.stream_event_text.get(0, ""),
        )
        for index in range(self.policy_snapshot.density):
            self.tiles[index].update_stream(
                image,
                self.stream_detections.get(index, ()),
                self.scenario.zones,
                self.latest_image_size,
                self.stream_placements.get(index, {}),
                {stage.stage: stage.model_id for stage in self.scenario.stages},
                self.stream_metrics.get(index),
                self.stream_event_text.get(index, ""),
            )

    def _on_pipeline_frame(self, index: int, frame: object) -> None:
        pipeline_frame = frame  # type: ignore[assignment]
        if pipeline_frame.scenario_id != self.scenario.id:
            return
        self.stream_detections[index] = pipeline_frame.detections
        self.stream_metrics[index] = pipeline_frame.metrics
        self.stream_placements[index] = pipeline_frame.placements
        self.frame_history.append(pipeline_frame.metrics)
        if pipeline_frame.events:
            self.events.extend(pipeline_frame.events)
            event = pipeline_frame.events[-1]
            self.stream_event_text[index] = (
                f"{event.type} · {event.label} {event.confidence:.0%} · {event.zone}"
            )
        if pipeline_frame.fallbacks:
            self.stream_fallbacks[index] = set(pipeline_frame.fallbacks)
            self.fallback_history.update(self.stream_fallbacks[index])
            self._refresh_fallback_display()
        if index == 0 and self.latest_image is not None:
            self.main_canvas.set_frame(
                self.latest_image,
                pipeline_frame.detections,
                pipeline_frame.zones,
                self.latest_image_size,
                self.stream_event_text.get(index, ""),
            )
        self._update_pipeline_metrics()

    def _on_placement_ready(
        self,
        index: int,
        payload: object,
        sequence: int,
        transition_ms: float,
    ) -> None:
        data = payload  # type: ignore[assignment]
        if data.get("scenario_id") != self.scenario.id:
            return
        placements: dict[str, RunnerInfo] = {}
        for stage, values in data["placements"].items():
            values = dict(values)
            values["execution_devices"] = tuple(values["execution_devices"])
            values["model_input_shape"] = tuple(values["model_input_shape"])
            placements[stage] = RunnerInfo(**values)
        self.stream_placements[index] = placements
        self._update_runtime_gops_label()
        if index == 0:
            self._update_pipeline_metrics()
        self.stream_fallbacks[index] = set(data.get("fallbacks", []))
        self._refresh_fallback_display()
        pending = self.pending_acks.get(sequence)
        if pending is not None:
            pending.discard(index)
            if not pending:
                self.pending_acks.pop(sequence, None)
                self.policy_transitions.append(
                    {
                        "sequence": sequence,
                        "scenario": self.scenario.id,
                        "policy": self.policy_snapshot.label,
                        "all_streams_ms": transition_ms,
                        "requested_at": self.policy_requested_at.get(sequence),
                        "requested_monotonic": self.policy_requested_monotonic.get(sequence),
                        "at": time.time(),
                    }
                )
                self.status_label.setText(
                    f"Policy/scenario live in {transition_ms:.0f} ms · "
                    f"{self.scenario.title} · F1 operator"
                )

    def _on_telemetry(self, frame: TelemetryFrame) -> None:
        self.latest_telemetry = frame
        self.telemetry_history.append(frame)
        for metric in frame.engine_metrics:
            self.gauges[metric.engine].update_metric(metric)
        self.rss_tile.set_value(
            f"{frame.process_rss_bytes / THEME.bytes_per_mib:.0f} MiB"
        )
        now = time.time()
        if now - self._last_rss_sample >= THEME.rss_sample_seconds:
            self._last_rss_sample = now
            self.rss_timeline.append(
                {
                    "elapsed_s": now - self.started_at,
                    "rss_bytes": float(frame.process_rss_bytes),
                }
            )

    def _update_pipeline_metrics(self) -> None:
        metrics = list(self.stream_metrics.values())
        if not metrics:
            return
        recent = [
            item
            for item in self.frame_history
            if item.scenario_id == self.scenario.id
            and time.time() - item.captured_at <= THEME.metric_window_seconds
        ]
        stream_zero = [item for item in recent if item.stream_index == 0]
        latencies = [item.end_to_end_ms for item in stream_zero]
        p50 = float(np.percentile(latencies, THEME.percentile_p50)) if latencies else None
        p95 = float(np.percentile(latencies, THEME.percentile_p95)) if latencies else None
        self.latency_tile.set_value(
            "—" if p50 is None else f"{p50:.0f}/{p95:.0f}"
        )
        self.inference_rate_tile.set_value(
            f"{sum(item.inferences_per_second for item in metrics):.1f}"
        )
        self.detection_rate_tile.set_value(
            f"{sum(item.detections_per_second for item in metrics):.1f}"
        )
        real_time = sum(
            bool(
                item.source_fps
                and item.processing_fps
                >= item.source_fps * THEME.streams_real_time_fraction
            )
            for item in metrics
        )
        primary = next(
            (item for item in metrics if item.stream_index == 0),
            None,
        )
        self.real_time_tile.set_value(
            "—" if primary is None else f"{primary.processing_fps:.1f} FPS"
        )
        self.real_time_tile.setToolTip(
            "Stream 0 processing FPS · "
            f"{real_time}/{self.policy_snapshot.density} streams meet the "
            f"{THEME.streams_real_time_fraction:.0%} real-time threshold"
        )
        event_counts: dict[str, int] = {}
        for item in metrics:
            for event_type, count in item.event_counts_60s.items():
                event_counts[event_type] = event_counts.get(event_type, 0) + count
        self.event_tile.set_value(str(sum(event_counts.values())))
        primary_stages = next(
            (
                item.stages
                for item in reversed(self.frame_history)
                if item.stream_index == 0 and item.scenario_id == self.scenario.id
            ),
            (),
        )
        placements = self.stream_placements.get(0, {})
        primary_stages = tuple(
            replace(
                metric,
                requested_device=placements[metric.stage].requested_device,
                execution_devices=placements[metric.stage].execution_devices,
            )
            if metric.stage in placements
            else metric
            for metric in primary_stages
        )
        self.stage_breakdown.set_metrics(primary_stages)

    def _on_stream_failed(self, index: int, traceback_text: str) -> None:
        self.last_error = f"stream {index}: {traceback_text}"
        LOGGER.error("Scenario stream %d failed:\n%s", index, traceback_text)
        self.status_label.setText(
            f"STREAM {index} ERROR — application remains live; see operator log"
        )
        self.status_label.setStyleSheet(
            f"color: {THEME.danger}; background: transparent;"
        )

    def _on_video_failed(self, traceback_text: str) -> None:
        self.last_error = traceback_text
        LOGGER.error("Video clock failed:\n%s", traceback_text)
        self.status_label.setText("VIDEO SOURCE ERROR — see logs/session-*.log")

    def _open_operator(self) -> None:
        dialog = OperatorOverlay(
            availability=self.availability,
            catalog=self.catalog,
            scenario=self.scenario,
            registry=self.registry,
            telemetry_map_path=self.telemetry_map_path,
            profiles_path=self.profiles_path,
            cache_dir=self.cache_dir,
            placements=self.stream_placements,
            events=tuple(self.events),
            parent=self,
        )
        dialog.exec()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        was_attract = self.attract_mode
        if was_attract:
            self.set_attract_mode(False)
            if key == Qt.Key.Key_F1:
                self._open_operator()
            elif key == Qt.Key.Key_F11:
                self.showFullScreen()
            elif key == Qt.Key.Key_Q:
                self.close()
            return
        if key == Qt.Key.Key_1:
            self.switch_scenario("retail")
        elif key == Qt.Key.Key_2:
            self.switch_scenario("smart_city")
        elif key == Qt.Key.Key_3:
            self.switch_scenario("medical")
        elif key == Qt.Key.Key_4:
            self.switch_scenario("gov_defense")
        elif key == Qt.Key.Key_N:
            self.toggle_npu()
        elif key == Qt.Key.Key_G:
            self.toggle_gpu()
        elif key == Qt.Key.Key_C:
            self.cycle_mode()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.change_density(1)
        elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.change_density(-1)
        elif key == Qt.Key.Key_A:
            self.set_attract_mode(not self.attract_mode)
        elif key == Qt.Key.Key_F1:
            self._open_operator()
        elif key == Qt.Key.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif key == Qt.Key.Key_Q:
            answer = QMessageBox.question(
                self,
                "Quit Engine Lab?",
                "Stop the measured pipeline and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.close()
        elif key == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.ticker_timer.stop()
        self.clock_timer.stop()
        self.attract_timer.stop()
        self.video_clock.requestInterruption()
        workers = list(self.stream_workers.values())
        self._stop_scenario_workers(workers, delete_objects=False)
        self.video_clock.wait(3000)
        self.telemetry_sampler.stop()

    def closeEvent(self, event: Any) -> None:
        self.shutdown()
        super().closeEvent(event)

    def save_screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pixmap: QPixmap = self.grab()
        if not pixmap.save(str(path), "PNG"):
            raise RuntimeError(f"could not save screenshot: {path}")

    @staticmethod
    def _measurement_stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        samples = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(samples)),
            "p50": float(np.percentile(samples, THEME.percentile_p50)),
            "p95": float(np.percentile(samples, THEME.percentile_p95)),
            "max": float(np.max(samples)),
            "mean": float(np.mean(samples)),
        }

    def diagnostic_snapshot(self) -> dict[str, Any]:
        frames = list(self.frame_history)
        recent = [
            item
            for item in frames
            if item.scenario_id == self.scenario.id
            and time.time() - item.captured_at <= THEME.metric_window_seconds
        ]
        stream_metrics = list(self.stream_metrics.values())
        strict_real_time = sum(
            bool(
                item.source_fps
                and item.processing_fps
                >= item.source_fps * THEME.streams_real_time_fraction
            )
            for item in stream_metrics
        )
        real_time_equivalents = sum(
            min(
                THEME.real_time_equivalent_cap,
                item.processing_fps / item.source_fps,
            )
            for item in stream_metrics
            if item.source_fps
        )
        measurement_summary = {
            "window_seconds": THEME.metric_window_seconds,
            "end_to_end_ms": self._measurement_stats(
                [item.end_to_end_ms for item in recent]
            ),
            "inference_ms": self._measurement_stats(
                [item.inference_ms for item in recent]
            ),
            "inferences_per_second": sum(
                item.inferences_per_second for item in stream_metrics
            ),
            "detections_per_second": sum(
                item.detections_per_second for item in stream_metrics
            ),
            "streams": {
                "configured": self.policy_snapshot.density,
                "processed": len(stream_metrics),
                "strict_streams_in_real_time": strict_real_time,
                "real_time_stream_equivalents": real_time_equivalents,
                "real_time_threshold_fraction": THEME.streams_real_time_fraction,
            },
        }
        latest_telemetry = None
        if self.latest_telemetry is not None:
            latest_telemetry = {
                "sampled_at": self.latest_telemetry.sampled_at,
                "engine_metrics": [
                    asdict(metric) for metric in self.latest_telemetry.engine_metrics
                ],
                "cpu_per_core_percent": list(
                    self.latest_telemetry.cpu_per_core_percent
                ),
                "process_rss_bytes": self.latest_telemetry.process_rss_bytes,
                "pdh_error": self.latest_telemetry.pdh_error,
            }
        gauge_states = {
            engine: {
                "disabled_by_operator": gauge.disabled,
                "last_measurement": (
                    asdict(gauge.engine_metric)
                    if gauge.engine_metric is not None
                    else None
                ),
            }
            for engine, gauge in self.gauges.items()
        }
        return {
            "captured_at": time.time(),
            "started_at": self.started_at,
            "startup_elapsed_s": self.startup_elapsed_s,
            "scenario": self.scenario.id,
            "scenario_history": list(self.scenario_history),
            "attract_mode": self.attract_mode,
            "policy": asdict(self.policy_snapshot),
            "density": self.policy_snapshot.density,
            "stream_metrics": {
                str(index): asdict(metrics)
                for index, metrics in self.stream_metrics.items()
            },
            "stream_placements": {
                str(index): {
                    stage: asdict(info) for stage, info in placements.items()
                }
                for index, placements in self.stream_placements.items()
            },
            "events": [event.to_dict() for event in self.events],
            "fallbacks": sorted(self.fallbacks),
            "fallback_history": sorted(self.fallback_history),
            "policy_transitions": list(self.policy_transitions),
            "rss_timeline": list(self.rss_timeline),
            "availability": self.availability.to_dict(),
            "measurement_summary": measurement_summary,
            "gauge_states": gauge_states,
            "latest_telemetry": latest_telemetry,
            "frame_samples": [asdict(item) for item in frames],
            "telemetry_samples": [
                {
                    "sampled_at": item.sampled_at,
                    "engine_metrics": [
                        asdict(metric) for metric in item.engine_metrics
                    ],
                    "process_rss_bytes": item.process_rss_bytes,
                    "pdh_error": item.pdh_error,
                }
                for item in self.telemetry_history
            ],
            "last_error": self.last_error,
        }

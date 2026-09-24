"""Minimal Phase 1 Qt HUD: retail video, detections, and three measured gauges."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.engine.runner import InferenceFrameMetrics, InferenceThread, RunnerInfo
from app.engine.stages import Detection
from app.overlay import draw_detections
from app.telemetry.devices_win import (
    enumerate_compute_accelerators,
    enumerate_display_adapters,
    is_intel,
    is_npu,
)
from app.telemetry.npu_fallback import NpuDutyCycle
from app.telemetry.sampler import TelemetryFrame, TelemetrySampler
from app.theme import THEME


RETAIL_BUSINESS_LINE = (
    "Self-checkout and loss prevention: detect the item, classify it, log the event — all on one SoC."
)
LOGGER = logging.getLogger("engine_lab")


def _label(text: str, *, size: int = THEME.font_regular, color: str = THEME.text, bold: bool = False) -> QLabel:
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
        f"background: {THEME.panel}; border: 1px solid {THEME.border}; border-radius: {THEME.radius_panel}px;"
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


def platform_text() -> tuple[str, str, str, str, str]:
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
    physical = psutil.cpu_count(logical=False) or 0
    logical = psutil.cpu_count(logical=True) or 0
    topology = f"{physical}P / {logical}L cores"
    return cpu_name(), gpu_name, gpu_driver, npu_name, npu_driver


class VideoCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._detections: tuple[Detection, ...] = ()
        self._source_size = (0, 0)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(
        self,
        image: QImage,
        detections: tuple[Detection, ...],
        source_size: tuple[int, int],
    ) -> None:
        self._image = image
        self._detections = detections
        self._source_size = source_size
        self.update()

    def _video_rect(self) -> QRectF:
        if self._image is None or self._image.isNull():
            return QRectF()
        scale = min(
            self.width() / self._image.width(),
            self.height() / self._image.height(),
        )
        width = self._image.width() * scale
        height = self._image.height() * scale
        return QRectF(
            (self.width() - width) / 2.0,
            (self.height() - height) / 2.0,
            width,
            height,
        )

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#03080D"))
        if self._image is None or self._image.isNull():
            painter.setPen(QPen(QColor(THEME.text_muted)))
            painter.setFont(QFont(THEME.font_family, THEME.font_title))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Starting measured pipeline…")
            return
        video_rect = self._video_rect()
        painter.drawImage(video_rect, self._image)
        draw_detections(
            painter,
            video_rect,
            self._detections,
            self._source_size[0],
            self._source_size[1],
        )
        painter.setPen(QPen(QColor(THEME.overlay_text)))
        painter.setFont(QFont(THEME.font_family, THEME.font_overlay))
        painter.fillRect(
            QRectF(
                video_rect.left(),
                video_rect.top(),
                THEME.overlay_badge_width,
                THEME.overlay_label_height,
            ),
            QColor(THEME.overlay_fill),
        )
        painter.drawText(
            QRectF(
                video_rect.left(),
                video_rect.top(),
                THEME.overlay_badge_width,
                THEME.overlay_label_height,
            ),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Retail · {len(self._detections)} detection(s)",
        )


class EngineGauge(QWidget):
    def __init__(self, engine: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.engine_metric = None
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_metric(self, metric: Any) -> None:
        self.engine_metric = metric
        self.setToolTip(f"{metric.source}\n{metric.detail}")
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        accent = {
            "NPU": THEME.npu,
            "GPU": THEME.gpu,
            "CPU": THEME.cpu,
        }[self.engine]
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(THEME.border), 1))
        painter.setBrush(QColor(THEME.panel_alt))
        painter.drawRoundedRect(rect, THEME.radius_panel, THEME.radius_panel)

        x = THEME.spacing_lg
        y = THEME.spacing_sm
        width = self.width() - THEME.spacing_lg * 2
        painter.setFont(QFont(THEME.font_family, THEME.font_semibold, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(accent)))
        painter.drawText(
            QRectF(x, y, width * 0.55, 32),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.engine,
        )
        state = "WAITING" if self.engine_metric is None else self.engine_metric.state
        painter.setFont(QFont(THEME.font_family, THEME.font_badge, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(x + width * 0.45, y, width * 0.55, 32),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            state,
        )

        value_text = "—" if self.engine_metric is None or self.engine_metric.value_percent is None else f"{self.engine_metric.value_percent:.0f}%"
        painter.setFont(QFont(THEME.font_family, THEME.font_engine_value, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(THEME.text)))
        painter.drawText(
            QRectF(x, y + 18, width, 72),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            value_text,
        )

        bar = QRectF(x, y + 86, width, THEME.gauge_height)
        painter.fillRect(bar, QColor(THEME.border))
        if self.engine_metric is not None and self.engine_metric.value_percent is not None:
            fraction = max(0.0, min(1.0, self.engine_metric.value_percent / 100.0))
            painter.fillRect(
                QRectF(bar.left(), bar.top(), bar.width() * fraction, bar.height()),
                QColor(accent),
            )
        source = "Measured source pending" if self.engine_metric is None else self.engine_metric.source
        source_font = QFont(THEME.font_family, THEME.font_regular)
        painter.setFont(source_font)
        painter.setPen(QPen(QColor(THEME.text_muted)))
        source_rect = QRectF(x, y + 118, width, 28)
        painter.drawText(
            source_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                source,
                Qt.TextElideMode.ElideRight,
                int(source_rect.width()),
            ),
        )


class MetricTile(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"background: {THEME.panel_alt}; border: 1px solid {THEME.border}; border-radius: {THEME.radius_small}px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            THEME.spacing_md,
            THEME.spacing_sm,
            THEME.spacing_md,
            THEME.spacing_sm,
        )
        self.title = _label(title, size=THEME.font_regular, color=THEME.text_muted, bold=True)
        self.value = _label("—", size=THEME.font_metric, color=THEME.text, bold=True)
        layout.addWidget(self.title)
        layout.addWidget(self.value)

    def set_value(self, value: str) -> None:
        self.value.setText(value)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        device: str,
        source_description: str,
        video_path: Path | None,
        camera_index: int | None,
        cache_dir: Path,
        telemetry_map_path: Path,
        fullscreen: bool,
    ) -> None:
        super().__init__()
        self.device = device.upper()
        self.source_description = source_description
        self.cache_dir = cache_dir
        self.started_at = time.time()
        self.frame_count = 0
        self.frame_history: deque[InferenceFrameMetrics] = deque(maxlen=900)
        self.telemetry_history: deque[TelemetryFrame] = deque(maxlen=100)
        self.latest_frame_metrics: InferenceFrameMetrics | None = None
        self.latest_telemetry: TelemetryFrame | None = None
        self.runner_info: RunnerInfo | None = None
        self.last_error: str | None = None
        self._shutdown = False
        self.npu_duty_cycle = NpuDutyCycle()
        self.inference_thread = InferenceThread(
            model_path=Path("models/yolo11n-fp16/yolo11n.xml").resolve(),
            labels_path=Path("models/yolo11n-fp16/labels.txt").resolve(),
            preprocess_config_path=Path("models/yolo11n-fp16/source_config.json").resolve(),
            video_path=video_path,
            camera_index=camera_index,
            device=self.device,
            cache_dir=cache_dir,
            npu_duty_cycle=self.npu_duty_cycle,
            parent=self,
        )
        self.telemetry_sampler = TelemetrySampler(
            telemetry_map_path=telemetry_map_path,
            npu_duty_cycle=self.npu_duty_cycle,
            interval_seconds=0.2,
            parent=self,
        )
        self._build_ui()
        self.inference_thread.runner_ready.connect(self._on_runner_ready)
        self.inference_thread.frame_ready.connect(self._on_frame_ready)
        self.inference_thread.failed.connect(self._on_failed)
        self.telemetry_sampler.frame_ready.connect(self._on_telemetry)
        if fullscreen:
            self.showFullScreen()
        else:
            self.show()

    def _build_ui(self) -> None:
        self.setWindowTitle("Engine Lab — Retail / YOLO11n FP16")
        self.setMinimumSize(THEME.minimum_window_width, THEME.minimum_window_height)
        self.resize(THEME.design_width, THEME.design_height)
        self.setStyleSheet(f"background: {THEME.background}; color: {THEME.text};")

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(
            THEME.spacing_lg,
            THEME.spacing_lg,
            THEME.spacing_lg,
            THEME.spacing_lg,
        )
        root_layout.setSpacing(THEME.spacing_md)
        self.setCentralWidget(root)

        cpu, gpu, gpu_driver, npu, npu_driver = platform_text()
        header = QHBoxLayout()
        header.setSpacing(THEME.spacing_lg)
        platform_panel = _panel()
        platform_layout = QVBoxLayout(platform_panel)
        platform_layout.setContentsMargins(
            THEME.spacing_md,
            THEME.spacing_sm,
            THEME.spacing_md,
            THEME.spacing_sm,
        )
        platform_layout.addWidget(_label(cpu, bold=True))
        platform_layout.addWidget(_label(f"iGPU · {gpu} · driver {gpu_driver}", color=THEME.text_muted))
        platform_layout.addWidget(_label(f"NPU · {npu} · driver {npu_driver}", color=THEME.text_muted))
        header.addWidget(platform_panel, 3)

        title_panel = _panel()
        title_layout = QVBoxLayout(title_panel)
        title_layout.setContentsMargins(
            THEME.spacing_md,
            THEME.spacing_sm,
            THEME.spacing_md,
            THEME.spacing_sm,
        )
        title_layout.addWidget(_label("Retail / POS", size=THEME.font_title, bold=True))
        title_layout.addWidget(_label(RETAIL_BUSINESS_LINE, color=THEME.text_muted))
        header.addWidget(title_panel, 5)

        mode_panel = _panel()
        mode_layout = QVBoxLayout(mode_panel)
        mode_layout.setContentsMargins(
            THEME.spacing_md,
            THEME.spacing_sm,
            THEME.spacing_md,
            THEME.spacing_sm,
        )
        self.device_label = _label(f"DEVICE · {self.device}", size=THEME.font_title, color=THEME.npu, bold=True)
        self.placement_label = _label("EXECUTION_DEVICES · starting", color=THEME.text_muted)
        self.source_label = _label(self.source_description, color=THEME.text_muted)
        mode_layout.addWidget(self.device_label)
        mode_layout.addWidget(self.placement_label)
        mode_layout.addWidget(self.source_label)
        header.addWidget(mode_panel, 3)
        root_layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(THEME.spacing_md)
        video_panel = _panel()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(THEME.spacing_sm, THEME.spacing_sm, THEME.spacing_sm, THEME.spacing_sm)
        self.video_canvas = VideoCanvas()
        video_layout.addWidget(self.video_canvas)
        body.addWidget(video_panel, 7)

        gauge_panel = QFrame()
        gauge_layout = QVBoxLayout(gauge_panel)
        gauge_layout.setContentsMargins(0, 0, 0, 0)
        gauge_layout.setSpacing(THEME.spacing_sm)
        self.gauges: dict[str, EngineGauge] = {}
        for engine in ("NPU", "GPU", "CPU"):
            gauge = EngineGauge(engine)
            self.gauges[engine] = gauge
            gauge_layout.addWidget(gauge, 1)
        body.addWidget(gauge_panel, 3)
        root_layout.addLayout(body, 1)

        tiles = QHBoxLayout()
        tiles.setSpacing(THEME.spacing_sm)
        self.fps_tile = MetricTile("PROCESSING FPS")
        self.inference_tile = MetricTile("INFERENCE ms")
        self.latency_tile = MetricTile("END-TO-END ms")
        self.detections_tile = MetricTile("DETECTIONS")
        self.rss_tile = MetricTile("PROCESS RSS")
        for tile in (
            self.fps_tile,
            self.inference_tile,
            self.latency_tile,
            self.detections_tile,
            self.rss_tile,
        ):
            tiles.addWidget(tile, 1)
        root_layout.addLayout(tiles)

        self.status_label = _label(
            "F11 fullscreen · Q quit · values are measured, not simulated",
            color=THEME.text_muted,
        )
        root_layout.addWidget(self.status_label)

    def start(self) -> None:
        self.telemetry_sampler.start()
        self.inference_thread.start()

    def _on_runner_ready(self, info: RunnerInfo) -> None:
        self.runner_info = info
        self.placement_label.setText(
            "EXECUTION_DEVICES · " + ", ".join(info.execution_devices)
        )
        self.status_label.setText(
            f"YOLO11n FP16 · measured placement verified · compile {info.compile_ms:.0f} ms · "
            f"input {info.model_input_shape} · F11 fullscreen · Q quit"
        )

    def _on_frame_ready(
        self,
        image: QImage,
        detections: tuple[Detection, ...],
        metrics: InferenceFrameMetrics,
    ) -> None:
        self.frame_count += 1
        self.latest_frame_metrics = metrics
        self.frame_history.append(metrics)
        self.video_canvas.set_frame(
            image,
            detections,
            (metrics.frame_width, metrics.frame_height),
        )
        source_fps = "—" if metrics.source_fps is None else f"{metrics.source_fps:.2f}"
        self.fps_tile.set_value(f"{metrics.processing_fps:.2f} / {source_fps}")
        self.inference_tile.set_value(f"{metrics.inference_ms:.2f}")
        self.latency_tile.set_value(f"{metrics.end_to_end_ms:.2f}")
        self.detections_tile.set_value(str(metrics.detection_count))

    def _on_telemetry(self, frame: TelemetryFrame) -> None:
        self.latest_telemetry = frame
        self.telemetry_history.append(frame)
        for metric in frame.engine_metrics:
            self.gauges[metric.engine].update_metric(metric)
        rss_mib = frame.process_rss_bytes / (1024.0 * 1024.0)
        self.rss_tile.set_value(f"{rss_mib:.0f} MiB")

    def _on_failed(self, traceback_text: str) -> None:
        self.last_error = traceback_text
        LOGGER.error("Inference pipeline failed:\n%s", traceback_text)
        self.status_label.setText("PIPELINE ERROR — see logs/session-*.log")
        self.status_label.setStyleSheet(f"color: {THEME.danger}; background: transparent;")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        elif key == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.inference_thread.requestInterruption()
        if not self.inference_thread.wait(5000):
            LOGGER.error("Inference thread did not stop within five seconds")
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
            "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
            "max": float(np.max(samples)),
            "mean": float(np.mean(samples)),
        }

    def diagnostic_snapshot(self) -> dict[str, Any]:
        telemetry = None
        if self.latest_telemetry is not None:
            telemetry = {
                "sampled_at": self.latest_telemetry.sampled_at,
                "engine_metrics": [asdict(metric) for metric in self.latest_telemetry.engine_metrics],
                "cpu_per_core_percent": list(self.latest_telemetry.cpu_per_core_percent),
                "process_rss_bytes": self.latest_telemetry.process_rss_bytes,
                "pdh_error": self.latest_telemetry.pdh_error,
            }
        frames = list(self.frame_history)
        frame_summary = {
            "processing_fps": self._measurement_stats([item.processing_fps for item in frames]),
            "decode_ms": self._measurement_stats([item.decode_ms for item in frames]),
            "preprocess_ms": self._measurement_stats([item.preprocess_ms for item in frames]),
            "inference_ms": self._measurement_stats([item.inference_ms for item in frames]),
            "postprocess_ms": self._measurement_stats([item.postprocess_ms for item in frames]),
            "end_to_end_ms": self._measurement_stats([item.end_to_end_ms for item in frames]),
        }
        telemetry_max: dict[str, float | None] = {"NPU": None, "GPU": None, "CPU": None}
        for frame in self.telemetry_history:
            for metric in frame.engine_metrics:
                if metric.value_percent is not None:
                    telemetry_max[metric.engine] = max(
                        telemetry_max[metric.engine] or 0.0,
                        metric.value_percent,
                    )
        return {
            "captured_at": time.time(),
            "started_at": self.started_at,
            "device_requested": self.device,
            "source": self.source_description,
            "frame_count": self.frame_count,
            "measurement_window_seconds": (
                frames[-1].captured_at - frames[0].captured_at if len(frames) > 1 else 0.0
            ),
            "runner": asdict(self.runner_info) if self.runner_info is not None else None,
            "measurement_summary": frame_summary,
            "telemetry_max_percent": telemetry_max,
            "latest_frame": asdict(self.latest_frame_metrics) if self.latest_frame_metrics is not None else None,
            "latest_telemetry": telemetry,
            "frame_samples": [asdict(item) for item in frames],
            "telemetry_samples": [
                {
                    "sampled_at": item.sampled_at,
                    "engine_metrics": [asdict(metric) for metric in item.engine_metrics],
                    "process_rss_bytes": item.process_rss_bytes,
                    "pdh_error": item.pdh_error,
                }
                for item in self.telemetry_history
            ],
            "last_error": self.last_error,
        }

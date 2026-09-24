"""Phase 2 Qt HUD with live device policy, telemetry, and stream tiles."""

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
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.engine.availability import AvailabilityMatrix
from app.engine.device_policy import DENSITIES, DeviceMode, DevicePolicy, PolicySnapshot
from app.engine.runner import (
    CompiledModelStore,
    RetailVideoClock,
    RunnerCompileConfig,
    RunnerInfo,
    StreamFrameMetrics,
    StreamWorker,
)
from app.engine.stages import Detection
from app.overlay import draw_detections
from app.telemetry.devices_win import (
    enumerate_compute_accelerators,
    enumerate_display_adapters,
    is_intel,
    is_npu,
)
from app.telemetry.npu_fallback import NpuDutyCycle
from app.telemetry.sampler import EngineMetric, TelemetryFrame, TelemetrySampler
from app.theme import THEME


RETAIL_BUSINESS_LINE = (
    "Self-checkout and loss prevention: detect the item, classify it, log the event — all on one SoC."
)
LOGGER = logging.getLogger("engine_lab")


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
            (item for item in enumerate_compute_accelerators() if is_intel(item) and is_npu(item)),
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
        device = next((item for item in enumerate_display_adapters() if is_intel(item)), None)
        if device is not None:
            gpu_name = device.friendly_name or device.description
            gpu_driver = device.driver_version or "unknown"
    except Exception:
        pass
    physical = psutil.cpu_count(logical=False) or 0
    logical = psutil.cpu_count(logical=True) or 0
    return cpu_name(), gpu_name, gpu_driver, npu_name, f"{npu_driver} · {physical}P/{logical}L"


class FrameCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._detections: tuple[Detection, ...] = ()
        self._source_size = (0, 0)
        self.placeholder = "Starting measured pipeline…"
        self.show_header = True
        self.setMinimumSize(320, 180)
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
        scale = min(self.width() / self._image.width(), self.height() / self._image.height())
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
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
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
        if self.show_header:
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


class StreamTile(QFrame):
    def __init__(self, stream_index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stream_index = stream_index
        self.setStyleSheet(
            f"background: {THEME.panel_alt}; border: 1px solid {THEME.border}; border-radius: {THEME.radius_small}px;"
        )
        self.setMinimumSize(260, 110)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(THEME.spacing_xs, THEME.spacing_xs, THEME.spacing_xs, THEME.spacing_xs)
        self.badge = _label(f"STREAM {stream_index} · starting", bold=True)
        self.canvas = FrameCanvas()
        self.canvas.setMinimumSize(80, 45)
        self.canvas.show_header = False
        self.canvas.placeholder = "starting"
        self.metrics = _label("FPS — · inference — ms", color=THEME.text_muted)
        layout.addWidget(self.badge)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.metrics)

    def update_stream(
        self,
        image: QImage,
        detections: tuple[Detection, ...],
        source_size: tuple[int, int],
        info: RunnerInfo | None,
        metrics: StreamFrameMetrics | None,
    ) -> None:
        if info is not None:
            self.badge.setText(
                f"STREAM {self.stream_index} · {info.requested_device} → "
                f"{','.join(info.execution_devices)}"
            )
            self.setToolTip(
                f"Requested: {info.requested_device}\n"
                f"EXECUTION_DEVICES: {', '.join(info.execution_devices)}\n"
                f"Performance hint: {info.performance_hint}"
            )
        if metrics is not None:
            self.metrics.setText(
                f"{metrics.processing_fps:.1f} FPS · infer {metrics.inference_ms:.2f} ms · "
                f"{metrics.detection_count} det"
            )
        self.canvas.set_frame(image, detections, source_size)


class EngineGauge(QWidget):
    def __init__(self, engine: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.engine_metric: EngineMetric | None = None
        self.disabled = False
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_metric(self, metric: EngineMetric) -> None:
        if self.disabled:
            return
        self.engine_metric = metric
        self.setToolTip(f"{metric.source}\n{metric.detail}")
        self.update()

    def set_disabled(self, disabled: bool) -> None:
        self.disabled = disabled
        self.update()

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
        painter.setFont(QFont(THEME.font_family, THEME.font_semibold, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(accent)))
        painter.drawText(
            QRectF(x, y, width * 0.5, 28),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{self.engine} · {provider}",
        )
        state = "OFF BY OPERATOR" if self.disabled else (
            "WAITING" if self.engine_metric is None else self.engine_metric.state
        )
        painter.setFont(QFont(THEME.font_family, THEME.font_badge, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(x + width * 0.35, y, width * 0.65, 28),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            state,
        )

        value = None if self.engine_metric is None else self.engine_metric.value_percent
        value_text = "—" if value is None else f"{value:.0f}%"
        painter.setFont(QFont(THEME.font_family, THEME.font_engine_value, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(THEME.text_muted if self.disabled else THEME.text)))
        painter.drawText(
            QRectF(x, y + 24, width, 66),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            value_text,
        )

        bar = QRectF(x, y + 92, width, THEME.gauge_height)
        painter.fillRect(bar, QColor(THEME.border))
        if value is not None:
            fraction = max(0.0, min(1.0, value / 100.0))
            painter.fillRect(
                QRectF(bar.left(), bar.top(), bar.width() * fraction, bar.height()),
                QColor(accent),
            )


class MetricTile(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"background: {THEME.panel_alt}; border: 1px solid {THEME.border}; border-radius: {THEME.radius_small}px;"
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


class AvailabilityDialog(QDialog):
    def __init__(self, matrix: AvailabilityMatrix, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("DeviceAvailability — model × device")
        self.resize(1100, 700)
        layout = QVBoxLayout(self)
        header = _label(
            f"Cache {'HIT' if matrix.cache_hit else 'MISS'} · fingerprint {matrix.fingerprint[:16]} · "
            f"available {','.join(matrix.available_devices)}",
            bold=True,
        )
        layout.addWidget(header)
        row_count = sum(len(model.devices) for model in matrix.models.values())
        table = QTableWidget(row_count, 5)
        table.setHorizontalHeaderLabels(["Model", "Device", "Status", "EXECUTION_DEVICES", "Error"])
        table.horizontalHeader().setStretchLastSection(True)
        row = 0
        for model_id, model in matrix.models.items():
            for device, result in model.devices.items():
                values = (
                    model_id,
                    device,
                    "PASS" if result.success else "FAIL",
                    ",".join(result.execution_devices),
                    result.error or "",
                )
                for column, value in enumerate(values):
                    table.setItem(row, column, QTableWidgetItem(value))
                row += 1
        table.resizeColumnsToContents()
        layout.addWidget(table)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        source_description: str,
        video_path: Path | None,
        camera_index: int | None,
        cache_dir: Path,
        telemetry_map_path: Path,
        availability: AvailabilityMatrix,
        fullscreen: bool,
        initial_device: str | None = None,
        initial_mode: DeviceMode | None = None,
        initial_density: int = 1,
        npu_enabled: bool = True,
        gpu_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.source_description = source_description
        self.cache_dir = cache_dir
        self.availability = availability
        self.started_at = time.time()
        self._shutdown = False
        self.npu_duty_cycle = NpuDutyCycle()
        retail_model_id = "yolo11n-fp16"
        retail_devices = tuple(
            device
            for device in availability.available_devices
            if availability.supports(retail_model_id, device)
        )
        if "CPU" not in retail_devices:
            raise RuntimeError("retail detector is unavailable on CPU; refusing silent fallback")
        initial_mode = initial_mode or self._mode_for_device(initial_device)
        self.policy = DevicePolicy(
            available_devices=retail_devices,
            mode=initial_mode,
            density=initial_density,
        )
        if not npu_enabled:
            self.policy.toggle_npu()
        if not gpu_enabled:
            self.policy.toggle_gpu()
        self.policy_snapshot = self.policy.snapshot()
        self.compiled_store = CompiledModelStore(
            Path("models/yolo11n-fp16/yolo11n.xml").resolve(),
            cache_dir,
        )
        self.video_clock = RetailVideoClock(video_path, camera_index, parent=self)
        self.telemetry_sampler = TelemetrySampler(
            telemetry_map_path,
            self.npu_duty_cycle,
            interval_seconds=0.2,
            parent=self,
        )
        self.stream_workers: dict[int, StreamWorker] = {}
        self.stream_info: dict[int, RunnerInfo] = {}
        self.stream_detections: dict[int, tuple[Detection, ...]] = {}
        self.stream_metrics: dict[int, StreamFrameMetrics] = {}
        self.frame_history: deque[StreamFrameMetrics] = deque(maxlen=2400)
        self.telemetry_history: deque[TelemetryFrame] = deque(maxlen=100)
        self.latest_telemetry: TelemetryFrame | None = None
        self.latest_image: QImage | None = None
        self.latest_image_size = (0, 0)
        self.pending_policy_acks: dict[int, set[int]] = {}
        self.policy_requested_at: dict[int, float] = {}
        self.policy_requested_monotonic: dict[int, float] = {}
        self.policy_by_sequence: dict[int, PolicySnapshot] = {
            self.policy_snapshot.sequence: self.policy_snapshot
        }
        self.policy_transitions: list[dict[str, Any]] = []
        self.last_error: str | None = None
        self._build_ui()
        self.video_clock.frame_ready.connect(self._on_video_frame)
        self.video_clock.failed.connect(self._on_video_failed)
        self.telemetry_sampler.frame_ready.connect(self._on_telemetry)
        if fullscreen:
            self.showFullScreen()
        else:
            self.show()

    @staticmethod
    def _mode_for_device(device: str | None) -> DeviceMode:
        if device is None:
            return DeviceMode.SPREAD
        return {
            "NPU": DeviceMode.NPU_ONLY,
            "GPU": DeviceMode.GPU_ONLY,
            "CPU": DeviceMode.CPU_ONLY,
        }[device.upper()]

    def _build_ui(self) -> None:
        self.setWindowTitle("Engine Lab — Retail / YOLO11n FP16 / Phase 2")
        self.setMinimumSize(THEME.minimum_window_width, THEME.minimum_window_height)
        self.resize(THEME.design_width, THEME.design_height)
        self.setStyleSheet(f"background: {THEME.background}; color: {THEME.text};")

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(
            THEME.spacing_md,
            THEME.spacing_md,
            THEME.spacing_md,
            THEME.spacing_md,
        )
        root_layout.setSpacing(THEME.spacing_sm)
        self.setCentralWidget(root)

        cpu, gpu, gpu_driver, npu, npu_driver = platform_text()
        header = QHBoxLayout()
        header.setSpacing(THEME.spacing_md)
        platform_panel = _panel()
        platform_layout = QVBoxLayout(platform_panel)
        platform_layout.setContentsMargins(
            THEME.spacing_md,
            THEME.spacing_sm,
            THEME.spacing_md,
            THEME.spacing_sm,
        )
        platform_layout.addWidget(_label(cpu, bold=True))
        platform_layout.addWidget(_label(f"iGPU · {gpu} · {gpu_driver}", color=THEME.text_muted))
        platform_layout.addWidget(_label(f"NPU · {npu} · {npu_driver}", color=THEME.text_muted))
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
        self.mode_label = _label(self.policy_snapshot.label, size=THEME.font_semibold, color=THEME.npu, bold=True)
        self.density_label = _label(f"STREAMS · {self.policy_snapshot.density}", color=THEME.text_muted)
        self.source_label = _label(self.source_description, color=THEME.text_muted)
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.density_label)
        mode_layout.addWidget(self.source_label)
        header.addWidget(mode_panel, 3)
        root_layout.addLayout(header)

        main_row = QHBoxLayout()
        main_row.setSpacing(THEME.spacing_md)
        video_panel = _panel()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(THEME.spacing_sm, THEME.spacing_sm, THEME.spacing_sm, THEME.spacing_sm)
        self.main_canvas = FrameCanvas()
        video_layout.addWidget(self.main_canvas)
        main_row.addWidget(video_panel, 7)

        gauge_panel = QFrame()
        gauge_layout = QVBoxLayout(gauge_panel)
        gauge_layout.setContentsMargins(0, 0, 0, 0)
        gauge_layout.setSpacing(THEME.spacing_sm)
        self.gauges: dict[str, EngineGauge] = {}
        for engine in ("NPU", "GPU", "CPU"):
            gauge = EngineGauge(engine)
            self.gauges[engine] = gauge
            gauge_layout.addWidget(gauge, 1)
        main_row.addWidget(gauge_panel, 3)
        root_layout.addLayout(main_row, 1)

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
        self.tile_grid = QGridLayout()
        self.tile_grid.setSpacing(THEME.spacing_sm)
        tile_layout.addLayout(self.tile_grid, 1)
        self.tiles = [StreamTile(index) for index in range(max(DENSITIES))]
        bottom_row.addWidget(tile_panel, 7)

        metrics_panel = QFrame()
        metrics_layout = QGridLayout(metrics_panel)
        metrics_layout.setSpacing(THEME.spacing_sm)
        self.fps_tile = MetricTile("FPS")
        self.inference_tile = MetricTile("INFER p50")
        self.latency_tile = MetricTile("E2E p50")
        self.real_time_tile = MetricTile("REAL-TIME")
        self.rss_tile = MetricTile("RSS")
        self.detections_tile = MetricTile("DETECTIONS")
        for index, tile in enumerate(
            (
                self.fps_tile,
                self.inference_tile,
                self.latency_tile,
                self.real_time_tile,
                self.rss_tile,
                self.detections_tile,
            )
        ):
            metrics_layout.addWidget(tile, index // 3, index % 3)
        bottom_row.addWidget(metrics_panel, 3)
        self.bottom_panels = (tile_panel, metrics_panel)
        root_layout.addLayout(bottom_row, 0)

        self.status_label = _label(
            "N/G toggles · C mode · +/- density · F1 availability · F11 fullscreen · Q quit",
            color=THEME.text_muted,
        )
        root_layout.addWidget(self.status_label)
        self._update_policy_header()

    def start(self) -> None:
        self.telemetry_sampler.start()
        self.video_clock.start()
        self._reconcile_streams(self.policy_snapshot, time.monotonic(), initial=True)

    def _update_policy_header(self) -> None:
        self.policy_snapshot = self.policy.snapshot()
        self.mode_label.setText(self.policy_snapshot.label)
        self.density_label.setText(f"STREAMS · {self.policy_snapshot.density}")
        self.video_clock.set_target_fps(
            THEME.tile_display_fps if self.policy_snapshot.density >= 4 else 0.0
        )
        self.gauges["NPU"].set_disabled(self.policy.is_disabled("NPU"))
        self.gauges["GPU"].set_disabled(self.policy.is_disabled("GPU"))
        self.gauges["CPU"].set_disabled(False)
        self._layout_tiles()

    def _layout_tiles(self) -> None:
        density = self.policy_snapshot.density
        columns = min(density, 4)
        rows = (density + columns - 1) // columns
        maximum_height = (
            THEME.single_tile_row_max_height
            if rows == 1
            else THEME.multi_tile_row_max_height
        )
        for panel in self.bottom_panels:
            panel.setMaximumHeight(maximum_height)
        for tile in self.tiles:
            tile.hide()
            self.tile_grid.removeWidget(tile)
        for index in range(density):
            row = index // columns
            column = index % columns
            self.tile_grid.addWidget(self.tiles[index], row, column)
            self.tiles[index].show()
        for row in range(rows):
            self.tile_grid.setRowStretch(row, 1)
        for column in range(columns):
            self.tile_grid.setColumnStretch(column, 1)

    def _stream_device(self, index: int) -> str:
        return self.policy.device_for_stream(index, "NPU")

    def _compile_config_for(self, index: int) -> RunnerCompileConfig:
        target = self._stream_device(index)
        snapshot = self.policy_snapshot
        spread_multi = (
            snapshot.mode == DeviceMode.SPREAD and snapshot.density > 1
        )
        if not target.startswith("CPU"):
            return RunnerCompileConfig(
                performance_hint="THROUGHPUT" if spread_multi else "LATENCY"
            )
        cpu_is_sole_active_device = snapshot.active_devices == ("CPU",)
        if snapshot.mode != DeviceMode.SPREAD or cpu_is_sole_active_device:
            return RunnerCompileConfig(performance_hint="THROUGHPUT")
        physical_cores = psutil.cpu_count(logical=False) or 1
        cpu_num_streams = max(1, physical_cores // snapshot.density)
        return RunnerCompileConfig(
            performance_hint="THROUGHPUT",
            cpu_num_streams=cpu_num_streams,
        )

    def _reconcile_streams(
        self,
        snapshot: PolicySnapshot,
        requested_at: float,
        *,
        initial: bool = False,
    ) -> None:
        active_indices = set(range(snapshot.density))
        for index in list(self.stream_workers):
            if index not in active_indices:
                worker = self.stream_workers.pop(index)
                worker.requestInterruption()
                if not worker.wait(3000):
                    LOGGER.error("Stream %d did not stop within three seconds", index)
                worker.deleteLater()
                self.stream_info.pop(index, None)
                self.stream_detections.pop(index, ())
                self.stream_metrics.pop(index, None)

        self.policy_by_sequence[snapshot.sequence] = snapshot
        if not initial:
            self.pending_policy_acks[snapshot.sequence] = set(range(snapshot.density))
            self.policy_requested_at[snapshot.sequence] = time.time()
            self.policy_requested_monotonic[snapshot.sequence] = requested_at
        for index in sorted(active_indices):
            target = self._stream_device(index)
            compile_config = self._compile_config_for(index)
            if index in self.stream_workers:
                self.stream_workers[index].set_policy(
                    snapshot.sequence,
                    target,
                    requested_at,
                    compile_config,
                )
            else:
                worker = StreamWorker(
                    stream_index=index,
                    model_path=Path("models/yolo11n-fp16/yolo11n.xml").resolve(),
                    labels_path=Path("models/yolo11n-fp16/labels.txt").resolve(),
                    preprocess_config_path=Path("models/yolo11n-fp16/source_config.json").resolve(),
                    video_path=self.video_clock.video_path,
                    camera_index=self.video_clock.camera_index,
                    device=target,
                    compiled_store=self.compiled_store,
                    npu_duty_cycle=self.npu_duty_cycle,
                    policy_sequence=snapshot.sequence,
                    compile_config=compile_config,
                    parent=self,
                )
                worker.frame_ready.connect(self._on_stream_frame)
                worker.placement_ready.connect(self._on_placement_ready)
                worker.failed.connect(self._on_stream_failed)
                self.stream_workers[index] = worker
                worker.start()
        self._update_policy_header()

    def _apply_policy_change(self, changed: bool) -> None:
        self._update_policy_header()
        if not changed:
            return
        requested_at = time.monotonic()
        self._reconcile_streams(self.policy_snapshot, requested_at)

    def toggle_npu(self) -> None:
        before = self.policy.snapshot()
        after = self.policy.toggle_npu()
        self._apply_policy_change(after.sequence != before.sequence)

    def toggle_gpu(self) -> None:
        before = self.policy.snapshot()
        after = self.policy.toggle_gpu()
        self._apply_policy_change(after.sequence != before.sequence)

    def cycle_mode(self) -> None:
        before = self.policy.snapshot()
        after = self.policy.cycle_mode()
        self._apply_policy_change(after.sequence != before.sequence)

    def change_density(self, direction: int) -> None:
        before = self.policy.snapshot()
        after = self.policy.cycle_density(direction)
        self._apply_policy_change(after.sequence != before.sequence)

    def _on_video_frame(self, image: QImage, frame_index: int) -> None:
        self.latest_image = image
        self.latest_image_size = (image.width(), image.height())
        primary_detections = self.stream_detections.get(0, ())
        self.main_canvas.set_frame(image, primary_detections, self.latest_image_size)
        for index in range(self.policy_snapshot.density):
            self.tiles[index].update_stream(
                image,
                self.stream_detections.get(index, ()),
                self.latest_image_size,
                self.stream_info.get(index),
                self.stream_metrics.get(index),
            )

    def _on_stream_frame(self, index: int, detections: object, metrics: object) -> None:
        detection_tuple = tuple(detections)  # type: ignore[arg-type]
        stream_metrics = metrics  # type: ignore[assignment]
        self.stream_detections[index] = detection_tuple  # type: ignore[assignment]
        self.stream_metrics[index] = stream_metrics
        self.frame_history.append(stream_metrics)
        if self.latest_image is not None and not self.latest_image.isNull():
            self.tiles[index].update_stream(
                self.latest_image,
                self.stream_detections.get(index, ()),
                self.latest_image_size,
                self.stream_info.get(index),
                self.stream_metrics.get(index),
            )
        if index == 0:
            self.main_canvas.set_frame(
                self.latest_image,
                self.stream_detections.get(0, ()),
                self.latest_image_size,
            )
        self._update_pipeline_metrics()

    def _on_placement_ready(
        self,
        index: int,
        info: object,
        sequence: int,
        transition_ms: float,
    ) -> None:
        runner_info = info  # type: ignore[assignment]
        self.stream_info[index] = runner_info
        roots = {value.upper().split(".", 1)[0] for value in runner_info.execution_devices}
        sequence_policy = self.policy_by_sequence.get(sequence, self.policy_snapshot)
        if sequence_policy.mode == DeviceMode.OFF and roots != {"CPU"}:
            self._on_stream_failed(index, f"off policy placement was {runner_info.execution_devices}")
            return
        if not sequence_policy.npu_enabled and "NPU" in roots:
            self._on_stream_failed(index, "disabled NPU appeared in EXECUTION_DEVICES")
            return
        if not sequence_policy.gpu_enabled and "GPU" in roots:
            self._on_stream_failed(index, "disabled GPU appeared in EXECUTION_DEVICES")
            return
        pending = self.pending_policy_acks.get(sequence)
        if pending is not None:
            pending.discard(index)
            if not pending:
                self.pending_policy_acks.pop(sequence, None)
                self.policy_transitions.append(
                    {
                        "sequence": sequence,
                        "label": sequence_policy.label,
                        "all_streams_ms": transition_ms,
                        "requested_at": self.policy_requested_at.get(sequence),
                        "requested_monotonic": self.policy_requested_monotonic.get(sequence),
                        "at": time.time(),
                    }
                )
                self.status_label.setText(
                    f"Policy {sequence} live in {transition_ms:.0f} ms · "
                    f"{self.policy_snapshot.label} · F1 availability · Q quit"
                )

    def _on_telemetry(self, frame: TelemetryFrame) -> None:
        self.latest_telemetry = frame
        self.telemetry_history.append(frame)
        for metric in frame.engine_metrics:
            self.gauges[metric.engine].update_metric(metric)
        self.rss_tile.set_value(f"{frame.process_rss_bytes / (1024.0 * 1024.0):.0f} MiB")

    def _update_pipeline_metrics(self) -> None:
        metrics = list(self.stream_metrics.values())
        if not metrics:
            return
        recent = [
            item
            for item in self.frame_history
            if time.time() - item.captured_at <= THEME.telemetry_recent_seconds
        ]
        selected_fps = metrics[0].processing_fps
        selected_source = metrics[0].source_fps
        inference_values = [item.inference_ms for item in recent if item.stream_index == 0]
        e2e_values = [item.end_to_end_ms for item in recent if item.stream_index == 0]
        self.fps_tile.set_value(
            f"{selected_fps:.1f}" if selected_source is None else f"{selected_fps:.1f} / {selected_source:.1f}"
        )
        self.inference_tile.set_value(
            f"{np.percentile(inference_values, 50):.2f}" if inference_values else "—"
        )
        self.latency_tile.set_value(
            f"{np.percentile(e2e_values, 50):.2f}" if e2e_values else "—"
        )
        real_time = 0
        for item in metrics:
            if (
                item.source_fps
                and item.processing_fps
                >= item.source_fps * THEME.streams_real_time_fraction
            ):
                real_time += 1
        self.real_time_tile.set_value(f"{real_time} / {self.policy_snapshot.density}")
        self.detections_tile.set_value(str(sum(item.detection_count for item in metrics)))

    def _on_stream_failed(self, index: int, traceback_text: str) -> None:
        self.last_error = f"stream {index}: {traceback_text}"
        for pending in self.pending_policy_acks.values():
            pending.discard(index)
        LOGGER.error("Stream %d failed:\n%s", index, traceback_text)
        self.status_label.setText(f"STREAM {index} ERROR — see logs/session-*.log")
        self.status_label.setStyleSheet(f"color: {THEME.danger}; background: transparent;")

    def _on_video_failed(self, traceback_text: str) -> None:
        self.last_error = traceback_text
        LOGGER.error("Video clock failed:\n%s", traceback_text)
        self.status_label.setText("VIDEO SOURCE ERROR — see logs/session-*.log")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Q,):
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
        elif key == Qt.Key.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
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
        elif key == Qt.Key.Key_F1:
            dialog = AvailabilityDialog(self.availability, self)
            dialog.exec()
        else:
            super().keyPressEvent(event)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.video_clock.requestInterruption()
        for worker in self.stream_workers.values():
            worker.requestInterruption()
        for worker in self.stream_workers.values():
            if not worker.wait(5000):
                LOGGER.error("Stream worker did not stop within five seconds")
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
        summary = {
            "inference_ms": self._measurement_stats([item.inference_ms for item in frames]),
            "end_to_end_ms": self._measurement_stats([item.end_to_end_ms for item in frames]),
        }
        latest_streams = list(self.stream_metrics.values())
        strict_real_time = sum(
            bool(
                item.source_fps
                and item.processing_fps
                >= item.source_fps * THEME.streams_real_time_fraction
            )
            for item in latest_streams
        )
        real_time_equivalents = sum(
            min(1.0, item.processing_fps / item.source_fps)
            for item in latest_streams
            if item.source_fps
        )
        summary["streams"] = {
            "configured": self.policy_snapshot.density,
            "processed": len(latest_streams),
            "strict_streams_in_real_time": strict_real_time,
            "real_time_stream_equivalents": real_time_equivalents,
            "real_time_threshold_fraction": THEME.streams_real_time_fraction,
        }
        telemetry_max: dict[str, float | None] = {"NPU": None, "GPU": None, "CPU": None}
        for frame in self.telemetry_history:
            for metric in frame.engine_metrics:
                if metric.value_percent is not None:
                    telemetry_max[metric.engine] = max(
                        telemetry_max[metric.engine] or 0.0,
                        metric.value_percent,
                    )
        gauge_states = {
            engine: {
                "disabled_by_operator": gauge.disabled,
                "last_measurement": (
                    asdict(gauge.engine_metric) if gauge.engine_metric is not None else None
                ),
            }
            for engine, gauge in self.gauges.items()
        }
        return {
            "captured_at": time.time(),
            "started_at": self.started_at,
            "source": self.source_description,
            "policy": asdict(self.policy_snapshot),
            "density": self.policy_snapshot.density,
            "stream_metrics": {
                str(index): asdict(metrics) for index, metrics in self.stream_metrics.items()
            },
            "stream_placements": {
                str(index): asdict(info) for index, info in self.stream_info.items()
            },
            "policy_transitions": list(self.policy_transitions),
            "gauge_states": gauge_states,
            "availability": self.availability.to_dict(),
            "measurement_summary": summary,
            "telemetry_max_percent": telemetry_max,
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

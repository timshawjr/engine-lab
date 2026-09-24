"""OpenVINO compilation and single-request asynchronous inference."""

from __future__ import annotations

import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openvino as ov
import psutil
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from app.engine.stages import (
    load_labels,
    load_preprocess_config,
    model_input_size,
    postprocess_yolo,
    preprocess_yolo,
)
from app.telemetry.npu_fallback import NpuDutyCycle


@dataclass(frozen=True)
class RunnerInfo:
    requested_device: str
    execution_devices: tuple[str, ...]
    compile_ms: float
    model_input_shape: tuple[int, ...]
    device_gops: float | None
    cache_dir: str


class OpenVINOSingleRunner:
    """Compile one static-shape model and run one asynchronous request at a time."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        cache_dir: Path,
        npu_duty_cycle: NpuDutyCycle,
    ) -> None:
        self.model_path = model_path
        self.device = device.upper()
        if self.device not in {"NPU", "GPU", "CPU"}:
            raise ValueError(f"unsupported Phase 1 device: {device}")
        self.cache_dir = cache_dir
        self.npu_duty_cycle = npu_duty_cycle
        self._callback_output: np.ndarray | None = None
        self._callback_finished_at: float | None = None
        self._callback_error: BaseException | None = None
        self._sequence = 0
        self.info = self._compile()

    def _compile(self) -> RunnerInfo:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        core = ov.Core()
        core.set_property({"CACHE_DIR": str(self.cache_dir)})
        model = core.read_model(str(self.model_path))
        input_size = model_input_size(model)
        config: dict[str, Any] = {
            "CACHE_DIR": str(self.cache_dir),
            "PERFORMANCE_HINT": "LATENCY",
        }
        if self.device == "CPU":
            physical_cores = psutil.cpu_count(logical=False) or 1
            config["PERFORMANCE_HINT"] = "THROUGHPUT"
            # OpenVINO 2026.4 exposes the ov::streams::num property under its
            # supported runtime alias NUM_STREAMS.
            config["NUM_STREAMS"] = physical_cores

        started = time.perf_counter()
        compiled = core.compile_model(model, self.device, config)
        compile_ms = (time.perf_counter() - started) * 1000.0
        raw_execution_devices = compiled.get_property("EXECUTION_DEVICES")
        if isinstance(raw_execution_devices, str):
            execution_devices = (raw_execution_devices,)
        else:
            execution_devices = tuple(str(value) for value in raw_execution_devices)
        roots = {value.upper().split(".", 1)[0] for value in execution_devices}
        if roots != {self.device}:
            raise RuntimeError(
                f"placement assertion failed: requested {self.device}, "
                f"EXECUTION_DEVICES={execution_devices}"
            )

        try:
            raw_gops = compiled.get_property("DEVICE_GOPS")
            device_gops = float(raw_gops) if raw_gops is not None else None
        except Exception:
            device_gops = None

        self._compiled = compiled
        self._queue = ov.AsyncInferQueue(compiled)
        self._queue.set_callback(self._callback)
        self._model_height, self._model_width = input_size
        return RunnerInfo(
            requested_device=self.device,
            execution_devices=execution_devices,
            compile_ms=compile_ms,
            model_input_shape=tuple(
                int(dimension.get_length())
                for dimension in model.inputs[0].get_partial_shape()
            ),
            device_gops=device_gops,
            cache_dir=str(self.cache_dir),
        )

    def _callback(self, request: Any, userdata: int) -> None:
        try:
            tensor = request.get_output_tensor(0)
            self._callback_output = np.array(tensor.data, copy=True)
            self._callback_finished_at = time.perf_counter()
            self._callback_error = None
        except BaseException as exc:
            self._callback_output = None
            self._callback_finished_at = time.perf_counter()
            self._callback_error = exc

    def infer(self, tensor: np.ndarray) -> tuple[np.ndarray, float]:
        expected = (1, 3, self._model_height, self._model_width)
        if tensor.shape != expected or tensor.dtype != np.float32:
            raise ValueError(f"input must be float32 {expected}, got {tensor.shape} {tensor.dtype}")
        if not tensor.flags.c_contiguous:
            tensor = np.ascontiguousarray(tensor)
        self._callback_output = None
        self._callback_finished_at = None
        self._callback_error = None
        self._sequence += 1
        started = time.perf_counter()
        self._queue.start_async(tensor, self._sequence)
        self._queue.wait_all()
        finished = self._callback_finished_at or time.perf_counter()
        duration_seconds = max(0.0, finished - started)
        if self._callback_error is not None:
            raise RuntimeError("OpenVINO inference callback failed") from self._callback_error
        if self._callback_output is None:
            raise RuntimeError("OpenVINO inference completed without an output tensor")
        if self.device == "NPU":
            self.npu_duty_cycle.record_inference(duration_seconds, ended_at=time.monotonic())
        return self._callback_output, duration_seconds * 1000.0

    @property
    def model_width(self) -> int:
        return self._model_width

    @property
    def model_height(self) -> int:
        return self._model_height


@dataclass(frozen=True)
class InferenceFrameMetrics:
    frame_index: int
    captured_at: float
    source_fps: float | None
    processing_fps: float
    decode_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    end_to_end_ms: float
    detection_count: int
    frame_width: int
    frame_height: int


class _ProcessingRate:
    def __init__(self, window_seconds: float = 2.0) -> None:
        self.window_seconds = window_seconds
        self._times: deque[float] = deque()

    def sample(self, now: float) -> float:
        cutoff = now - self.window_seconds
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        elapsed = max(self._times[-1] - self._times[0], 1e-9)
        return (len(self._times) - 1) / elapsed


class InferenceThread(QThread):
    frame_ready = Signal(object, object, object)
    runner_ready = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        preprocess_config_path: Path,
        video_path: Path | None,
        camera_index: int | None,
        device: str,
        cache_dir: Path,
        npu_duty_cycle: NpuDutyCycle,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.model_path = model_path
        self.labels_path = labels_path
        self.preprocess_config_path = preprocess_config_path
        self.video_path = video_path
        self.camera_index = camera_index
        self.device = device.upper()
        self.cache_dir = cache_dir
        self.npu_duty_cycle = npu_duty_cycle

    def _open_capture(self) -> tuple[cv2.VideoCapture, bool]:
        if self.camera_index is not None:
            capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                raise RuntimeError(f"could not open camera index {self.camera_index}")
            return capture, False
        if self.video_path is None:
            raise ValueError("loop source requires a video path")
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {self.video_path}")
        return capture, True

    @staticmethod
    def _to_qimage(frame: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        rgb_bytes = rgb.tobytes()
        return QImage(
            rgb_bytes,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()

    def run(self) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            labels = load_labels(self.labels_path)
            preprocess_config = load_preprocess_config(self.preprocess_config_path)
            runner = OpenVINOSingleRunner(
                model_path=self.model_path,
                device=self.device,
                cache_dir=self.cache_dir,
                npu_duty_cycle=self.npu_duty_cycle,
            )
            self.runner_ready.emit(runner.info)
            capture, loop = self._open_capture()
            source_fps_raw = float(capture.get(cv2.CAP_PROP_FPS))
            source_fps = source_fps_raw if source_fps_raw > 0 else None
            frame_index = 0
            rate = _ProcessingRate()
            next_frame_deadline = time.perf_counter()

            while not self.isInterruptionRequested():
                end_to_end_started = time.perf_counter()
                decode_started = time.perf_counter()
                ok, frame = capture.read()
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
                if not ok or frame is None:
                    if not loop:
                        break
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"video loop could not restart: {self.video_path}")
                frame_index += 1

                preprocessed = preprocess_yolo(
                    frame,
                    model_width=runner.model_width,
                    model_height=runner.model_height,
                    metadata=preprocess_config,
                )
                raw_output, inference_ms = runner.infer(preprocessed.tensor)
                postprocess_started = time.perf_counter()
                detections = postprocess_yolo(
                    raw_output,
                    labels=labels,
                    transform=preprocessed.transform,
                )
                postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
                completed_at = time.perf_counter()
                metrics = InferenceFrameMetrics(
                    frame_index=frame_index,
                    captured_at=time.time(),
                    source_fps=source_fps,
                    processing_fps=rate.sample(completed_at),
                    decode_ms=decode_ms,
                    preprocess_ms=preprocessed.duration_ms,
                    inference_ms=inference_ms,
                    postprocess_ms=postprocess_ms,
                    end_to_end_ms=(completed_at - end_to_end_started) * 1000.0,
                    detection_count=len(detections),
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                )
                self.frame_ready.emit(self._to_qimage(frame), detections, metrics)

                if source_fps is not None:
                    next_frame_deadline += 1.0 / source_fps
                    remaining = next_frame_deadline - time.perf_counter()
                    if remaining > 0:
                        self.msleep(max(1, round(remaining * 1000.0)))
                    else:
                        next_frame_deadline = time.perf_counter()
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if capture is not None:
                capture.release()

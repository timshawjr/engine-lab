"""OpenVINO compilation and single-request asynchronous inference."""

from __future__ import annotations

import queue
import threading
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
    inspect_model_input,
    load_preprocess_config,
    postprocess_yolo,
    preprocess_yolo,
)
from app.telemetry.npu_fallback import NpuDutyCycle

@dataclass(frozen=True)
class RunnerCompileConfig:
    performance_hint: str = "LATENCY"
    cpu_num_streams: int | None = None

    def __post_init__(self) -> None:
        if self.performance_hint not in {"LATENCY", "THROUGHPUT"}:
            raise ValueError(f"unsupported performance hint: {self.performance_hint}")
        if self.cpu_num_streams is not None and self.cpu_num_streams < 1:
            raise ValueError("cpu_num_streams must be positive")


@dataclass(frozen=True)
class RunnerInfo:
    requested_device: str
    execution_devices: tuple[str, ...]
    compile_ms: float
    model_input_shape: tuple[int, ...]
    device_gops: float | None
    cache_dir: str
    performance_hint: str
    cpu_num_streams: int | None
    input_layout: str


class CompiledModelStore:
    """Thread-safe compile cache shared by Phase 2 stream workers."""

    def __init__(self, model_path: Path, cache_dir: Path) -> None:
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._core = ov.Core()
        self._model = self._core.read_model(str(model_path))
        self._input_spec = inspect_model_input(self._model)
        self._model_height = self._input_spec.height
        self._model_width = self._input_spec.width
        self._model_input_shape = self._input_spec.shape
        self._lock = threading.Lock()
        # Concurrent infer() calls on one CompiledModel deadlock inside OpenVINO's
        # data_dispatcher, which is shared per compiled model rather than per request. The
        # captured hang showed several streams blocked in data_dispatcher._data_dispatch at
        # once. Serialising per store keeps streams on different devices independent while
        # making same-device streams safe.
        self._infer_lock = threading.Lock()
        self._entries: dict[tuple[str, RunnerCompileConfig], tuple[Any, RunnerInfo]] = {}

    @staticmethod
    def _normalize_device(device: str) -> str:
        normalized = device.strip().upper()
        if normalized in {"NPU", "GPU", "CPU"}:
            return normalized
        if normalized.startswith("AUTO:"):
            devices = [item.strip() for item in normalized.split(":", 1)[1].split(",") if item.strip()]
            if not devices or any(item not in {"NPU", "GPU", "CPU"} for item in devices):
                raise ValueError(f"invalid AUTO device priority string: {device}")
            return "AUTO:" + ",".join(devices)
        raise ValueError(f"unsupported OpenVINO device string: {device}")

    def get(
        self,
        device: str,
        compile_config: RunnerCompileConfig | None = None,
    ) -> tuple[Any, RunnerInfo]:
        device = self._normalize_device(device)
        compile_config = compile_config or RunnerCompileConfig()
        key = (device, compile_config)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                return cached
            config: dict[str, Any] = {
                "PERFORMANCE_HINT": compile_config.performance_hint,
            }
            # The GPU blob save/reload round-trip is lossy on this platform:
            # measured on crossroad-1016, a fresh GPU compile reports 52
            # detections over 5 gov-entry frames (max conf 0.959) while
            # reloading the blob it just wrote reports 5 (max conf 0.443).
            # CPU and NPU round-trips are lossless (52 in every mode), so the
            # cache stays enabled for those devices and GPU compiles run
            # without one.
            effective_cache_dir = str(self.cache_dir)
            if device == "GPU":
                effective_cache_dir = ""
            else:
                config["CACHE_DIR"] = effective_cache_dir
            cpu_num_streams: int | None = None
            if device == "CPU":
                physical_cores = psutil.cpu_count(logical=False) or 1
                config["PERFORMANCE_HINT"] = "THROUGHPUT"
                cpu_num_streams = compile_config.cpu_num_streams or physical_cores
                config["NUM_STREAMS"] = cpu_num_streams
            started = time.perf_counter()
            compiled = self._core.compile_model(self._model, device, config)
            compile_ms = (time.perf_counter() - started) * 1000.0
            raw_execution_devices = compiled.get_property("EXECUTION_DEVICES")
            if isinstance(raw_execution_devices, str):
                execution_devices = (raw_execution_devices,)
            else:
                execution_devices = tuple(str(value) for value in raw_execution_devices)
            roots = {value.upper().split(".", 1)[0] for value in execution_devices}
            if device.startswith("AUTO:"):
                allowed = set(device.split(":", 1)[1].split(","))
                if not roots or not roots.issubset(allowed):
                    raise RuntimeError(
                        f"placement assertion failed: requested {device}, "
                        f"EXECUTION_DEVICES={execution_devices}"
                    )
            elif roots != {device}:
                raise RuntimeError(
                    f"placement assertion failed: requested {device}, "
                    f"EXECUTION_DEVICES={execution_devices}"
                )
            try:
                raw_gops = compiled.get_property("DEVICE_GOPS")
                device_gops = float(raw_gops) if raw_gops is not None else None
            except Exception:
                device_gops = None
            if device == "CPU":
                try:
                    cpu_num_streams = int(compiled.get_property("NUM_STREAMS"))
                except Exception as exc:
                    raise RuntimeError(
                        "compiled CPU model did not expose NUM_STREAMS"
                    ) from exc
            info = RunnerInfo(
                requested_device=device,
                execution_devices=execution_devices,
                compile_ms=compile_ms,
                model_input_shape=self._model_input_shape,
                device_gops=device_gops,
                cache_dir=effective_cache_dir,
                performance_hint=config["PERFORMANCE_HINT"],
                cpu_num_streams=cpu_num_streams,
                input_layout=self._input_spec.layout,
            )
            self._entries[key] = (compiled, info)
            return compiled, info


class OpenVINOSingleRunner:
    """Compile one static-shape model and run one asynchronous request at a time."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        cache_dir: Path,
        npu_duty_cycle: NpuDutyCycle,
        compiled_store: CompiledModelStore | None = None,
        compile_config: RunnerCompileConfig | None = None,
    ) -> None:
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.npu_duty_cycle = npu_duty_cycle
        self.compiled_store = compiled_store or CompiledModelStore(model_path, cache_dir)
        self.compile_config = compile_config or RunnerCompileConfig()
        self.device = self.compiled_store._normalize_device(device)
        self.input_spec = self.compiled_store._input_spec
        self.info = self._compile()
        self.device = self.info.requested_device
        self._execution_roots = {
            value.upper().split(".", 1)[0] for value in self.info.execution_devices
        }
        # The app-measured NPU duty cycle may only count work that was actually bound to the
        # NPU. An AUTO-compiled model reports its whole priority list in EXECUTION_DEVICES, so
        # testing membership of that set would credit CPU/GPU inferences to the NPU gauge - a
        # number that was never measured (SPEC 6.3).
        self._npu_bound = self.device.strip().upper() == "NPU"

    def _compile(self) -> RunnerInfo:
        compiled, info = self.compiled_store.get(self.device, self.compile_config)
        self._compiled = compiled
        self._output_count = len(compiled.outputs)
        # A synchronous InferRequest, not an AsyncInferQueue.
        #
        # AsyncInferQueue routes every call through OpenVINO's *Python* data_dispatcher
        # wrapper, which holds the GIL while it works. With several streams calling it at
        # once the GIL is contended hard enough that a sibling thread cannot run a plain
        # numpy call: the captured hang showed three workers blocked inside
        # data_dispatcher._data_dispatch and a fourth starved in np.ascontiguousarray.
        # Windows then kills the process as an Application Hang. A synchronous InferRequest
        # is a direct C call that releases the GIL while it runs, so the threads cannot
        # deadlock against each other.
        self._infer_request = compiled.create_infer_request()
        self._model_height = self.input_spec.height
        self._model_width = self.input_spec.width
        return info

    def infer_all(
        self,
        tensor: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], float]:
        expected = tuple(self.input_spec.shape)
        if tensor.shape != expected or tensor.dtype != np.float32:
            raise ValueError(f"input must be float32 {expected}, got {tensor.shape} {tensor.dtype}")
        if not tensor.flags.c_contiguous:
            tensor = np.ascontiguousarray(tensor)
        started = time.perf_counter()
        # Serialise inference per compiled model. OpenVINO's data_dispatcher is shared by
        # every request created from one CompiledModel, and concurrent infer() calls on it
        # deadlock - the captured hang had several streams inside data_dispatcher._data_dispatch
        # simultaneously. Streams on different devices use different stores and stay parallel.
        with self.compiled_store._infer_lock:
            self._infer_request.infer({0: tensor})
            outputs = tuple(
                np.array(self._infer_request.get_output_tensor(index).data, copy=True)
                for index in range(self._output_count)
            )
        finished = time.perf_counter()
        duration_seconds = max(0.0, finished - started)
        if not outputs:
            raise RuntimeError("OpenVINO inference completed without an output tensor")
        if self._npu_bound:
            self.npu_duty_cycle.record_inference(duration_seconds, ended_at=time.monotonic())
        return outputs, duration_seconds * 1000.0

    def infer(self, tensor: np.ndarray) -> tuple[np.ndarray, float]:
        outputs, duration_ms = self.infer_all(tensor)
        return outputs[0], duration_ms

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


@dataclass(frozen=True)
class StreamFrameMetrics:
    stream_index: int
    frame_index: int
    captured_at: float
    requested_device: str
    execution_devices: tuple[str, ...]
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


class RetailVideoClock(QThread):
    """Decodes the visible source independently so recompiles never freeze the HUD."""

    frame_ready = Signal(object, int)
    failed = Signal(str)

    #: Frame within the current file that is on screen. Distinct from the
    #: cumulative counter emitted with frame_ready, which climbs past the file
    #: length because the clip loops. Anything syncing a second capture to this
    #: one needs the file position, not the counter.
    last_position: int = -1

    def __init__(
        self,
        video_path: Path | None,
        camera_index: int | None,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.camera_index = camera_index
        self._target_fps = 0.0
        self._path_lock = threading.Lock()
        self._requested_video_path = video_path

    def set_video_path(self, video_path: Path | None) -> None:
        with self._path_lock:
            self._requested_video_path = video_path

    def set_target_fps(self, target_fps: float) -> None:
        self._target_fps = max(0.0, float(target_fps))

    def run(self) -> None:
        capture: cv2.VideoCapture | None = None
        current_video_path: Path | None = None
        source_fps: float | None = None
        try:
            next_deadline = time.perf_counter()
            frame_index = 0
            while not self.isInterruptionRequested():
                with self._path_lock:
                    requested_video_path = self._requested_video_path
                if self.camera_index is not None:
                    if capture is None or not capture.isOpened():
                        if capture is not None:
                            capture.release()
                        capture = cv2.VideoCapture(self.camera_index)
                        source_fps_raw = float(capture.get(cv2.CAP_PROP_FPS))
                        source_fps = source_fps_raw if source_fps_raw > 0 else None
                else:
                    if requested_video_path is None:
                        raise ValueError("loop source requires a video path")
                    if capture is not None and requested_video_path != current_video_path:
                        capture.release()
                        capture = None
                    if capture is None:
                        capture = cv2.VideoCapture(str(requested_video_path))
                        current_video_path = requested_video_path
                        source_fps_raw = float(capture.get(cv2.CAP_PROP_FPS))
                        source_fps = source_fps_raw if source_fps_raw > 0 else None
                if capture is None or not capture.isOpened():
                    raise RuntimeError("video clock could not open the source")
                ok, frame = capture.read()
                if not ok or frame is None:
                    if self.camera_index is None:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = capture.read()
                    if not ok or frame is None:
                        if self.camera_index is None:
                            raise RuntimeError("video loop could not restart")
                        self.msleep(10)
                        next_deadline = time.perf_counter()
                        continue
                frame_index += 1
                # Publish the file position, not the cumulative counter. The
                # clip loops, so frame_index climbs past the file length while a
                # capture wraps at it; anything seeking to frame_index would land
                # past the end of the file.
                try:
                    self.last_position = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                except Exception:  # noqa: BLE001 - backend-dependent
                    self.last_position = -1
                self.frame_ready.emit(InferenceThread._to_qimage(frame), frame_index)
                effective_fps = source_fps
                if self._target_fps > 0 and source_fps:
                    effective_fps = min(source_fps, self._target_fps)
                if effective_fps:
                    next_deadline += 1.0 / effective_fps
                    remaining = next_deadline - time.perf_counter()
                    if remaining > 0:
                        self.msleep(max(1, round(remaining * 1000.0)))
                    else:
                        next_deadline = time.perf_counter()
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if capture is not None:
                capture.release()


class StreamWorker(QThread):
    """One independent decode + AsyncInferQueue stream with live policy swaps."""

    frame_ready = Signal(int, object, object)
    placement_ready = Signal(int, object, int, float)
    failed = Signal(int, str)

    def __init__(
        self,
        stream_index: int,
        model_path: Path,
        labels_path: Path,
        preprocess_config_path: Path,
        video_path: Path | None,
        camera_index: int | None,
        device: str,
        compiled_store: CompiledModelStore,
        npu_duty_cycle: NpuDutyCycle,
        policy_sequence: int,
        compile_config: RunnerCompileConfig,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.stream_index = stream_index
        self.model_path = model_path
        self.labels_path = labels_path
        self.preprocess_config_path = preprocess_config_path
        self.video_path = video_path
        self.camera_index = camera_index
        self.compiled_store = compiled_store
        self.npu_duty_cycle = npu_duty_cycle
        self._policy_updates: queue.Queue[
            tuple[int, str, float, RunnerCompileConfig]
        ] = queue.Queue()
        self._pending_sequence = policy_sequence
        self._pending_device = device
        self._pending_requested_at = time.monotonic()
        self._pending_compile_config = compile_config

    def set_policy(
        self,
        sequence: int,
        device: str,
        requested_at: float,
        compile_config: RunnerCompileConfig,
    ) -> None:
        while True:
            try:
                self._policy_updates.get_nowait()
            except queue.Empty:
                break
        self._pending_sequence = sequence
        self._pending_device = device
        self._pending_requested_at = requested_at
        self._pending_compile_config = compile_config
        self._policy_updates.put((sequence, device, requested_at, compile_config))

    def _open_capture(self) -> tuple[cv2.VideoCapture, bool]:
        if self.camera_index is not None:
            capture = cv2.VideoCapture(self.camera_index)
            loop = False
        else:
            if self.video_path is None:
                raise ValueError("stream loop source requires a video path")
            capture = cv2.VideoCapture(str(self.video_path))
            loop = True
        if not capture.isOpened():
            raise RuntimeError(f"stream {self.stream_index} could not open source")
        return capture, loop

    def run(self) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            labels = load_labels(self.labels_path)
            preprocess_config = load_preprocess_config(self.preprocess_config_path)
            runner = OpenVINOSingleRunner(
                self.model_path,
                self._pending_device,
                self.compiled_store.cache_dir,
                self.npu_duty_cycle,
                compiled_store=self.compiled_store,
                compile_config=self._pending_compile_config,
            )
            self.placement_ready.emit(
                self.stream_index,
                runner.info,
                self._pending_sequence,
                (time.monotonic() - self._pending_requested_at) * 1000.0,
            )
            capture, loop = self._open_capture()
            source_fps_raw = float(capture.get(cv2.CAP_PROP_FPS))
            source_fps = source_fps_raw if source_fps_raw > 0 else None
            frame_index = 0
            rate = _ProcessingRate()
            next_deadline = time.perf_counter()
            while not self.isInterruptionRequested():
                while True:
                    try:
                        sequence, target_device, requested_at, compile_config = (
                            self._policy_updates.get_nowait()
                        )
                    except queue.Empty:
                        break
                    if (
                        target_device != runner.device
                        or compile_config != runner.compile_config
                    ):
                        replacement = OpenVINOSingleRunner(
                            self.model_path,
                            target_device,
                            self.compiled_store.cache_dir,
                            self.npu_duty_cycle,
                            compiled_store=self.compiled_store,
                            compile_config=compile_config,
                        )
                        runner = replacement
                    self.placement_ready.emit(
                        self.stream_index,
                        runner.info,
                        sequence,
                        (time.monotonic() - requested_at) * 1000.0,
                    )

                end_to_end_started = time.perf_counter()
                decode_started = time.perf_counter()
                ok, frame = capture.read()
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
                if not ok or frame is None:
                    if loop:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = capture.read()
                    if not ok or frame is None:
                        if loop:
                            raise RuntimeError(
                                f"stream {self.stream_index} loop could not restart"
                            )
                        self.msleep(10)
                        next_deadline = time.perf_counter()
                        continue
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
                metrics = StreamFrameMetrics(
                    stream_index=self.stream_index,
                    frame_index=frame_index,
                    captured_at=time.time(),
                    requested_device=runner.device,
                    execution_devices=runner.info.execution_devices,
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
                self.frame_ready.emit(self.stream_index, detections, metrics)
                if source_fps is not None:
                    next_deadline += 1.0 / source_fps
                    remaining = next_deadline - time.perf_counter()
                    if remaining > 0:
                        self.msleep(max(1, round(remaining * 1000.0)))
                    else:
                        next_deadline = time.perf_counter()
        except Exception:
            self.failed.emit(self.stream_index, traceback.format_exc())
        finally:
            if capture is not None:
                capture.release()

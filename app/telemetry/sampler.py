"""5 Hz background telemetry publisher with an immutable UI frame."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
from PySide6.QtCore import QObject, Signal

from app.telemetry.cpu import CPUTelemetry
from app.telemetry.devices_win import (
    enumerate_compute_accelerators,
    enumerate_display_adapters,
    is_intel,
    is_npu,
)
from app.telemetry.npu_fallback import NpuDutyCycle
from app.telemetry.pdh import GPUEngineCounter, PDHError, classify_device


@dataclass(frozen=True)
class EngineMetric:
    engine: str
    value_percent: float | None
    source: str
    state: str
    detail: str


@dataclass(frozen=True)
class TelemetryFrame:
    sampled_at: float
    engine_metrics: tuple[EngineMetric, ...]
    cpu_per_core_percent: tuple[float, ...]
    process_rss_bytes: int
    pdh_error: str | None

    def metric(self, engine: str) -> EngineMetric | None:
        return next((item for item in self.engine_metrics if item.engine == engine), None)


class TelemetrySampler(QObject):
    frame_ready = Signal(object)

    def __init__(
        self,
        telemetry_map_path: Path,
        npu_duty_cycle: NpuDutyCycle,
        interval_seconds: float = 0.2,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.telemetry_map_path = telemetry_map_path
        self.npu_duty_cycle = npu_duty_cycle
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry sampler is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="engine-lab-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout_seconds)
        self._thread = None

    def _load_map(self) -> dict[str, Any]:
        try:
            return json.loads(self.telemetry_map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _device_luids(self) -> tuple[set[int], set[int]]:
        npu_luids: set[int] = set()
        gpu_luids: set[int] = set()
        try:
            for device in enumerate_compute_accelerators():
                if is_intel(device) and is_npu(device) and device.luid is not None:
                    npu_luids.add(device.luid)
        except Exception:
            pass
        try:
            for device in enumerate_display_adapters():
                if is_intel(device) and device.luid is not None:
                    gpu_luids.add(device.luid)
        except Exception:
            pass
        return npu_luids, gpu_luids

    def _run(self) -> None:
        cpu = CPUTelemetry()
        telemetry_map = self._load_map()
        npu_pdh_configured = (
            telemetry_map.get("counter_source") == "pdh_gpu_engine"
            and any(
                instance.get("device") == "NPU"
                for instance in telemetry_map.get("instances", [])
            )
        )
        npu_luids, gpu_luids = self._device_luids()
        pdh_counter: GPUEngineCounter | None = None
        pdh_error: str | None = None
        if npu_pdh_configured or telemetry_map.get("counter_source") == "pdh_gpu_engine":
            try:
                pdh_counter = GPUEngineCounter()
                pdh_counter.collect()
            except Exception as exc:
                pdh_error = f"{type(exc).__name__}: {exc}"
                npu_pdh_configured = False

        process = psutil.Process(os.getpid())
        next_sample = time.monotonic()
        while not self._stop_event.is_set():
            try:
                if pdh_counter is not None:
                    samples = pdh_counter.sample_utilization()
                else:
                    samples = []
                npu_values = [
                    sample.value
                    for sample in samples
                    if sample.instance.pid == os.getpid()
                    and classify_device(
                        sample.instance,
                        telemetry_map.get("npu_phys_ids", []),
                        npu_luids,
                        gpu_luids,
                        telemetry_map.get("gpu_phys_ids", []),
                    )
                    == "NPU"
                ]
                gpu_values = [
                    sample.value
                    for sample in samples
                    if sample.instance.pid == os.getpid()
                    and sample.instance.engtype
                    and classify_device(
                        sample.instance,
                        telemetry_map.get("npu_phys_ids", []),
                        npu_luids,
                        gpu_luids,
                        telemetry_map.get("gpu_phys_ids", []),
                    )
                    == "GPU"
                ]

                if npu_values:
                    npu_metric = EngineMetric(
                        engine="NPU",
                        value_percent=max(npu_values),
                        source="PDH GPU Engine · this process · busiest Neural engine",
                        state="ACTIVE" if max(npu_values) > 0 else "MEASURED IDLE",
                        detail=f"{len(npu_values)} matching process instance(s)",
                    )
                elif npu_pdh_configured:
                    npu_metric = EngineMetric(
                        engine="NPU",
                        value_percent=None,
                        source="PDH GPU Engine · this process",
                        state="NO SAMPLE",
                        detail="counter is available; no NPU instance for this process in this sample",
                    )
                else:
                    fallback = self.npu_duty_cycle.sample()
                    npu_metric = EngineMetric(
                        engine="NPU",
                        value_percent=fallback.value_percent,
                        source="NPU busy (app-measured)",
                        state="APP MEASURED",
                        detail=(
                            f"{fallback.sample_count} inference(s), "
                            f"{fallback.inference_seconds * 1000.0:.1f} ms in "
                            f"{fallback.window_seconds:.1f} s"
                        ),
                    )

                if gpu_values:
                    gpu_metric = EngineMetric(
                        engine="GPU",
                        value_percent=max(gpu_values),
                        source="PDH GPU Engine · this process · busiest engine",
                        state="ACTIVE" if max(gpu_values) > 0 else "MEASURED IDLE",
                        detail=f"{len(gpu_values)} matching process engine(s)",
                    )
                else:
                    gpu_metric = EngineMetric(
                        engine="GPU",
                        value_percent=None,
                        source="PDH GPU Engine · this process",
                        state="NO COUNTER" if pdh_counter is None else "NO SAMPLE",
                        detail=pdh_error or "no GPU engine instance for this process in this sample",
                    )

                cpu_reading = cpu.sample()
                frame = TelemetryFrame(
                    sampled_at=time.time(),
                    engine_metrics=(
                        npu_metric,
                        gpu_metric,
                        EngineMetric(
                            engine="CPU",
                            value_percent=cpu_reading.total_percent,
                            source="psutil · system CPU",
                            state="ACTIVE" if cpu_reading.total_percent > 0 else "MEASURED IDLE",
                            detail=(
                                f"{cpu_reading.physical_cores or '?'} physical / "
                                f"{cpu_reading.logical_cores or '?'} logical cores"
                            ),
                        ),
                    ),
                    cpu_per_core_percent=cpu_reading.per_core_percent,
                    process_rss_bytes=int(process.memory_info().rss),
                    pdh_error=pdh_error,
                )
                self.frame_ready.emit(frame)
            except Exception as exc:
                pdh_error = f"{type(exc).__name__}: {exc}"
                fallback = self.npu_duty_cycle.sample()
                fallback_frame = TelemetryFrame(
                    sampled_at=time.time(),
                    engine_metrics=(
                        EngineMetric(
                            "NPU",
                            fallback.value_percent,
                            "NPU busy (app-measured)",
                            "APP MEASURED",
                            f"PDH error: {pdh_error}",
                        ),
                        EngineMetric("GPU", None, "PDH GPU Engine", "NO COUNTER", pdh_error),
                        EngineMetric("CPU", None, "psutil · system CPU", "NO SAMPLE", pdh_error),
                    ),
                    cpu_per_core_percent=(),
                    process_rss_bytes=0,
                    pdh_error=pdh_error,
                )
                self.frame_ready.emit(fallback_frame)

            next_sample += self.interval_seconds
            delay = next_sample - time.monotonic()
            if delay > 0:
                self._stop_event.wait(delay)
            else:
                next_sample = time.monotonic()
        if pdh_counter is not None:
            try:
                pdh_counter.close()
            except PDHError:
                pass

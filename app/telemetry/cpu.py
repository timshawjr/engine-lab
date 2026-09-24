"""Measured CPU telemetry backed by psutil."""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class CPUReading:
    total_percent: float
    per_core_percent: tuple[float, ...]
    physical_cores: int | None
    logical_cores: int | None


class CPUTelemetry:
    def __init__(self) -> None:
        psutil.cpu_percent(interval=None)
        self._last_per_core = psutil.cpu_percent(interval=None, percpu=True)
        self._last_total = psutil.cpu_percent(interval=None)

    def sample(self) -> CPUReading:
        self._last_total = psutil.cpu_percent(interval=None)
        self._last_per_core = tuple(psutil.cpu_percent(interval=None, percpu=True))
        return CPUReading(
            total_percent=float(self._last_total),
            per_core_percent=self._last_per_core,
            physical_cores=psutil.cpu_count(logical=False),
            logical_cores=psutil.cpu_count(logical=True),
        )

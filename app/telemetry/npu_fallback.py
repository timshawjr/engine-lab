"""Application-measured NPU duty cycle used only when PDH is unavailable."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class NpuDutyCycleReading:
    value_percent: float
    inference_seconds: float
    window_seconds: float
    sample_count: int


class NpuDutyCycle:
    def __init__(self, window_seconds: float = 5.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self._events: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()
        self._started_at = time.monotonic()

    def record_inference(self, duration_seconds: float, ended_at: float | None = None) -> None:
        if duration_seconds < 0:
            raise ValueError("inference duration cannot be negative")
        timestamp = time.monotonic() if ended_at is None else ended_at
        with self._lock:
            self._events.append((timestamp, duration_seconds))

    def sample(self, now: float | None = None) -> NpuDutyCycleReading:
        timestamp = time.monotonic() if now is None else now
        cutoff = timestamp - self.window_seconds
        with self._lock:
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()
            inference_seconds = sum(duration for _, duration in self._events)
            sample_count = len(self._events)
        window_start = max(self._started_at, cutoff)
        elapsed = max(timestamp - window_start, 1e-9)
        value = min(100.0, inference_seconds / elapsed * 100.0)
        return NpuDutyCycleReading(
            value_percent=value,
            inference_seconds=inference_seconds,
            window_seconds=elapsed,
            sample_count=sample_count,
        )

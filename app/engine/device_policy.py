"""Phase 2 device-mode, toggle, and stream-assignment policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


DENSITIES = (1, 2, 4, 6, 8)
DEVICE_ORDER = ("NPU", "GPU", "CPU")


class DeviceMode(str, Enum):
    AUTO = "auto"
    SPREAD = "spread"
    NPU_ONLY = "npu_only"
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"
    SPLIT = "split"
    OFF = "off"


MODE_CYCLE = (
    DeviceMode.AUTO,
    DeviceMode.SPREAD,
    DeviceMode.NPU_ONLY,
    DeviceMode.GPU_ONLY,
    DeviceMode.CPU_ONLY,
    DeviceMode.SPLIT,
)

_MODE_REQUIREMENTS = {
    DeviceMode.NPU_ONLY: {"NPU"},
    DeviceMode.GPU_ONLY: {"GPU"},
}


@dataclass(frozen=True)
class PolicySnapshot:
    sequence: int
    mode: DeviceMode
    npu_enabled: bool
    gpu_enabled: bool
    density: int
    available_devices: tuple[str, ...]
    active_devices: tuple[str, ...]
    label: str


class DevicePolicy:
    def __init__(
        self,
        available_devices: Iterable[str] = DEVICE_ORDER,
        mode: DeviceMode = DeviceMode.SPREAD,
        density: int = 1,
    ) -> None:
        normalized = {
            device.upper() for device in available_devices if device.upper() in DEVICE_ORDER
        }
        if "CPU" not in normalized:
            normalized.add("CPU")
        self.available_devices = tuple(
            device for device in DEVICE_ORDER if device in normalized
        )
        required = _MODE_REQUIREMENTS.get(mode, set())
        if not required.issubset(self.available_devices):
            raise ValueError(f"mode {mode.value} requires unavailable devices")
        self.mode = mode
        if density not in DENSITIES:
            raise ValueError(f"unsupported density {density}")
        self.density = density
        self.npu_enabled = "NPU" in self.available_devices
        self.gpu_enabled = "GPU" in self.available_devices
        self.sequence = 0

    def _enabled_devices(self) -> tuple[str, ...]:
        enabled = []
        for device in DEVICE_ORDER:
            if device == "NPU" and not self.npu_enabled:
                continue
            if device == "GPU" and not self.gpu_enabled:
                continue
            enabled.append(device)
        return tuple(enabled)

    def _actual_devices(self) -> tuple[str, ...]:
        enabled = self._enabled_devices()
        if self.mode == DeviceMode.OFF or (not self.npu_enabled and not self.gpu_enabled):
            return ("CPU",)
        if self.mode == DeviceMode.NPU_ONLY:
            return ("NPU",) if self.npu_enabled else ("CPU",)
        if self.mode == DeviceMode.GPU_ONLY:
            return ("GPU",) if self.gpu_enabled else ("CPU",)
        if self.mode == DeviceMode.CPU_ONLY:
            return ("CPU",)
        return enabled

    def snapshot(self) -> PolicySnapshot:
        enabled = self._enabled_devices()
        actual_devices = self._actual_devices()
        if self.mode == DeviceMode.OFF or (not self.npu_enabled and not self.gpu_enabled):
            label = "OFF · NPU off · GPU off"
        else:
            mode_label = self.mode.value.upper()
            if self.mode == DeviceMode.AUTO:
                actual = "AUTO:" + ",".join(actual_devices)
            elif self.mode == DeviceMode.SPREAD:
                actual = "ROUND-ROBIN " + "/".join(actual_devices)
            elif self.mode == DeviceMode.SPLIT:
                actual = "STAGE SPLIT " + "/".join(actual_devices)
            else:
                actual = "/".join(actual_devices)
            disabled = [
                name
                for name, enabled_state in (
                    ("NPU", self.npu_enabled),
                    ("GPU", self.gpu_enabled),
                )
                if not enabled_state
            ]
            suffix = f" · {', '.join(disabled)} off" if disabled else ""
            label = f"{mode_label} · {actual}{suffix}"
        return PolicySnapshot(
            sequence=self.sequence,
            mode=self.mode,
            npu_enabled=self.npu_enabled,
            gpu_enabled=self.gpu_enabled,
            density=self.density,
            available_devices=self.available_devices,
            active_devices=actual_devices,
            label=label,
        )

    def _changed(self) -> PolicySnapshot:
        self.sequence += 1
        return self.snapshot()

    def toggle_npu(self) -> PolicySnapshot:
        if "NPU" not in self.available_devices:
            return self.snapshot()
        self.npu_enabled = not self.npu_enabled
        return self._changed()

    def toggle_gpu(self) -> PolicySnapshot:
        if "GPU" not in self.available_devices:
            return self.snapshot()
        self.gpu_enabled = not self.gpu_enabled
        return self._changed()

    def set_mode(self, mode: DeviceMode) -> PolicySnapshot:
        required = _MODE_REQUIREMENTS.get(mode, set())
        if not required.issubset(self.available_devices):
            raise ValueError(f"mode {mode.value} is unavailable")
        if mode == self.mode:
            return self.snapshot()
        self.mode = mode
        return self._changed()

    def cycle_mode(self) -> PolicySnapshot:
        current_index = MODE_CYCLE.index(self.mode) if self.mode in MODE_CYCLE else -1
        for offset in range(1, len(MODE_CYCLE) + 1):
            candidate = MODE_CYCLE[(current_index + offset) % len(MODE_CYCLE)]
            if _MODE_REQUIREMENTS.get(candidate, set()).issubset(self.available_devices):
                self.mode = candidate
                return self._changed()
        raise RuntimeError("no available device mode")

    def set_density(self, density: int) -> PolicySnapshot:
        if density not in DENSITIES:
            raise ValueError(f"unsupported density {density}")
        if density == self.density:
            return self.snapshot()
        self.density = density
        return self._changed()

    def cycle_density(self, direction: int) -> PolicySnapshot:
        if direction == 0:
            raise ValueError("density direction must be non-zero")
        index = DENSITIES.index(self.density)
        next_index = max(0, min(len(DENSITIES) - 1, index + (1 if direction > 0 else -1)))
        return self.set_density(DENSITIES[next_index])

    def device_for_stream(
        self,
        stream_index: int,
        stage_preference: str | None = None,
    ) -> str:
        if stream_index < 0:
            raise ValueError("stream index must be non-negative")
        active = self._enabled_devices()
        if not active:
            active = ("CPU",)
        if self.mode == DeviceMode.AUTO:
            return "AUTO:" + ",".join(active)
        if self.mode in {DeviceMode.CPU_ONLY, DeviceMode.OFF}:
            return "CPU"
        if self.mode == DeviceMode.NPU_ONLY:
            return "NPU" if self.npu_enabled else "CPU"
        if self.mode == DeviceMode.GPU_ONLY:
            return "GPU" if self.gpu_enabled else "CPU"
        if self.mode == DeviceMode.SPLIT:
            preference = (stage_preference or "").upper()
            if preference == "NPU" and self.npu_enabled:
                return "NPU"
            if preference == "GPU" and self.gpu_enabled:
                return "GPU"
            return active[stream_index % len(active)]
        return active[stream_index % len(active)]

    def is_disabled(self, engine: str) -> bool:
        engine = engine.upper()
        if engine == "NPU":
            return not self.npu_enabled
        if engine == "GPU":
            return not self.gpu_enabled
        return False

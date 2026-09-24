"""DeviceAvailability gate and fingerprinted warm-cache probe."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import openvino as ov
import psutil
from PySide6.QtCore import QThread, Signal

from app.telemetry.devices_win import (
    enumerate_compute_accelerators,
    enumerate_display_adapters,
    is_intel,
    is_npu,
)


DEVICES = ("NPU", "GPU", "CPU")


@dataclass(frozen=True)
class CompileResult:
    success: bool
    execution_devices: tuple[str, ...]
    error: str | None
    compile_ms: float | None


@dataclass(frozen=True)
class ModelAvailability:
    model_id: str
    devices: dict[str, CompileResult]


@dataclass(frozen=True)
class AvailabilityMatrix:
    generated_at: str
    fingerprint: str
    cache_hit: bool
    models: dict[str, ModelAvailability]

    @property
    def available_devices(self) -> tuple[str, ...]:
        return tuple(
            device
            for device in DEVICES
            if any(result.devices.get(device, CompileResult(False, (), "not probed", None)).success for result in self.models.values())
        )

    def supports(self, model_id: str, device: str) -> bool:
        root = device.split(":", 1)[0] if device.startswith("AUTO:") else device
        if device.startswith("AUTO:"):
            roots = device.split(":", 1)[1].split(",")
            return all(self.supports(model_id, item) for item in roots)
        result = self.models.get(model_id)
        return bool(result and result.devices.get(root, CompileResult(False, (), "not probed", None)).success)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "generated_at": self.generated_at,
            "fingerprint": self.fingerprint,
            "cache_hit": self.cache_hit,
            "available_devices": list(self.available_devices),
            "models": {
                model_id: {
                    "devices": {
                        device: asdict(result)
                        for device, result in model.devices.items()
                    }
                }
                for model_id, model in self.models.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, cache_hit: bool) -> "AvailabilityMatrix":
        models: dict[str, ModelAvailability] = {}
        for model_id, model_payload in payload["models"].items():
            devices: dict[str, CompileResult] = {}
            for device, result in model_payload["devices"].items():
                devices[device] = CompileResult(
                    success=bool(result["success"]),
                    execution_devices=tuple(result.get("execution_devices", [])),
                    error=result.get("error"),
                    compile_ms=result.get("compile_ms"),
                )
            models[model_id] = ModelAvailability(model_id=model_id, devices=devices)
        return cls(
            generated_at=payload["generated_at"],
            fingerprint=payload["fingerprint"],
            cache_hit=cache_hit,
            models=models,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_files(model: dict[str, Any], root: Path) -> list[Path]:
    model_dir = root / "models" / model["id"]
    if model["source"] == "huggingface":
        names = list(model["files"])
    else:
        names = [Path(model["url_xml"]).name, Path(model["url_bin"]).name]
    return [model_dir / name for name in names]


def _driver_versions() -> dict[str, str | None]:
    npu_version = None
    gpu_version = None
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
            npu_version = device.driver_version
    except Exception:
        pass
    try:
        device = next((item for item in enumerate_display_adapters() if is_intel(item)), None)
        if device is not None:
            gpu_version = device.driver_version
    except Exception:
        pass
    return {"npu": npu_version, "gpu": gpu_version}


def _fingerprint(
    models: list[dict[str, Any]],
    available_devices: list[str],
    root: Path,
) -> str:
    model_hashes: dict[str, str] = {}
    for model in models:
        digest = hashlib.sha256()
        for path in _model_files(model, root):
            digest.update(path.name.encode("utf-8"))
            digest.update(_sha256(path).encode("ascii"))
        model_hashes[model["id"]] = digest.hexdigest()
    payload = {
        "schema": 1,
        "openvino": ov.__version__,
        "windows": platform.version(),
        "python": platform.python_version(),
        "drivers": _driver_versions(),
        "devices": available_devices,
        "models": model_hashes,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {text}\n")


def _load_cache(cache_path: Path, fingerprint: str) -> AvailabilityMatrix | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("fingerprint") != fingerprint or payload.get("schema") != 1:
        return None
    try:
        return AvailabilityMatrix.from_dict(payload, cache_hit=True)
    except (KeyError, TypeError, ValueError):
        return None


def _write_cache(cache_path: Path, matrix: AvailabilityMatrix) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(matrix.to_dict(), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, cache_path)


def probe_device_availability(
    config_path: Path,
    cache_dir: Path,
    *,
    force: bool = False,
    progress: Callable[[int, str], None] | None = None,
) -> AvailabilityMatrix:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models = config["models"]
    root = config_path.resolve().parent.parent
    core = ov.Core()
    available = [device for device in DEVICES if device in core.available_devices]
    fingerprint = _fingerprint(models, available, root)
    cache_path = cache_dir / "availability.json"
    log_path = cache_dir.parent / "logs" / "availability.log"
    if not force:
        cached = _load_cache(cache_path, fingerprint)
        if cached is not None:
            _log(log_path, f"availability cache HIT {fingerprint}")
            if progress is not None:
                progress(len(DEVICES) * len(models), f"Availability cache hit · {fingerprint[:12]}")
            return cached

    core.set_property({"CACHE_DIR": str(cache_dir)})
    matrix_models: dict[str, ModelAvailability] = {}
    total = len(models) * len(DEVICES)
    completed = 0
    for model in models:
        model_id = model["id"]
        model_path = _model_files(model, root)[0]
        device_results: dict[str, CompileResult] = {}
        try:
            model_ir = core.read_model(str(model_path))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            device_results = {
                device: CompileResult(False, (), message, None) for device in DEVICES
            }
            _log(log_path, f"model={model_id} read FAIL {message}")
            matrix_models[model_id] = ModelAvailability(model_id, device_results)
            completed += len(DEVICES)
            if progress is not None:
                progress(completed, f"{model_id} · IR read failed")
            continue
        for device in DEVICES:
            if device not in available:
                result = CompileResult(False, (), "device unavailable", None)
                device_results[device] = result
                _log(log_path, f"model={model_id} device={device} SKIP device unavailable")
            else:
                config_values: dict[str, Any] = {
                    "CACHE_DIR": str(cache_dir),
                    "PERFORMANCE_HINT": "LATENCY",
                }
                if device == "CPU":
                    config_values["PERFORMANCE_HINT"] = "THROUGHPUT"
                    config_values["NUM_STREAMS"] = psutil.cpu_count(logical=False) or 1
                started = time.perf_counter()
                try:
                    compiled = core.compile_model(model_ir, device, config_values)
                    compile_ms = (time.perf_counter() - started) * 1000.0
                    raw_devices = compiled.get_property("EXECUTION_DEVICES")
                    execution = (
                        (raw_devices,)
                        if isinstance(raw_devices, str)
                        else tuple(str(value) for value in raw_devices)
                    )
                    roots = {
                        value.upper().split(".", 1)[0] for value in execution
                    }
                    if roots != {device}:
                        message = (
                            "placement assertion failed: "
                            f"requested {device}, EXECUTION_DEVICES={list(execution)}"
                        )
                        result = CompileResult(False, execution, message, compile_ms)
                        _log(
                            log_path,
                            f"model={model_id} device={device} FAIL {message}",
                        )
                    else:
                        result = CompileResult(True, execution, None, compile_ms)
                        _log(
                            log_path,
                            f"model={model_id} device={device} PASS compile_ms={compile_ms:.2f} "
                            f"EXECUTION_DEVICES={list(execution)}",
                        )
                    del compiled
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:1000]
                    result = CompileResult(False, (), message, None)
                    _log(log_path, f"model={model_id} device={device} FAIL {message}")
            device_results[device] = result
            completed += 1
            if progress is not None:
                state = "PASS" if result.success else "FAIL"
                progress(
                    completed,
                    f"{model_id} · {device} {state} ({completed}/{total})",
                )
        matrix_models[model_id] = ModelAvailability(model_id, device_results)

    matrix = AvailabilityMatrix(
        generated_at=datetime.now(timezone.utc).isoformat(),
        fingerprint=fingerprint,
        cache_hit=False,
        models=matrix_models,
    )
    _write_cache(cache_path, matrix)
    _log(log_path, f"availability cache WRITE {fingerprint} devices={list(matrix.available_devices)}")
    return matrix


class AvailabilityWorker(QThread):
    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        config_path: Path,
        cache_dir: Path,
        *,
        force: bool = False,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.cache_dir = cache_dir
        self.force = force

    def run(self) -> None:
        try:
            matrix = probe_device_availability(
                self.config_path,
                self.cache_dir,
                force=self.force,
                progress=lambda value, text: self.progress.emit(value, text),
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.completed.emit(matrix)

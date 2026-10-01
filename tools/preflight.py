#!/usr/bin/env python3
"""Print the Phase 0/booth PASS/FAIL verification table."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openvino as ov
import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS_CONFIG = ROOT / "config" / "models.json"
PROFILES_CONFIG = ROOT / "config" / "platform_profiles.json"
TELEMETRY_MAP = ROOT / "config" / "telemetry_map.json"
CACHE_DIR = ROOT / "cache"
MINIMUM_NPU_DRIVER = "32.0.100.5540"
DEVICES = ("NPU", "GPU", "CPU")
RUNTIME_INFERENCE_COUNT = 20


@dataclass
class Check:
    category: str
    name: str
    status: str
    detail: str
    remediation: str = ""


class Preflight:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, category: str, name: str, status: str, detail: str, remediation: str = "") -> None:
        if status not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(f"invalid status {status}")
        self.checks.append(Check(category, name, status, detail, remediation))

    @property
    def failures(self) -> int:
        return sum(check.status == "FAIL" for check in self.checks)

    @property
    def warnings(self) -> int:
        return sum(check.status == "WARN" for check in self.checks)

    def print(self) -> None:
        category_width = max(len(check.category) for check in self.checks)
        name_width = max(len(check.name) for check in self.checks)
        print(f"Engine Lab preflight — {datetime.now(timezone.utc).isoformat()}")
        print(f"Repository: {ROOT}")
        print(f"Python: {sys.executable}")
        print()
        print(f"{'STATUS':6}  {'CATEGORY':<{category_width}}  {'CHECK':<{name_width}}  DETAIL")
        print("-" * (16 + category_width + name_width + 6))
        for check in self.checks:
            print(
                f"{check.status:6}  {check.category:<{category_width}}  "
                f"{check.name:<{name_width}}  {check.detail}"
            )
            if check.remediation and check.status in {"WARN", "FAIL"}:
                label = "remediation" if check.status == "FAIL" else "operator note"
                print(f"{'':6}  {'':<{category_width}}  {'':<{name_width}}  {label}: {check.remediation}")
        print()
        passed = sum(check.status == "PASS" for check in self.checks)
        print(
            f"Summary: {passed} PASS, {self.warnings} WARN, {self.failures} FAIL — "
            f"{len(self.checks)} total rows"
        )


def _windows_build() -> tuple[int, int]:
    version = sys.getwindowsversion()
    ubr = 0
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            ) as key:
                ubr = int(winreg.QueryValueEx(key, "UBR")[0])
        except (OSError, ValueError):
            pass
    return version.build, ubr


def _vc_runtime_installed() -> tuple[bool, str]:
    if os.name != "nt":
        return False, "not Windows"
    import winreg

    versions: list[str] = []
    installed = False
    for hive, path in (
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        ),
    ):
        try:
            with winreg.OpenKey(hive, path) as key:
                version = str(winreg.QueryValueEx(key, "Version")[0])
                value = int(winreg.QueryValueEx(key, "Installed")[0])
                versions.append(version)
                installed = installed or bool(value)
        except OSError:
            continue
    return installed, ", ".join(sorted(set(versions))) or "not found"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _dependency_checks(report: Preflight) -> None:
    required = {
        "openvino": "2026.4.0",
        "PySide6": "6.11.2",
        "psutil": "7.2.2",
    }
    for distribution, expected in required.items():
        actual = _package_version(distribution)
        status = "PASS" if actual == expected else "FAIL"
        report.add(
            "dependency",
            distribution,
            status,
            f"expected {expected}, found {actual or 'not installed'}",
            f"Install requirements.txt with .venv\\Scripts\\python.exe -m pip install -r requirements.txt",
        )
    report.add(
        "dependency",
        "OpenVINO import",
        "PASS" if ov.__version__.startswith("2026.4.0") else "FAIL",
        f"runtime reports {ov.__version__}",
        "Reinstall the pinned openvino wheel in the repo-local .venv",
    )
    for distribution in ("numpy", "opencv-python", "huggingface_hub", "pyyaml"):
        actual = _package_version(distribution)
        report.add(
            "dependency",
            distribution,
            "PASS" if actual else "FAIL",
            actual or "not installed",
            "Install requirements.txt in the repo-local .venv",
        )

    python_ok = sys.version_info[:2] == (3, 12)
    report.add(
        "runtime",
        "Python 3.12",
        "PASS" if python_ok else "FAIL",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Recreate .venv with Python 3.12",
    )
    executable = Path(sys.executable).resolve()
    venv_python = (ROOT / ".venv" / "Scripts" / "python.exe").resolve()
    in_venv = executable == venv_python
    report.add(
        "runtime",
        "repo-local venv",
        "PASS" if in_venv else "FAIL",
        str(executable),
        r"Run C:\dev\engine-lab\.venv\Scripts\python.exe tools\preflight.py",
    )
    installed, version = _vc_runtime_installed()
    report.add(
        "runtime",
        "VC++ x64 runtime",
        "PASS" if installed else "FAIL",
        version,
        "Install Microsoft Visual C++ Redistributable 2015-2022 x64, then reboot if required",
    )


def _device_checks(report: Preflight, core: ov.Core) -> tuple[list[Any], list[Any], set[str]]:
    from app.telemetry.devices_win import (
        driver_at_least,
        enumerate_compute_accelerators,
        enumerate_display_adapters,
        is_intel,
        is_npu,
        select_driver,
    )

    try:
        compute_devices = enumerate_compute_accelerators()
        compute_error = ""
    except Exception as exc:
        compute_devices = []
        compute_error = f"{type(exc).__name__}: {exc}"
    try:
        display_devices = enumerate_display_adapters()
        display_error = ""
    except Exception as exc:
        display_devices = []
        display_error = f"{type(exc).__name__}: {exc}"

    npu_device = select_driver(compute_devices, npu=True)
    gpu_device = select_driver(display_devices, npu=False)
    available = {device.upper() for device in core.available_devices}

    npu_present = npu_device is not None and "NPU" in available
    gpu_present = gpu_device is not None and is_intel(gpu_device) and "GPU" in available
    report.add(
        "device",
        "NPU present",
        "PASS" if npu_present else "FAIL",
        (
            f"SetupAPI={npu_device.friendly_name if npu_device else 'not found'}; "
            f"OpenVINO devices={','.join(core.available_devices)}"
            + (f"; SetupAPI error={compute_error}" if compute_error else "")
        ),
        "Install Intel NPU driver 32.0.100.5540 or later, reboot, and verify Device Manager",
    )
    report.add(
        "device",
        "GPU present",
        "PASS" if gpu_present else "FAIL",
        (
            f"SetupAPI Intel GPU={gpu_device.friendly_name if gpu_device and is_intel(gpu_device) else 'not found'}; "
            f"OpenVINO devices={','.join(core.available_devices)}"
            + (f"; SetupAPI error={display_error}" if display_error else "")
        ),
        "Install the current Intel Arc/Iris Xe graphics driver and reboot",
    )

    if npu_device is None:
        npu_driver_status = "FAIL"
        npu_driver_detail = "no Intel NPU device"
    else:
        npu_driver_ok = driver_at_least(npu_device.driver_version, MINIMUM_NPU_DRIVER)
        npu_driver_status = "PASS" if npu_driver_ok else "FAIL"
        npu_driver_detail = (
            f"{npu_device.driver_version or 'unknown'} from {npu_device.friendly_name}; "
            f"minimum {MINIMUM_NPU_DRIVER}"
        )
    report.add(
        "driver",
        "NPU driver version",
        npu_driver_status,
        npu_driver_detail,
        f"Update Intel NPU driver to {MINIMUM_NPU_DRIVER} or later, reboot, then reset the OpenVINO cache",
    )

    if gpu_device is None or not is_intel(gpu_device):
        gpu_driver_status = "FAIL"
        gpu_driver_detail = "no Intel display adapter"
    else:
        gpu_driver_status = "PASS" if gpu_device.driver_version else "WARN"
        gpu_driver_detail = f"{gpu_device.driver_version or 'unknown'} from {gpu_device.friendly_name}"
    report.add(
        "driver",
        "GPU driver version",
        gpu_driver_status,
        gpu_driver_detail,
        "Install the current Intel Arc/Iris Xe graphics driver and reboot",
    )

    return compute_devices, display_devices, available


def _telemetry_checks(report: Preflight, compute_devices: list[Any], available: set[str]) -> dict[str, Any] | None:
    from app.telemetry.devices_win import is_npu

    if not TELEMETRY_MAP.exists():
        report.add(
            "telemetry",
            "telemetry map",
            "FAIL",
            f"missing {TELEMETRY_MAP.relative_to(ROOT)}",
            "Run .venv\\Scripts\\python.exe tools\\probe_telemetry.py",
        )
        return None
    try:
        telemetry_map = json.loads(TELEMETRY_MAP.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.add("telemetry", "telemetry map", "FAIL", f"invalid JSON: {exc}", "Re-run tools/probe_telemetry.py")
        return None

    required_keys = {
        "probed_on",
        "windows_build",
        "driver_versions",
        "npu_phys_ids",
        "gpu_phys_ids",
        "counter_source",
        "instances",
        "fallback",
        "notes",
    }
    missing = sorted(required_keys - set(telemetry_map))
    schema_ok = not missing and telemetry_map.get("fallback") == "npu_duty_cycle"
    report.add(
        "telemetry",
        "map schema",
        "PASS" if schema_ok else "FAIL",
        "required schema present" if schema_ok else f"missing/invalid: {missing or 'fallback'}",
        "Re-run tools/probe_telemetry.py; do not hand-edit measured counter classifications",
    )
    npu_instances = [
        instance
        for instance in telemetry_map.get("instances", [])
        if instance.get("device") == "NPU" and instance.get("pattern")
    ]
    npu_hardware = any(
        is_npu(device) for device in compute_devices
    ) or "NPU" in available
    if npu_instances:
        report.add(
            "telemetry",
            "NPU counters",
            "PASS",
            f"{len(npu_instances)} NPU-matching PDH instance pattern(s), source={telemetry_map.get('counter_source')}",
        )
    elif npu_hardware:
        report.add(
            "telemetry",
            "NPU counters",
            "WARN",
            "no NPU-matching PDH instance observed; app-measured duty cycle fallback is declared",
            "Re-probe while an NPU workload is active; the HUD must show 'NPU busy (app-measured)'",
        )
    else:
        report.add(
            "telemetry",
            "NPU counters",
            "FAIL",
            "NPU hardware is absent, so the app-measured fallback cannot prove NPU placement",
            "Install the Intel NPU driver and reboot before the booth",
        )
    return telemetry_map


def _model_paths(model: dict[str, Any]) -> list[Path]:
    model_dir = ROOT / "models" / model["id"]
    if model["source"] == "huggingface":
        names = list(model["files"]) + ["source_config.json", "labels.txt", "labels.json"]
    elif model["source"] == "converted":
        # Locally built IR (CLIP): the entry lists its own required artefacts,
        # including the baked vocabulary, and has no source_config or labels.
        names = list(model["files"])
    else:
        names = [Path(model["url_xml"]).name, Path(model["url_bin"]).name, "labels.json"]
    return [model_dir / name for name in names]


def _static_shape(model: ov.Model) -> tuple[list[int] | None, str]:
    if len(model.inputs) != 1:
        return None, f"expected one input, found {len(model.inputs)}"
    shape: list[int] = []
    for dimension in model.inputs[0].get_partial_shape():
        try:
            if not dimension.is_static:
                return None, f"dynamic dimension in {model.inputs[0].get_partial_shape()}"
            shape.append(int(dimension.get_length()))
        except Exception as exc:
            return None, f"could not read static input shape: {exc}"
    return shape, ""


def _placement(compiled: Any, device: str) -> tuple[list[str] | None, str]:
    try:
        raw = compiled.get_property("EXECUTION_DEVICES")
    except Exception as exc:
        return None, f"EXECUTION_DEVICES unavailable: {exc}"
    if isinstance(raw, str):
        values = [raw]
    else:
        values = [str(value) for value in raw]
    return values, ""


def _model_and_compile_checks(report: Preflight, config: dict[str, Any], core: ov.Core, available: set[str]) -> None:
    for model in config["models"]:
        model_id = model["id"]
        paths = _model_paths(model)
        missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists() or path.stat().st_size <= 0]
        if missing:
            report.add(
                "model files",
                model_id,
                "FAIL",
                f"missing/empty: {', '.join(missing)}",
                (
                    f"Build locally: {model.get('conversion_tool', 'see config/models.json')}"
                    if model.get("source") == "converted"
                    else "Run .venv\\Scripts\\python.exe tools\\download_models.py"
                ),
            )
            continue
        report.add(
            "model files",
            model_id,
            "PASS",
            f"{len(paths)} required local files present ({sum(path.stat().st_size for path in paths)} bytes)",
        )

        try:
            model_ir = core.read_model(str(paths[0]))
        except Exception as exc:
            report.add(
                "model IR",
                model_id,
                "FAIL",
                f"OpenVINO could not read {paths[0].name}: {exc}",
                "Re-download the exact pinned model files; do not substitute a source",
            )
            continue

        actual_shape, shape_error = _static_shape(model_ir)
        expected_shape = [int(value) for value in model["input_shape"]]
        shape_ok = actual_shape == expected_shape
        detail = f"configured={expected_shape}, IR={actual_shape}"
        if shape_error:
            detail = shape_error
        report.add(
            "model IR",
            model_id,
            "PASS" if shape_ok else "FAIL",
            detail,
            "Correct config/models.json only from the actual IR/source documentation, then re-run source verification and preflight",
        )

        for device in DEVICES:
            if device not in available:
                report.add(
                    "compile",
                    f"{model_id} on {device}",
                    "FAIL",
                    f"OpenVINO device {device} is unavailable",
                    "Install the matching Intel driver, reboot, and re-run preflight",
                )
                continue
            started = time.perf_counter()
            try:
                compile_config: dict[str, Any] = {
                    "CACHE_DIR": str(CACHE_DIR),
                    "PERFORMANCE_HINT": "LATENCY",
                }
                if device == "CPU":
                    compile_config["PERFORMANCE_HINT"] = "THROUGHPUT"
                    compile_config["NUM_STREAMS"] = psutil.cpu_count(logical=False) or 1
                compiled = core.compile_model(model_ir, device, compile_config)
                compile_ms = (time.perf_counter() - started) * 1000.0
                devices, placement_error = _placement(compiled, device)
                placement_ok = bool(devices) and any(
                    value.upper().split(".", 1)[0] == device for value in devices
                )
                if actual_shape is None:
                    raise RuntimeError(shape_error or "input shape is not static")
                input_tensor = np.zeros(tuple(actual_shape), dtype=np.float32)
                queue = ov.AsyncInferQueue(compiled)
                inference_started = time.perf_counter()
                for sample_index in range(RUNTIME_INFERENCE_COUNT):
                    queue.start_async(input_tensor, sample_index)
                    queue.wait_all()
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                status = "PASS" if placement_ok else "WARN"
                detail = (
                    f"compiled in {compile_ms:.0f} ms; "
                    f"{RUNTIME_INFERENCE_COUNT}/{RUNTIME_INFERENCE_COUNT} AsyncInferQueue runs "
                    f"in {inference_ms:.0f} ms; EXECUTION_DEVICES={devices or 'unavailable'}"
                )
                remediation = "Use EXECUTION_DEVICES as placement proof before rendering a device badge"
                report.add("compile", f"{model_id} on {device}", status, detail, remediation)
                if placement_error:
                    report.add(
                        "placement",
                        f"{model_id} on {device}",
                        "WARN",
                        placement_error,
                        "Use EXECUTION_DEVICES as placement proof before rendering a device badge",
                    )
                del queue
                del compiled
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000.0
                message = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
                if device == "CPU":
                    status = "FAIL"
                    remediation = "Re-download/verify the model and inspect the OpenVINO error; CPU must support every pinned model"
                else:
                    status = "WARN"
                    remediation = "Model/device incompatibility is an expected DeviceAvailability outcome; show the per-model failure in the operator overlay"
                report.add(
                    "compile",
                    f"{model_id} on {device}",
                    status,
                    f"failed after {elapsed:.0f} ms: {message}",
                    remediation,
                )


def _phase1_runtime_checks(report: Preflight, core: ov.Core, available: set[str]) -> None:
    from app.engine.stages import load_preprocess_config, model_input_size, preprocess_yolo

    model_path = ROOT / "models" / "yolo11n-fp16" / "yolo11n.xml"
    video_path = ROOT / "media" / "store-aisle-detection.mp4"
    if not model_path.is_file() or not video_path.is_file():
        report.add(
            "phase 1 runtime",
            "YOLO11n retail inputs",
            "FAIL",
            "model or retail video is missing",
            "Run tools/download_models.py",
        )
        return
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        report.add(
            "phase 1 runtime",
            "YOLO11n retail inputs",
            "FAIL",
            "retail video did not decode a frame",
            "Re-run tools/download_models.py and tools/preflight.py",
        )
        return
    try:
        model = core.read_model(str(model_path))
        model_height, model_width = model_input_size(model)
        preprocessed = preprocess_yolo(
            frame,
            model_width=model_width,
            model_height=model_height,
            metadata=load_preprocess_config(ROOT / "models" / "yolo11n-fp16" / "source_config.json"),
        )
    except Exception as exc:
        report.add(
            "phase 1 runtime",
            "YOLO11n retail input",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            "Restore the pinned YOLO11n FP16 model and its source preprocessing metadata",
        )
        return

    physical_cores = None
    try:
        import psutil

        physical_cores = psutil.cpu_count(logical=False)
    except Exception:
        physical_cores = None
    for device in DEVICES:
        if device not in available:
            report.add(
                "phase 1 runtime",
                f"YOLO11n inference on {device}",
                "FAIL",
                "device unavailable",
                "Install the matching Intel driver and reboot",
            )
            continue
        config: dict[str, Any] = {
            "CACHE_DIR": str(CACHE_DIR),
            "PERFORMANCE_HINT": "LATENCY",
        }
        if device == "CPU":
            config["PERFORMANCE_HINT"] = "THROUGHPUT"
            config["NUM_STREAMS"] = physical_cores or 1
        try:
            compile_started = time.perf_counter()
            compiled = core.compile_model(model, device, config)
            compile_ms = (time.perf_counter() - compile_started) * 1000.0
            inference_started = time.perf_counter()
            output = compiled(preprocessed.tensor)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            raw_devices = compiled.get_property("EXECUTION_DEVICES")
            devices = [raw_devices] if isinstance(raw_devices, str) else [str(item) for item in raw_devices]
            roots = {item.upper().split(".", 1)[0] for item in devices}
            placement_ok = roots == {device}
            detail = (
                f"one real retail-frame inference in {inference_ms:.2f} ms "
                f"(compile {compile_ms:.2f} ms); EXECUTION_DEVICES={devices}"
            )
            if device == "CPU":
                stream_count = compiled.get_property("NUM_STREAMS")
                detail += f"; NUM_STREAMS={stream_count} (physical cores={physical_cores})"
                streams_ok = int(stream_count) == int(physical_cores or 1)
            else:
                streams_ok = True
            report.add(
                "phase 1 runtime",
                f"YOLO11n inference on {device}",
                "PASS" if placement_ok and streams_ok and output is not None else "FAIL",
                detail,
                "Use EXECUTION_DEVICES as placement proof and preserve physical-core CPU streams",
            )
        except Exception as exc:
            report.add(
                "phase 1 runtime",
                f"YOLO11n inference on {device}",
                "FAIL",
                f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500],
                "Fix the device/driver/model runtime error; Phase 1 requires one inference on every device",
            )


def _availability_check(report: Preflight) -> Any | None:
    from app.engine.availability import DEVICES as AVAILABILITY_DEVICES
    from app.engine.availability import probe_device_availability

    try:
        matrix = probe_device_availability(MODELS_CONFIG, CACHE_DIR)
    except Exception as exc:
        report.add(
            "phase 2 availability",
            "model × device gate",
            "FAIL",
            f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500],
            "Restore the model files, cache write permission, and pinned OpenVINO runtime",
        )
        return None
    expected = len(matrix.models) * len(AVAILABILITY_DEVICES)
    actual = sum(len(model.devices) for model in matrix.models.values())
    successful = sum(
        result.success
        for model in matrix.models.values()
        for result in model.devices.values()
    )
    complete = actual == expected and "CPU" in matrix.available_devices
    status = "PASS" if complete else "FAIL"
    report.add(
        "phase 2 availability",
        "model × device gate",
        status,
        (
            f"cache={'hit' if matrix.cache_hit else 'cold'}; fingerprint={matrix.fingerprint[:16]}; "
            f"results={successful}/{expected} successful; devices={list(matrix.available_devices)}"
        ),
        "Inspect logs/availability.log and the F1 operator matrix; never hide a per-model failure",
    )
    return matrix


def _phase3_scenario_checks(
    report: Preflight,
    availability: Any,
) -> None:
    from app.engine.pipelines import ScenarioModelRegistry, load_scenario_catalog

    expected_ids = (
        "retail",
        "metro",
        "manufacturing",
        "robotics",
        "education",
        "health",
        "federal",
    )
    try:
        catalog = load_scenario_catalog(
            ROOT / "config" / "scenarios.json",
            MODELS_CONFIG,
            ROOT / "media",
        )
        registry = ScenarioModelRegistry(MODELS_CONFIG, ROOT / "models", CACHE_DIR)
    except Exception as exc:
        for scenario_id in expected_ids:
            report.add(
                "phase 3 scenario",
                scenario_id,
                "FAIL",
                f"{type(exc).__name__}: {exc}",
                "Restore config/scenarios.json and the verified model/media inventory",
            )
        return
    actual_ids = tuple(scenario.id for scenario in catalog.values())
    if actual_ids != expected_ids:
        report.add(
            "phase 3 scenario",
            "ordered scenario catalog",
            "FAIL",
            f"expected {expected_ids}, found {actual_ids}",
            "Restore deterministic scenario order retail, metro, manufacturing, robotics, education, health, federal",
        )
    for scenario in catalog.values():
        fallback_devices: list[str] = []
        try:
            media = catalog.media_path(scenario.id, camera_index=None)
            if media is None or not media.is_file():
                raise FileNotFoundError(f"missing media {media}")
            if not scenario.event_rules or not scenario.ticker:
                raise ValueError("event rules and ticker copy are required")
            required_rules = {
                "retail": {"confidence_min", "iou_threshold", "picked_up_dwell_s"},
                "metro": {"confidence_min", "count_dwell_s"},
                "manufacturing": {"confidence_min", "count_dwell_s"},
                "robotics": {"confidence_min", "count_dwell_s"},
                "education": {
                    "confidence_min",
                    "count_dwell_s",
                    "pose_keypoint_threshold",
                },
                "health": {"confidence_min", "fall_angle_deg", "posture_dwell_s"},
                "federal": {"confidence_min", "plate_confidence_min"},
            }[scenario.id]
            missing_rules = sorted(required_rules - set(scenario.event_rules))
            if missing_rules:
                raise ValueError(f"missing event rules: {missing_rules}")
            for stage in scenario.inference_stages:
                bundle = registry.bundle(stage.model_id)
                if not bundle.model_path.is_file():
                    raise FileNotFoundError(f"missing model {bundle.model_path}")
                if not availability.supports(stage.model_id, stage.device_pref):
                    if not availability.supports(stage.model_id, "CPU"):
                        raise RuntimeError(
                            f"{stage.model_id} is unavailable on {stage.device_pref} and CPU"
                        )
                    fallback_devices.append(f"{stage.stage}→CPU")
            for zone in scenario.zones:
                x, y, width, height = zone.roi
                if min(x, y, width, height) < 0.0 or x + width > 1.0 or y + height > 1.0:
                    raise ValueError(f"zone {zone.name} is outside normalized bounds")
            detail = (
                f"video={scenario.video_id}; stages={len(scenario.stages)}; "
                f"zones={len(scenario.zones)}; ticker={len(scenario.ticker)}"
            )
            if fallback_devices:
                detail += "; explicit CPU fallback=" + ",".join(fallback_devices)
            report.add("phase 3 scenario", scenario.id, "PASS", detail)
        except Exception as exc:
            report.add(
                "phase 3 scenario",
                scenario.id,
                "FAIL",
                f"{type(exc).__name__}: {exc}",
                "Restore the scenario graph and its verified models/media; never hide fallback",
            )


def _phase3_delivery_checks(report: Preflight) -> None:
    telemetry = json.loads(TELEMETRY_MAP.read_text(encoding="utf-8"))
    hud_source = (ROOT / "app" / "hud.py").read_text(encoding="utf-8")
    theme_source = (ROOT / "app" / "theme.py").read_text(encoding="utf-8")
    forbidden_metric_literals = (
        "np.percentile(samples, 50)",
        "np.percentile(samples, 95)",
        "np.percentile(latencies, 50)",
        "np.percentile(latencies, 95)",
        "value / 100.0",
        "min(100.0, value)",
    )
    metric_literals_clear = not any(
        literal in hud_source for literal in forbidden_metric_literals
    )
    required_theme_tokens = (
        "percentile_p50",
        "percentile_p95",
        "gauge_max_percent",
        "streams_real_time_fraction",
    )
    theme_tokens_present = all(token in theme_source for token in required_theme_tokens)
    telemetry_declared = (
        telemetry.get("counter_source") == "pdh_gpu_engine"
        and telemetry.get("fallback") == "npu_duty_cycle"
    )
    status = "PASS" if metric_literals_clear and theme_tokens_present and telemetry_declared else "FAIL"
    report.add(
        "phase 3 honesty",
        "HUD metric traceability",
        status,
        (
            f"counter_source={telemetry.get('counter_source')}; "
            f"fallback={telemetry.get('fallback')}; "
            f"theme_tokens={theme_tokens_present}; forbidden_literals={not metric_literals_clear}"
        ),
        "Keep measured metric scales/windows in theme.py or config; never hardcode a displayed value",
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    required_runbook_terms = (
        "pre-show checklist",
        "task manager cross-check",
        "npu compiled yesterday but not today",
        "metro",
        "federal",
        "attract",
    )
    runbook_ok = all(term in readme for term in required_runbook_terms)
    report.add(
        "phase 3 delivery",
        "README booth run-book",
        "PASS" if runbook_ok else "FAIL",
        f"required sections/commands present={runbook_ok}",
        "Restore the Phase 3 pre-show checklist, key map, recovery, and exact scenario commands",
    )

    launcher = (ROOT / "run_demo.bat").read_text(encoding="utf-8").lower()
    # The launcher starts windowed so an operator can reposition it; F11 toggles
    # fullscreen from inside the app, so --fullscreen is no longer required here.
    launcher_ok = all(token in launcher for token in ("pythonw.exe", "--scenario retail"))
    report.add(
        "phase 3 delivery",
        "one-click launcher",
        "PASS" if launcher_ok else "FAIL",
        f"pythonw/default-scenario tokens present={launcher_ok}",
        "Keep run_demo.bat as the no-console entry point (windowed; F11 goes fullscreen)",
    )

    benchmark = ROOT / "tools" / "benchmark_matrix.py"
    benchmark_ok = benchmark.is_file() and "attract-d4" in benchmark.read_text(encoding="utf-8")
    report.add(
        "phase 3 delivery",
        "benchmark matrix tool",
        "PASS" if benchmark_ok else "FAIL",
        f"benchmark tool present={benchmark.is_file()}; attract run declared={benchmark_ok}",
        "Restore tools/benchmark_matrix.py and its full scenario/density matrix",
    )


def _video_checks(report: Preflight, config: dict[str, Any]) -> None:
    for media in config["media"]:
        path = ROOT / "media" / media["file"]
        if not path.exists() or path.stat().st_size <= 0:
            report.add(
                "video",
                media["id"],
                "FAIL",
                f"missing/empty {path.relative_to(ROOT)}",
                "Run .venv\\Scripts\\python.exe tools\\download_models.py",
            )
            continue
        capture = cv2.VideoCapture(str(path))
        try:
            opened = capture.isOpened()
            ok, frame = capture.read()
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        valid = opened and ok and frame is not None and frame.size > 0 and fps > 0 and width > 0 and height > 0
        frame_shape = tuple(int(value) for value in frame.shape) if frame is not None else None
        report.add(
            "video",
            media["id"],
            "PASS" if valid else "FAIL",
            f"decoded={ok}, frame={frame_shape}, fps={fps:.3f}, size={width}x{height}",
            "Re-download the exact pinned MP4 and confirm the codec is supported by OpenCV",
        )


def _config_checks(report: Preflight, config: dict[str, Any]) -> None:
    ids = [model["id"] for model in config.get("models", [])]
    unique = len(ids) == len(set(ids))
    static = all(
        isinstance(model.get("input_shape"), list)
        and model["input_shape"]
        and all(isinstance(value, int) and value > 0 for value in model["input_shape"])
        for model in config.get("models", [])
    )
    report.add(
        "config",
        "models.json",
        "PASS" if unique and static else "FAIL",
        f"schema_version={config.get('schema_version')}, verified_on={config.get('verified_on')}, models={len(ids)}",
        "Restore the verified config and re-run source verification",
    )
    try:
        profiles = json.loads(PROFILES_CONFIG.read_text(encoding="utf-8"))
        sourced = all(
            profile.get("source")
            for profile in profiles.get("profiles", [])
            for value in profile.get("peak_tops", {}).values()
            if value is not None
        )
        report.add(
            "config",
            "platform profiles",
            "PASS" if sourced else "FAIL",
            f"{len(profiles.get('profiles', []))} profiles; every displayed peak value has a source",
            "Add a verified source URL before displaying any peak value",
        )
    except (OSError, json.JSONDecodeError) as exc:
        report.add("config", "platform profiles", "FAIL", str(exc), "Restore config/platform_profiles.json")


def _write_results(report: Preflight) -> None:
    payload = {
        "checked_on": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "checks": [asdict(check) for check in report.checks],
        "summary": {
            "pass": sum(check.status == "PASS" for check in report.checks),
            "warn": report.warnings,
            "fail": report.failures,
        },
    }
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    output = logs / "preflight-latest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Machine-readable result: {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="development-only: omit model × device compilation; never use for phase exit verification",
    )
    args = parser.parse_args(argv)

    report = Preflight()
    build, ubr = _windows_build()
    windows_ok = build >= 26100
    report.add(
        "system",
        "Windows build",
        "PASS" if windows_ok else "FAIL",
        f"10.0.{build}.{ubr} (requires 26100+)",
        "Install Windows 11 24H2 or later and reboot",
    )
    _dependency_checks(report)
    try:
        config = json.loads(MODELS_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.add("config", "models.json", "FAIL", str(exc), "Restore the verified config/models.json")
        config = {"models": [], "media": []}
    _config_checks(report, config)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    try:
        core.set_property({"CACHE_DIR": str(CACHE_DIR)})
    except Exception as exc:
        report.add(
            "runtime",
            "model cache",
            "WARN",
            f"could not configure CACHE_DIR: {exc}",
            "Check write permission for the repo-local cache directory",
        )
    else:
        report.add("runtime", "model cache", "PASS", str(CACHE_DIR))

    compute_devices, _, available = _device_checks(report, core)
    _telemetry_checks(report, compute_devices, available)
    _video_checks(report, config)
    if args.skip_compile:
        report.add("compile", "matrix", "WARN", "skipped by explicit --skip-compile", "Run without this switch for phase exit verification")
    else:
        _model_and_compile_checks(report, config, core, available)
        availability_matrix = _availability_check(report)
        if availability_matrix is not None:
            _phase3_scenario_checks(report, availability_matrix)
    _phase3_delivery_checks(report)
    _phase1_runtime_checks(report, core, available)

    report.print()
    _write_results(report)
    return 1 if report.failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Preflight interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

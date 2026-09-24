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
import openvino as ov


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS_CONFIG = ROOT / "config" / "models.json"
PROFILES_CONFIG = ROOT / "config" / "platform_profiles.json"
TELEMETRY_MAP = ROOT / "config" / "telemetry_map.json"
CACHE_DIR = ROOT / "cache"
MINIMUM_NPU_DRIVER = "32.0.100.5540"
DEVICES = ("NPU", "GPU", "CPU")


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
                "Run .venv\\Scripts\\python.exe tools\\download_models.py",
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
                compiled = core.compile_model(model_ir, device)
                elapsed = (time.perf_counter() - started) * 1000.0
                devices, placement_error = _placement(compiled, device)
                placement_ok = bool(devices) and any(
                    value.upper().split(".", 1)[0] == device for value in devices
                )
                status = "PASS" if placement_ok else "WARN"
                detail = f"compiled in {elapsed:.0f} ms; EXECUTION_DEVICES={devices or 'unavailable'}"
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

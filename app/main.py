"""Engine Lab entry point.

Phase 0 provides the setup/preflight entry point and a local-only self-test.
The booth UI and inference runner are introduced in later phases.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONFIG = ROOT / "config" / "models.json"
REQUIRED_PROFILES = ROOT / "config" / "platform_profiles.json"
TELEMETRY_MAP = ROOT / "config" / "telemetry_map.json"

LOGGER = logging.getLogger("engine_lab")


def _session_log() -> logging.Handler:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    handler = logging.FileHandler(logs / f"session-{stamp}.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    return handler


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selftest_check(checks: list[tuple[str, bool, str]], name: str, operation: Callable[[], str]) -> None:
    try:
        detail = operation()
    except Exception as exc:
        checks.append((name, False, f"{type(exc).__name__}: {exc}"))
    else:
        checks.append((name, True, detail))


def _selftest() -> int:
    """Run deterministic local checks while outbound socket connections are blocked."""

    checks: list[tuple[str, bool, str]] = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def blocked(*args: object, **kwargs: object) -> None:
        raise OSError("network disabled by Engine Lab --selftest")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked  # type: ignore[method-assign]
    socket.create_connection = blocked  # type: ignore[assignment]
    try:
        _selftest_check(
            checks,
            "Windows build",
            lambda: (
                f"10.0.{sys.getwindowsversion().build}.{getattr(sys.getwindowsversion(), 'ubr', 0)}"
                if sys.getwindowsversion().build >= 26100
                else f"unsupported build {sys.getwindowsversion().build}"
            ),
        )
        _selftest_check(
            checks,
            "Python version",
            lambda: (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                if sys.version_info[:2] == (3, 12)
                else f"expected 3.12, found {sys.version_info.major}.{sys.version_info.minor}"
            ),
        )
        _selftest_check(
            checks,
            "repo-local venv",
            lambda: (
                "PASS"
                if Path(sys.executable).resolve()
                == (ROOT / ".venv" / "Scripts" / "python.exe").resolve()
                else f"not running from {ROOT / '.venv'}"
            ),
        )

        def dependency_check() -> str:
            expected = {
                "openvino": "2026.4.0",
                "PySide6": "6.11.2",
                "psutil": "7.2.2",
            }
            values = {
                name: importlib.metadata.version(name)
                for name in expected
            }
            mismatches = [f"{name}={values[name]}" for name in expected if values[name] != expected[name]]
            if mismatches:
                raise RuntimeError("version mismatch: " + ", ".join(mismatches))
            for module in ("cv2", "huggingface_hub", "yaml", "numpy"):
                __import__(module)
            return ", ".join(f"{name}={version}" for name, version in values.items())

        _selftest_check(checks, "runtime dependencies", dependency_check)
        _selftest_check(
            checks,
            "OpenVINO devices",
            lambda: ", ".join(__import__("openvino").Core().available_devices),
        )

        def models_check() -> str:
            config = _load_json(REQUIRED_CONFIG)
            models = config.get("models", [])
            media = config.get("media", [])
            if not models or not media:
                raise RuntimeError("models.json has no models or media")
            ids = [model["id"] for model in models]
            if len(ids) != len(set(ids)):
                raise RuntimeError("duplicate model id")
            for model in models:
                shape = model.get("input_shape")
                if not isinstance(shape, list) or not shape or not all(
                    isinstance(value, int) and value > 0 for value in shape
                ):
                    raise RuntimeError(f"non-static configured shape for {model['id']}: {shape}")
                model_dir = ROOT / "models" / model["id"]
                if model["source"] == "huggingface":
                    files = [*model["files"], "source_config.json", "labels.txt", "labels.json"]
                else:
                    files = [
                        Path(model["url_xml"]).name,
                        Path(model["url_bin"]).name,
                        "labels.json",
                    ]
                missing = [name for name in files if not (model_dir / name).is_file()]
                if missing:
                    raise RuntimeError(f"{model['id']} missing {', '.join(missing)}")
            for item in media:
                path = ROOT / "media" / item["file"]
                if not path.is_file() or path.stat().st_size <= 0:
                    raise RuntimeError(f"missing media {item['id']}")
                with path.open("rb") as handle:
                    if handle.read(12)[4:8] != b"ftyp":
                        raise RuntimeError(f"media {item['id']} has no MP4 ftyp signature")
            return f"{len(models)} models and {len(media)} videos present"

        _selftest_check(checks, "local model/media inventory", models_check)

        def profiles_check() -> str:
            profiles = _load_json(REQUIRED_PROFILES).get("profiles", [])
            for profile in profiles:
                if not profile.get("source"):
                    raise RuntimeError(f"profile {profile.get('id')} has no source")
                for engine, value in profile.get("peak_tops", {}).items():
                    if value is not None and not profile.get("source"):
                        raise RuntimeError(f"profile {profile.get('id')} has unsourced {engine} peak")
            return f"{len(profiles)} sourced platform profiles"

        _selftest_check(checks, "platform profile sources", profiles_check)

        def telemetry_check() -> str:
            telemetry = _load_json(TELEMETRY_MAP)
            required = {
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
            missing = sorted(required - set(telemetry))
            if missing:
                raise RuntimeError("missing keys: " + ", ".join(missing))
            if telemetry["counter_source"] not in {"pdh_gpu_engine", "none"}:
                raise RuntimeError(f"unknown counter source {telemetry['counter_source']!r}")
            if telemetry["fallback"] != "npu_duty_cycle":
                raise RuntimeError("NPU fallback is not declared")
            return f"provider={telemetry['counter_source']}, NPU instances={sum(i.get('device') == 'NPU' for i in telemetry['instances'])}"

        _selftest_check(checks, "telemetry map", telemetry_check)
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection  # type: ignore[assignment]

    print("Engine Lab local-only self-test (outbound network blocked)")
    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    failures = sum(not passed for _, passed, _ in checks)
    print(f"Self-test summary: {len(checks) - failures} PASS, {failures} FAIL")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Engine Lab heterogeneous AI demo")
    parser.add_argument("--selftest", action="store_true", help="run local-only Phase 0 checks")
    parser.add_argument(
        "--source",
        default="loop",
        help="input source: loop (default) or camera:<index>",
    )
    parser.add_argument(
        "--scenario",
        choices=("retail", "smart_city", "medical", "gov_defense"),
        default="retail",
    )
    parser.add_argument("--fullscreen", action="store_true", help="run the booth UI fullscreen")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("OV_TELEMETRY_OPT_IN", "0")
    handler = _session_log()
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    LOGGER.info("Engine Lab started source=%s scenario=%s selftest=%s", args.source, args.scenario, args.selftest)
    if args.selftest:
        return _selftest()
    print("Engine Lab Phase 0 setup is installed and verified by tools/preflight.py.")
    print("The Qt inference application is intentionally scheduled for Phase 1; no demo-mode metrics are fabricated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

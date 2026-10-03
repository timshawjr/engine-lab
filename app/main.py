"""Engine Lab application entry point."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QProgressDialog

from app.engine.availability import DEVICES, AvailabilityMatrix, AvailabilityWorker
from app.engine.device_policy import DENSITIES, DeviceMode
from app.engine.pipelines import (
    ScenarioCatalog,
    ScenarioModelRegistry,
    load_scenario_catalog,
)
from app.hud import MainWindow
from app import hangwatch


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONFIG = ROOT / "config" / "models.json"
REQUIRED_PROFILES = ROOT / "config" / "platform_profiles.json"
REQUIRED_RAG_CONFIG = ROOT / "config" / "rag.json"
REQUIRED_SCENARIOS = ROOT / "config" / "scenarios.json"
TELEMETRY_MAP = ROOT / "config" / "telemetry_map.json"
SCENARIO_IDS = (
    "retail",
    "metro",
    "manufacturing",
    "education",
    "health",
    "federal",
)

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
                if model["source"] in {"huggingface", "converted"}:
                    # "converted" models (CLIP) already list their own
                    # vocabulary artefacts, so only the IR pair is required
                    # here; they carry no labels.txt or source_config.json.
                    files = (
                        [*model["files"], "source_config.json", "labels.txt", "labels.json"]
                        if model["source"] == "huggingface"
                        else list(model["files"])
                    )
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

        def phase1_inventory_check() -> str:
            import openvino

            from app.engine.stages import load_labels, model_input_size
            from app.theme import THEME

            model_path = ROOT / "models" / "yolo11n-fp16" / "yolo11n.xml"
            model = openvino.Core().read_model(str(model_path))
            input_size = model_input_size(model)
            labels = load_labels(model_path.with_name("labels.txt"))
            if input_size != (640, 640):
                raise RuntimeError(f"unexpected Phase 1 input size: {input_size}")
            engine_pixels_at_96_dpi = round(THEME.font_engine_value * 96 / 72)
            if len(labels) != 80 or engine_pixels_at_96_dpi < 96:
                raise RuntimeError(
                    f"Phase 1 UI/model invariant failed: labels={len(labels)}, "
                    f"engine_font={THEME.font_engine_value}pt/{engine_pixels_at_96_dpi}px"
                )
            return f"YOLO11n static input={input_size}, labels={len(labels)}, 1920x1080 theme ready"

        _selftest_check(checks, "Phase 1 local inventory", phase1_inventory_check)

        def phase2_local_inventory_check() -> str:
            from app.engine.availability import AvailabilityMatrix
            from app.engine.device_policy import DevicePolicy, DeviceMode

            policy = DevicePolicy(available_devices=("NPU", "GPU", "CPU"))
            policy.toggle_npu()
            policy.toggle_gpu()
            both_off = policy.snapshot()
            if (
                policy.device_for_stream(0) != "CPU"
                or both_off.mode != DeviceMode.SPREAD
                or both_off.active_devices != ("CPU",)
            ):
                raise RuntimeError("Phase 2 both-off policy did not resolve to CPU")
            policy.set_density(8)
            availability_path = ROOT / "cache" / "availability.json"
            payload = json.loads(availability_path.read_text(encoding="utf-8"))
            matrix = AvailabilityMatrix.from_dict(payload, cache_hit=True)
            # Derive the expected model count from the config instead of
            # hardcoding it, so adding a model cannot silently invalidate this
            # self-test.
            configured = _load_json(REQUIRED_CONFIG)["models"]
            expected_ids = {model["id"] for model in configured}
            devices = set(matrix.available_devices)
            result_count = sum(len(model.devices) for model in matrix.models.values())
            problems: list[str] = []
            if set(matrix.models) != expected_ids:
                problems.append(
                    f"cache covers {sorted(set(matrix.models) ^ expected_ids)} unexpectedly"
                )
            for model_id, model in matrix.models.items():
                unknown = set(model.devices) - devices
                if unknown:
                    problems.append(f"{model_id} reports unknown devices {sorted(unknown)}")
            if "CPU" not in devices:
                problems.append("CPU missing from available devices")
            # Every model must be probed on every available device.
            if result_count != len(expected_ids) * len(devices):
                problems.append(
                    f"expected {len(expected_ids) * len(devices)} device results, "
                    f"found {result_count}"
                )
            if problems:
                raise RuntimeError("Phase 2 availability cache is incomplete: " + "; ".join(problems))
            return (
                f"policy both-off=CPU, density=8, availability="
                f"{len(expected_ids)} models x {len(devices)} devices"
            )

        _selftest_check(checks, "Phase 2 local inventory", phase2_local_inventory_check)

        def phase3_local_inventory_check() -> str:
            from app.engine.pipelines import load_scenario_catalog

            catalog = load_scenario_catalog(
                REQUIRED_SCENARIOS,
                REQUIRED_CONFIG,
                ROOT / "media",
            )
            if tuple(scenario.id for scenario in catalog.values()) != SCENARIO_IDS:
                raise RuntimeError("scenario order does not match the booth contract")
            stage_count = 0
            for scenario in catalog.values():
                if not scenario.stages or not scenario.zones or not scenario.ticker:
                    raise RuntimeError(f"scenario {scenario.id} is incomplete")
                media_path = catalog.media_path(scenario.id, camera_index=None)
                if media_path is None or not media_path.is_file():
                    raise RuntimeError(f"scenario {scenario.id} media is missing")
                stage_count += len(scenario.stages)
            return f"{len(SCENARIO_IDS)} ordered scenarios, {stage_count} stages, normalized zones, event thresholds"

        _selftest_check(checks, "Phase 3 local inventory", phase3_local_inventory_check)
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
    parser.add_argument("--selftest", action="store_true", help="run local-only setup checks")
    parser.add_argument(
        "--source",
        default="loop",
        help="input source: loop (default) or camera:<index>",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_IDS,
        default="retail",
        help="initial vertical",
    )
    parser.add_argument(
        "--device",
        choices=("NPU", "GPU", "CPU"),
        default=None,
        help="start in an explicit single-device mode; default starts in spread mode",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in DeviceMode if mode is not DeviceMode.OFF),
        default=None,
        help="start in a device policy mode",
    )
    parser.add_argument(
        "--density",
        type=int,
        choices=DENSITIES,
        default=1,
        help="initial number of independent streams",
    )
    parser.add_argument("--npu-off", action="store_true", help="start with the NPU operator toggle off")
    parser.add_argument("--gpu-off", action="store_true", help="start with the GPU operator toggle off")
    parser.add_argument(
        "--npu-toggle-after",
        type=float,
        default=0.0,
        help="diagnostic: toggle N after this many seconds (0 disables the scheduled action)",
    )
    parser.add_argument(
        "--gpu-toggle-after",
        type=float,
        default=0.0,
        help="diagnostic: toggle GPU after this many seconds (0 disables the scheduled action)",
    )
    parser.add_argument(
        "--switch-to",
        choices=SCENARIO_IDS,
        default=None,
        help="scenario used by --scenario-switch-after",
    )
    parser.add_argument(
        "--scenario-switch-after",
        type=float,
        default=0.0,
        help="diagnostic: switch to --switch-to after this many seconds",
    )
    parser.add_argument(
        "--attract",
        action="store_true",
        help="start in deterministic no-gauge attract mode",
    )
    parser.add_argument(
        "--rag-page",
        action="store_true",
        help="start on the document Q&A page (loads the GenAI pipelines)",
    )
    parser.add_argument(
        "--rag-ask",
        default="",
        help="diagnostic: ask this question on the RAG page after --rag-switch-after",
    )
    parser.add_argument(
        "--rag-switch-after",
        type=float,
        default=0.0,
        help="diagnostic: switch to the document Q&A page after this many seconds",
    )
    parser.add_argument(
        "--force-availability",
        action="store_true",
        help="diagnostic: ignore the fingerprinted DeviceAvailability cache",
    )
    parser.add_argument("--fullscreen", action="store_true", help="run the booth UI fullscreen")
    parser.add_argument(
        "--exit-after",
        type=float,
        default=0.0,
        help="diagnostic: exit after this many seconds (0 keeps the app running)",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="diagnostic: save a PNG before diagnostic exit",
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        help="diagnostic: write the final measured UI state as JSON",
    )
    return parser


def _camera_index(value: str) -> int | None:
    if value == "loop":
        return None
    if value.startswith("camera:"):
        try:
            index = int(value.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("camera source must be camera:<index>") from exc
        if index < 0:
            raise ValueError("camera index must be non-negative")
        return index
    raise ValueError("source must be loop or camera:<index>")


def _write_diagnostic(path: Path, payload: dict) -> None:
    path = path if path.is_absolute() else ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_availability_gate(*, force: bool) -> AvailabilityMatrix:
    probe_count = len(_load_json(REQUIRED_CONFIG)["models"]) * len(DEVICES)
    progress = QProgressDialog(
        "Checking model × device availability…",
        "",
        0,
        probe_count,
    )
    progress.setWindowTitle("Engine Lab — DeviceAvailability")
    progress.setCancelButton(None)
    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.show()

    worker = AvailabilityWorker(
        REQUIRED_CONFIG,
        ROOT / "cache",
        force=force,
    )
    loop = QEventLoop()
    result: dict[str, object] = {}

    def on_progress(value: int, text: str) -> None:
        progress.setValue(value)
        progress.setLabelText(text)

    def on_completed(matrix: object) -> None:
        result["matrix"] = matrix
        loop.quit()

    def on_failed(error: str) -> None:
        result["error"] = error
        loop.quit()

    worker.progress.connect(on_progress)
    worker.completed.connect(on_completed)
    worker.failed.connect(on_failed)
    worker.start()
    loop.exec()
    worker.wait()
    progress.close()
    if "error" in result:
        raise RuntimeError(f"DeviceAvailability gate failed: {result['error']}")
    matrix = result.get("matrix")
    if not isinstance(matrix, AvailabilityMatrix):
        raise RuntimeError("DeviceAvailability gate returned no matrix")
    LOGGER.info(
        "DeviceAvailability %s: devices=%s models=%d fingerprint=%s",
        "cache hit" if matrix.cache_hit else "cold probe",
        matrix.available_devices,
        len(matrix.models),
        matrix.fingerprint,
    )
    return matrix


def _prewarm_scenarios(
    registry: ScenarioModelRegistry,
    catalog: ScenarioCatalog,
    availability: AvailabilityMatrix,
) -> list[str]:
    model_ids = {
        stage.model_id
        for scenario in catalog.values()
        for stage in scenario.inference_stages
    }
    progress = QProgressDialog(
        "Precompiling enabled scenario models…",
        "",
        0,
        max(1, len(model_ids) * 2),
    )
    progress.setWindowTitle("Engine Lab — scenario compiler")
    progress.setCancelButton(None)
    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.show()
    fallbacks = registry.prewarm(
        catalog,
        availability,
        progress=lambda value, text: (
            progress.setValue(value),
            progress.setLabelText(text),
            QApplication.processEvents(),
        ),
    )
    progress.close()
    return fallbacks


def _run_application(args: argparse.Namespace) -> int:
    startup_origin = time.perf_counter()
    camera_index = _camera_index(args.source)
    if args.exit_after < 0:
        raise ValueError("--exit-after cannot be negative")
    for name, value in (
        ("--npu-toggle-after", args.npu_toggle_after),
        ("--gpu-toggle-after", args.gpu_toggle_after),
        ("--scenario-switch-after", args.scenario_switch_after),
    ):
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
        if value and (args.exit_after <= 0 or value >= args.exit_after):
            raise ValueError(f"{name} requires --exit-after greater than the action time")
    if args.scenario_switch_after and args.switch_to is None:
        raise ValueError("--scenario-switch-after requires --switch-to")
    if args.scenario_switch_after and args.switch_to == args.scenario:
        raise ValueError("--switch-to must differ from the initial --scenario")
    if args.screenshot is not None and args.exit_after <= 0:
        raise ValueError("--screenshot requires --exit-after")

    application = QApplication.instance() or QApplication([])
    application.setApplicationName("Engine Lab")
    application.setOrganizationName("Engine Lab")
    application.setStyle("Fusion")
    availability = _run_availability_gate(force=args.force_availability)
    catalog = load_scenario_catalog(
        REQUIRED_SCENARIOS,
        REQUIRED_CONFIG,
        ROOT / "media",
    )
    registry = ScenarioModelRegistry(
        REQUIRED_CONFIG,
        ROOT / "models",
        ROOT / "cache",
    )
    prewarm_fallbacks = _prewarm_scenarios(registry, catalog, availability)
    initial_mode = DeviceMode(args.mode) if args.mode is not None else None
    if args.device is not None and initial_mode is None:
        initial_mode = {
            "NPU": DeviceMode.NPU_ONLY,
            "GPU": DeviceMode.GPU_ONLY,
            "CPU": DeviceMode.CPU_ONLY,
        }[args.device]
    window = MainWindow(
        catalog=catalog,
        registry=registry,
        initial_scenario=args.scenario,
        camera_index=camera_index,
        cache_dir=ROOT / "cache",
        telemetry_map_path=TELEMETRY_MAP,
        profiles_path=REQUIRED_PROFILES,
        availability=availability,
        prewarm_fallbacks=prewarm_fallbacks,
        fullscreen=args.fullscreen,
        initial_mode=initial_mode,
        initial_density=args.density,
        npu_enabled=not args.npu_off,
        gpu_enabled=not args.gpu_off,
        attract_mode=args.attract,
        rag_config_path=REQUIRED_RAG_CONFIG,
        startup_origin_perf=startup_origin,
    )
    window.start()
    if args.rag_page or args.rag_switch_after:
        window.set_page("rag")
    if args.rag_ask:
        if not args.rag_switch_after and not args.rag_page:
            raise ValueError("--rag-ask requires --rag-page or --rag-switch-after")

        def ask_rag() -> None:
            window.rag_input.setText(args.rag_ask)
            window._ask_rag()

        QTimer.singleShot(
            round((args.rag_switch_after or 1.0) * 1000.0) + 500, ask_rag
        )
    if args.npu_toggle_after:
        QTimer.singleShot(
            round(args.npu_toggle_after * 1000.0),
            window.toggle_npu,
        )
    if args.gpu_toggle_after:
        QTimer.singleShot(
            round(args.gpu_toggle_after * 1000.0),
            window.toggle_gpu,
        )
    if args.scenario_switch_after:
        QTimer.singleShot(
            round(args.scenario_switch_after * 1000.0),
            lambda: window.switch_scenario(args.switch_to, reason="scheduled"),
        )

    diagnostic_path = args.diagnostic_output
    if diagnostic_path is None and args.exit_after > 0:
        mode_name = initial_mode.value if initial_mode is not None else "spread"
        suffix = "-attract" if args.attract else ""
        diagnostic_path = Path("logs") / f"phase3-{args.scenario}-{mode_name}-d{args.density}{suffix}.json"
    snapshot_written = False

    def write_snapshot() -> None:
        nonlocal snapshot_written
        if diagnostic_path is None or snapshot_written:
            return
        _write_diagnostic(diagnostic_path, window.diagnostic_snapshot())
        snapshot_written = True
        LOGGER.info("Wrote diagnostic snapshot %s", diagnostic_path)

    application.aboutToQuit.connect(write_snapshot)
    application.aboutToQuit.connect(window.shutdown)

    if args.exit_after > 0:
        def finish() -> None:
            LOGGER.info("Diagnostic exit after %.1f seconds", args.exit_after)
            if args.screenshot is not None:
                screenshot = args.screenshot
                if not screenshot.is_absolute():
                    screenshot = ROOT / screenshot
                window.save_screenshot(screenshot)
                LOGGER.info("Wrote diagnostic screenshot %s", screenshot)
            window.shutdown()
            application.quit()

        QTimer.singleShot(round(args.exit_after * 1000.0), finish)
    return int(application.exec())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("OV_TELEMETRY_OPT_IN", "0")
    os.chdir(ROOT)
    handler = _session_log()
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    # A Windows "Application Hang" kills the process with no traceback, so let the app record its
    # own stacks while it is still stuck (see app/hangwatch.py).
    hangwatch.arm(getattr(handler, "stream", None))
    LOGGER.info(
        "Engine Lab started source=%s scenario=%s device=%s mode=%s density=%s "
        "npu_off=%s gpu_off=%s npu_toggle_after=%s gpu_toggle_after=%s "
        "switch_to=%s scenario_switch_after=%s attract=%s selftest=%s",
        args.source,
        args.scenario,
        args.device,
        args.mode,
        args.density,
        args.npu_off,
        args.gpu_off,
        args.npu_toggle_after,
        args.gpu_toggle_after,
        args.switch_to,
        args.scenario_switch_after,
        args.attract,
        args.selftest,
    )
    if args.selftest:
        return _selftest()
    return _run_application(args)


if __name__ == "__main__":
    raise SystemExit(main())

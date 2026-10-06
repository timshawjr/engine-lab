"""Phase 3 scenario graphs, model registry, and multi-stage stream workers."""

from __future__ import annotations

import json
import logging
import queue
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import psutil
from PySide6.QtCore import QThread, Signal

from app.engine.availability import AvailabilityMatrix
from app.engine.device_policy import DeviceMode, DevicePolicy, PolicySnapshot
from app.engine.events import BusinessEvent, EventTracker
from app.engine.runner import (
    CompiledModelStore,
    OpenVINOSingleRunner,
    RunnerCompileConfig,
    RunnerInfo,
)
from app.engine.stages import (
    Detection,
    Keypoint,
    ModelInputSpec,
    crop_detection,
    crop_detection_head,
    decode_pose,
    load_labels,
    load_preprocess_config,
    load_zero_shot_vocabulary,
    postprocess_classification,
    postprocess_ssd,
    postprocess_yolo,
    postprocess_zero_shot,
    preprocess_classification,
    preprocess_clip,
    preprocess_omz_image,
    preprocess_yolo,
)
from app.scenarios.health import confident_pose_angle
from app.scenarios.retail import classification_allowed, classification_candidates
from app.telemetry.npu_fallback import NpuDutyCycle

LOGGER = logging.getLogger("engine_lab")


LOGICAL_STAGES = {"zone_event", "posture_event", "track_event"}


def _idle_stage_metric(
    assignment: StageAssignment,
    runner: OpenVINOSingleRunner,
    output_count: int,
    inference_count: int,
) -> StageMetric:
    """A real stage measurement for a frame where nothing was submitted.

    The stage still compiled and is still placed on its device, so it must keep
    reporting that placement rather than disappearing from the pipeline bar.
    """

    return StageMetric(
        stage=assignment.stage,
        model_id=assignment.model_id,
        requested_device=assignment.requested_device,
        execution_devices=runner.info.execution_devices,
        preprocess_ms=0.0,
        inference_ms=0.0,
        postprocess_ms=0.0,
        output_count=output_count,
        inference_count=inference_count,
    )


@dataclass(frozen=True)
class Zone:
    name: str
    roi: tuple[float, float, float, float]
    kind: str


@dataclass(frozen=True)
class StageSpec:
    stage: str
    model_id: str
    device_pref: str
    vocabulary: str | None = None
    # Devices where this model was measured to produce no usable output, read
    # from models.json. A model can compile correctly on a device and still be
    # blind there, so device selection must avoid these.
    unusable_devices: frozenset[str] = frozenset()

    def can_run_on(self, device: str) -> bool:
        """False when this model was measured to emit no usable output here.

        Only meaningful for a single device name; an ``AUTO:`` priority list is
        filtered separately in ``_fallback_for_model``.
        """
        return device not in self.unusable_devices


@dataclass(frozen=True)
class ScenarioConfig:
    id: str
    title: str
    vertical: str
    business_line: str
    video_id: str
    stages: tuple[StageSpec, ...]
    zones: tuple[Zone, ...]
    event_rules: dict[str, Any]
    ticker: tuple[str, ...]

    @property
    def inference_stages(self) -> tuple[StageSpec, ...]:
        return tuple(stage for stage in self.stages if stage.stage not in LOGICAL_STAGES)


@dataclass(frozen=True)
class StageAssignment:
    stage: str
    model_id: str
    intended_device: str
    requested_device: str
    compile_config: RunnerCompileConfig


@dataclass(frozen=True)
class StageMetric:
    stage: str
    model_id: str
    requested_device: str
    execution_devices: tuple[str, ...]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    output_count: int
    inference_count: int


@dataclass(frozen=True)
class PipelineFrameMetrics:
    stream_index: int
    scenario_id: str
    frame_index: int
    captured_at: float
    source_fps: float | None
    processing_fps: float
    inferences_per_second: float
    decode_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    end_to_end_ms: float
    detection_count: int
    detections_per_second: float
    frame_width: int
    frame_height: int
    stages: tuple[StageMetric, ...]
    event_counts_60s: dict[str, int]


@dataclass(frozen=True)
class PipelineFrame:
    stream_index: int
    scenario_id: str
    detections: tuple[Detection, ...]
    raw_detections: tuple[Detection, ...]
    events: tuple[BusinessEvent, ...]
    metrics: PipelineFrameMetrics
    placements: dict[str, RunnerInfo]
    zones: tuple[Zone, ...]
    fallbacks: tuple[str, ...]


@dataclass(frozen=True)
class ModelBundle:
    model_id: str
    task: str
    model_path: Path
    labels_path: Path | None
    source_config_path: Path | None
    labels: tuple[str, ...] | dict[int, str]
    keypoint_names: tuple[str, ...]
    source_urls: tuple[str, ...]


class ScenarioCatalog:
    def __init__(
        self,
        scenarios: Iterable[ScenarioConfig],
        media_paths: dict[str, Path],
    ) -> None:
        scenario_values = tuple(scenarios)
        scenario_ids = [scenario.id for scenario in scenario_values]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario ids must be unique")
        self._scenarios = {scenario.id: scenario for scenario in scenario_values}
        self.media_paths = dict(media_paths)

    def __getitem__(self, scenario_id: str) -> ScenarioConfig:
        try:
            return self._scenarios[scenario_id]
        except KeyError as exc:
            raise KeyError(f"unknown scenario {scenario_id!r}") from exc

    def values(self) -> tuple[ScenarioConfig, ...]:
        return tuple(self._scenarios.values())

    def media_path(self, scenario_id: str, camera_index: int | None) -> Path | None:
        if camera_index is not None:
            return None
        return self.media_paths[self[scenario_id].video_id]


def load_scenario_catalog(
    scenario_path: Path,
    models_config_path: Path,
    media_root: Path,
) -> ScenarioCatalog:
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    models_payload = json.loads(models_config_path.read_text(encoding="utf-8"))
    media_paths = {
        item["id"]: media_root / item["file"]
        for item in models_payload["media"]
    }
    # Measured "compiles but emits nothing" facts, keyed by model id. These are
    # verified data in models.json, not a hardcoded list in code.
    unusable_by_model = {
        item["id"]: frozenset(item.get("no_usable_output_on", {}))
        for item in models_payload["models"]
    }
    scenarios: list[ScenarioConfig] = []
    for item in payload["scenarios"]:
        zones = tuple(
            Zone(
                name=zone["name"],
                roi=tuple(float(value) for value in zone["roi"]),
                kind=zone["kind"],
            )
            for zone in item["zones"]
        )
        for zone in zones:
            x, y, width, height = zone.roi
            if min(x, y, width, height) < 0.0 or x + width > 1.0 or y + height > 1.0:
                raise ValueError(f"zone {zone.name} is outside normalized [0,1] bounds")
        scenarios.append(
            ScenarioConfig(
                id=item["id"],
                title=item["title"],
                vertical=item["vertical"],
                business_line=item["business_line"],
                video_id=item["video_id"],
                stages=tuple(
                    StageSpec(
                        stage=stage["stage"],
                        model_id=stage["model_id"],
                        device_pref=stage["device_pref"],
                        vocabulary=stage.get("vocabulary"),
                        unusable_devices=unusable_by_model.get(
                            stage["model_id"], frozenset()
                        ),
                    )
                    for stage in item["stages"]
                ),
                zones=zones,
                event_rules=dict(item["event_rules"]),
                ticker=tuple(item["ticker"]),
            )
        )
    return ScenarioCatalog(scenarios, media_paths)


class ScenarioModelRegistry:
    def __init__(
        self,
        models_config_path: Path,
        models_root: Path,
        cache_dir: Path,
    ) -> None:
        payload = json.loads(models_config_path.read_text(encoding="utf-8"))
        self.model_root = models_root
        self.cache_dir = cache_dir
        self.model_entries = {item["id"]: item for item in payload["models"]}
        self.bundles: dict[str, ModelBundle] = {}
        self.stores: dict[str, CompiledModelStore] = {}
        for model_id, entry in self.model_entries.items():
            model_dir = models_root / model_id
            if entry["source"] in {"huggingface", "converted"}:
                # "converted" IR is built locally from a PyTorch checkpoint by
                # tools/build_clip_zero_shot.py; the first listed file is the
                # OpenVINO .xml, same layout as the downloaded models.
                model_path = model_dir / entry["files"][0]
            else:
                model_path = model_dir / Path(entry["url_xml"]).name
            labels_path = model_dir / "labels.txt"
            if not labels_path.is_file():
                labels_path = model_dir / "labels.json"
            source_config_path = model_dir / "source_config.json"
            labels: tuple[str, ...] | dict[int, str] = ()
            keypoint_names: tuple[str, ...] = ()
            source_urls: list[str] = [str(entry["spec_url"])]
            if labels_path.name == "labels.txt" and labels_path.is_file():
                labels = load_labels(labels_path)
            elif labels_path.is_file():
                label_payload = json.loads(labels_path.read_text(encoding="utf-8"))
                if "class_ids" in label_payload:
                    labels = {
                        int(class_id): str(label)
                        for class_id, label in label_payload["class_ids"].items()
                    }
                elif "labels" in label_payload:
                    labels = tuple(str(label) for label in label_payload["labels"])
                if "keypoints" in label_payload:
                    keypoint_names = tuple(
                        str(name) for name in label_payload["keypoints"]
                    )
                if label_payload.get("source_url"):
                    source_urls.append(str(label_payload["source_url"]))
            self.bundles[model_id] = ModelBundle(
                model_id=model_id,
                task=str(entry["task"]),
                model_path=model_path,
                labels_path=labels_path if labels_path.is_file() else None,
                source_config_path=(
                    source_config_path if source_config_path.is_file() else None
                ),
                labels=labels,
                keypoint_names=keypoint_names,
                source_urls=tuple(dict.fromkeys(source_urls)),
            )

    def bundle(self, model_id: str) -> ModelBundle:
        return self.bundles[model_id]

    def store(self, model_id: str) -> CompiledModelStore:
        if model_id not in self.stores:
            bundle = self.bundle(model_id)
            self.stores[model_id] = CompiledModelStore(
                bundle.model_path,
                self.cache_dir,
            )
        return self.stores[model_id]

    def prewarm(
        self,
        catalog: ScenarioCatalog,
        availability: AvailabilityMatrix,
        progress: Callable[[int, str], None] | None = None,
    ) -> list[str]:
        default_policy = DevicePolicy(
            available_devices=availability.available_devices,
            mode=DeviceMode.SPREAD,
            density=1,
        )
        targets: dict[tuple[str, str], str] = {}
        for scenario in catalog.values():
            for assignment in build_stage_assignments(
                default_policy,
                scenario,
                0,
                availability=availability,
            ):
                if assignment.stage in LOGICAL_STAGES:
                    continue
                targets.setdefault(
                    (assignment.model_id, assignment.requested_device),
                    assignment.intended_device,
                )
        fallbacks: list[str] = []
        total = len(targets)
        for index, ((model_id, target), intended) in enumerate(targets.items(), start=1):
            if not availability.supports(model_id, target):
                target = "CPU"
                message = f"{model_id}: ran on CPU ({intended} compile unavailable)"
                if message not in fallbacks:
                    fallbacks.append(message)
            compile_config = RunnerCompileConfig(
                performance_hint="THROUGHPUT" if target == "CPU" else "LATENCY"
            )
            self.store(model_id).get(target, compile_config)
            if progress is not None:
                progress(index, f"Precompiled {model_id} on {target}")
        if not total:
            raise ValueError("no scenario inference models found")
        return fallbacks


def _stage_target(
    policy: DevicePolicy,
    scenario: ScenarioConfig,
    stream_index: int,
    stage: StageSpec,
) -> tuple[str, str]:
    snapshot = policy.snapshot()
    if snapshot.mode == DeviceMode.AUTO:
        return "AUTO:" + ",".join(snapshot.active_devices), stage.device_pref
    if snapshot.mode in {DeviceMode.CPU_ONLY, DeviceMode.OFF}:
        return "CPU", "CPU"
    if snapshot.mode == DeviceMode.NPU_ONLY:
        return ("NPU", "NPU") if policy.npu_enabled else ("CPU", "NPU")
    if snapshot.mode == DeviceMode.GPU_ONLY:
        return ("GPU", "GPU") if policy.gpu_enabled else ("CPU", "GPU")
    if snapshot.mode == DeviceMode.SPLIT:
        if stage.device_pref == "NPU":
            if policy.npu_enabled:
                return "NPU", "NPU"
            if policy.gpu_enabled:
                return "GPU", "NPU"
        if stage.device_pref == "GPU":
            if policy.gpu_enabled:
                return "GPU", "GPU"
            if policy.npu_enabled and stage.can_run_on("NPU"):
                return "NPU", "GPU"
        return "CPU", stage.device_pref

    auxiliary = stage.stage not in {"detector", "perimeter_detector", "plate_detector"}
    if auxiliary:
        if stage.device_pref == "NPU" and not policy.npu_enabled:
            if policy.gpu_enabled:
                return "GPU", "NPU"
            return "CPU", "NPU"
        if stage.device_pref == "GPU" and not policy.gpu_enabled:
            # Skip the NPU for a model measured to emit nothing there: the CPU
            # is slower but actually detects, which is the whole point.
            if policy.npu_enabled and stage.can_run_on("NPU"):
                return "NPU", "GPU"
            return "CPU", "GPU"
        return stage.device_pref, stage.device_pref
    if scenario.id == "federal" and stage.device_pref == "GPU" and policy.gpu_enabled:
        return "GPU", "GPU"
    requested = policy.device_for_stream(stream_index, stage.device_pref)
    if not requested.startswith("AUTO:") and not stage.can_run_on(requested):
        # Round-robin picked a device this model is measured to be blind on.
        # Take the next enabled device that can actually run it.
        for device in snapshot.active_devices:
            if device != requested and stage.can_run_on(device):
                return device, device
        return "CPU", requested
    return requested, requested


def _fallback_for_model(
    availability: AvailabilityMatrix,
    model_id: str,
    requested: str,
    active_devices: tuple[str, ...],
    unusable_devices: frozenset[str] = frozenset(),
) -> str:
    def usable(device: str) -> bool:
        return availability.supports(model_id, device) and device not in unusable_devices

    if requested.startswith("AUTO:"):
        priorities = tuple(
            device
            for device in requested.split(":", 1)[1].split(",")
            if device
        )
        supported = tuple(device for device in priorities if usable(device))
        return "AUTO:" + ",".join(supported) if supported else "CPU"
    if usable(requested):
        return requested
    for device in active_devices:
        if device != requested and usable(device):
            return device
    return "CPU"


def build_stage_assignments(
    policy: DevicePolicy,
    scenario: ScenarioConfig,
    stream_index: int,
    availability: AvailabilityMatrix | None = None,
) -> tuple[StageAssignment, ...]:
    snapshot: PolicySnapshot = policy.snapshot()
    assignments: list[StageAssignment] = []
    for stage in scenario.stages:
        if stage.stage in LOGICAL_STAGES:
            assignments.append(
                StageAssignment(
                    stage=stage.stage,
                    model_id=stage.model_id,
                    intended_device="CPU",
                    requested_device="CPU",
                    compile_config=RunnerCompileConfig(performance_hint="THROUGHPUT"),
                )
            )
            continue
        requested, intended = _stage_target(policy, scenario, stream_index, stage)
        if availability is not None:
            requested = _fallback_for_model(
                availability,
                stage.model_id,
                requested,
                snapshot.active_devices,
                stage.unusable_devices,
            )
        spread_multi = snapshot.mode == DeviceMode.SPREAD and snapshot.density > 1
        if requested.startswith("CPU"):
            cpu_only = (
                snapshot.active_devices == ("CPU",)
                or snapshot.mode != DeviceMode.SPREAD
            )
            cpu_streams = None
            if spread_multi and not cpu_only:
                cpu_streams = max(
                    1,
                    (psutil.cpu_count(logical=False) or 1) // snapshot.density,
                )
            compile_config = RunnerCompileConfig(
                performance_hint="THROUGHPUT",
                cpu_num_streams=cpu_streams,
            )
        else:
            compile_config = RunnerCompileConfig(
                performance_hint="THROUGHPUT" if spread_multi else "LATENCY"
            )
        assignments.append(
            StageAssignment(
                stage=stage.stage,
                model_id=stage.model_id,
                intended_device=intended,
                requested_device=requested,
                compile_config=compile_config,
            )
        )
    return tuple(assignments)


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


class ScenarioStreamWorker(QThread):
    #: How far this worker may fall behind the wall clock before it starts
    #: dropping frames. The display clock and this worker read the same file
    #: through separate captures, so they stay aligned only while inference keeps
    #: up with real time. Three frames at 24 fps is 125 ms: enough to absorb
    #: ordinary jitter without dropping steadily.
    CATCH_UP_THRESHOLD_FRAMES = 3.0

    #: How far this worker may trail the display clock before it seeks forward to
    #: rejoin it. Rate-matching alone does not close an accumulated offset, which
    #: is why a device toggle left the boxes describing an earlier moment.
    RESYNC_THRESHOLD_FRAMES = 6

    frame_ready = Signal(int, object)
    placement_ready = Signal(int, object, int, float)
    failed = Signal(int, str)

    def __init__(
        self,
        stream_index: int,
        scenario: ScenarioConfig,
        catalog: ScenarioCatalog,
        camera_index: int | None,
        registry: ScenarioModelRegistry,
        assignments: tuple[StageAssignment, ...],
        policy_sequence: int,
        npu_duty_cycle: NpuDutyCycle,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.stream_index = stream_index
        self.scenario = scenario
        self.catalog = catalog
        self.camera_index = camera_index
        self.registry = registry
        self.assignments = assignments
        self.npu_duty_cycle = npu_duty_cycle
        self._runtime_updates: queue.Queue[
            tuple[
                int,
                ScenarioConfig,
                tuple[StageAssignment, ...],
                float,
            ]
        ] = queue.Queue()
        self._policy_sequence = policy_sequence
        self._requested_at = time.monotonic()
        self._event_trackers: dict[str, EventTracker] = {
            scenario.id: EventTracker(scenario, stream_index)
        }
        self._event_tracker = self._event_trackers[scenario.id]
        self._latest_placements: dict[str, RunnerInfo] = {}
        self._fallbacks: list[str] = []
        # Wall-clock anchor for keeping this capture level with the display
        # clock; see the catch-up block in the frame loop. Initialised here so
        # a worker that has not started yet still reports sane values.
        self._sync_started = time.perf_counter()
        self._sync_frames = 0
        #: Frames skipped because this worker had fallen behind real time.
        self.dropped_frames = 0
        #: The display clock's most recent frame index. The clock is the
        #: reference the picture is drawn from, so this worker seeks to it when
        #: it has fallen too far behind. Updated from the UI thread; an int
        #: assignment is atomic enough for a display reference that is
        #: re-published every frame.
        self._reference_frame = -1
        #: Frames seeked over to rejoin the display clock.
        self.resynced_frames = 0
        #: Last observed capture position, for diagnostics.
        self.last_position = -1
        # stage -> device string that failed to compile for that stage. Used so a known-failing
        # device is not retried on every frame, while a policy change to a different device
        # string clears the match and retries.
        self._cpu_fallback_devices: dict[str, str] = {}
        # Baked CLIP text embeddings are configuration, not per-frame work, so
        # they are loaded once per worker and reused for every frame. Keyed by
        # (model, vocabulary) because two scenarios can share one CLIP model
        # with different vocabularies.
        self._zero_shot_cache: dict[
            tuple[str, str | None],
            tuple[np.ndarray, tuple[str, ...], tuple[int, ...]],
        ] = {}

    def set_runtime(
        self,
        sequence: int,
        scenario: ScenarioConfig,
        assignments: tuple[StageAssignment, ...],
        requested_at: float,
    ) -> None:
        while True:
            try:
                self._runtime_updates.get_nowait()
            except queue.Empty:
                break
        self._policy_sequence = sequence
        self._requested_at = requested_at
        self._runtime_updates.put(
            (sequence, scenario, assignments, requested_at)
        )

    def set_policy(
        self,
        sequence: int,
        assignments: tuple[StageAssignment, ...],
        requested_at: float,
    ) -> None:
        self.set_runtime(
            sequence,
            self.scenario,
            assignments,
            requested_at,
        )

    def _open_capture(
        self,
        scenario: ScenarioConfig,
    ) -> tuple[cv2.VideoCapture, bool]:
        if self.camera_index is not None:
            capture = cv2.VideoCapture(self.camera_index)
            loop = False
        else:
            media_path = self.catalog.media_path(scenario.id, camera_index=None)
            if media_path is None:
                raise ValueError("looping scenario requires a media path")
            capture = cv2.VideoCapture(str(media_path))
            loop = True
        if not capture.isOpened():
            raise RuntimeError(f"stream {self.stream_index} could not open scenario video")
        return capture, loop

    def _create_runner(
        self,
        assignment: StageAssignment,
    ) -> OpenVINOSingleRunner:
        bundle = self.registry.bundle(assignment.model_id)
        store = self.registry.store(assignment.model_id)
        return OpenVINOSingleRunner(
            bundle.model_path,
            assignment.requested_device,
            self.registry.cache_dir,
            self.npu_duty_cycle,
            compiled_store=store,
            compile_config=assignment.compile_config,
        )

    def _create_runner_or_cpu_fallback(
        self,
        assignment: StageAssignment,
    ) -> OpenVINOSingleRunner:
        """Compile the assigned device, degrading the stage to CPU instead of killing the stream.

        The startup availability gate is a cached probe, so a compile or placement failure can
        still happen later: a driver update can invalidate the NPU blob cache, an NPU compile can
        fail transiently, or an AUTO placement can land outside the priority list. The spec
        requires that case to keep the tile live with a labelled
        ``ran on CPU (<device> compile failed)`` badge (5.2/5.5), not to end the stream. If the
        CPU fallback also fails the stage genuinely cannot run, so the exception propagates to the
        stream's failure path.
        """

        try:
            return self._create_runner(assignment)
        except Exception as exc:
            if assignment.requested_device == "CPU":
                raise
            LOGGER.error(
                "stream %s stage %s failed to compile on %s (%s: %s); falling back to CPU",
                self.stream_index,
                assignment.stage,
                assignment.requested_device,
                type(exc).__name__,
                exc,
            )
            fallback = replace(
                assignment,
                requested_device="CPU",
                compile_config=RunnerCompileConfig(performance_hint="THROUGHPUT"),
            )
            runner = self._create_runner(fallback)
            self._cpu_fallback_devices[assignment.stage] = assignment.requested_device
            message = (
                f"{assignment.stage} ran on CPU ({assignment.requested_device} compile failed)"
            )
            if message not in self._fallbacks:
                self._fallbacks.append(message)
            return runner

    def _record_placement(
        self,
        assignment: StageAssignment,
        runner: OpenVINOSingleRunner,
    ) -> None:
        self._latest_placements[assignment.stage] = runner.info
        roots = {
            value.upper().split(".", 1)[0]
            for value in runner.info.execution_devices
        }
        if (
            assignment.intended_device in {"NPU", "GPU"}
            and not assignment.requested_device.startswith("AUTO:")
            and assignment.intended_device not in roots
            and self._cpu_fallback_devices.get(assignment.stage) != assignment.requested_device
        ):
            message = (
                f"{assignment.stage} ran on {','.join(runner.info.execution_devices)} "
                f"({assignment.intended_device} unavailable/off)"
            )
            if message not in self._fallbacks:
                self._fallbacks.append(message)

    def _placement_payload(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.id,
            "placements": {
                stage: asdict(info)
                for stage, info in self._latest_placements.items()
            },
            "fallbacks": list(self._fallbacks),
        }

    def _run_detector(
        self,
        frame: np.ndarray,
        assignment: StageAssignment,
        runner: OpenVINOSingleRunner,
    ) -> tuple[tuple[Detection, ...], StageMetric]:
        bundle = self.registry.bundle(assignment.model_id)
        metadata = (
            load_preprocess_config(bundle.source_config_path)
            if bundle.source_config_path is not None
            else {}
        )
        preprocess_started = time.perf_counter()
        if assignment.model_id.startswith("yolo11"):
            preprocessed = preprocess_yolo(
                frame,
                model_width=runner.model_width,
                model_height=runner.model_height,
                metadata=metadata,
            )
        else:
            preprocessed = preprocess_omz_image(frame, runner.input_spec)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        output, inference_ms = runner.infer(preprocessed.tensor)
        postprocess_started = time.perf_counter()
        rules = self.scenario.event_rules
        if assignment.model_id.startswith("yolo11"):
            detections = postprocess_yolo(
                output,
                labels=bundle.labels,
                transform=preprocessed.transform,
                confidence_threshold=float(rules["confidence_min"]),
                iou_threshold=float(rules["iou_threshold"]),
                max_detections=int(rules["max_detections"]),
            )
        else:
            detections = postprocess_ssd(
                output,
                labels=bundle.labels,
                transform=preprocessed.transform,
                confidence_threshold=float(rules["confidence_min"]),
                iou_threshold=float(rules["iou_threshold"]),
                max_detections=int(rules["max_detections"]),
            )
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        return detections, StageMetric(
            stage=assignment.stage,
            model_id=assignment.model_id,
            requested_device=assignment.requested_device,
            execution_devices=runner.info.execution_devices,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            output_count=len(detections),
            inference_count=1,
        )

    def _run_classifier(
        self,
        frame: np.ndarray,
        detections: tuple[Detection, ...],
        assignment: StageAssignment,
        runner: OpenVINOSingleRunner,
    ) -> tuple[tuple[Detection, ...], StageMetric]:
        bundle = self.registry.bundle(assignment.model_id)
        metadata = load_preprocess_config(bundle.source_config_path)
        top_k = int(self.scenario.event_rules["classify_top_k"])
        minimum = float(self.scenario.event_rules["classification_min"])
        candidates = classification_candidates(
            detections,
            confidence_min=float(self.scenario.event_rules["confidence_min"]),
            top_k=top_k,
            detector_labels=self.scenario.event_rules.get("classify_detector_labels", ()),
        )
        classified = list(detections)
        classified_by_id = {id(item): index for index, item in enumerate(classified)}
        preprocess_ms = 0.0
        inference_ms = 0.0
        classified_count = 0
        postprocess_started = time.perf_counter()
        for candidate in candidates:
            crop = crop_detection(frame, candidate)
            preprocessed = preprocess_classification(
                crop,
                runner.input_spec,
                metadata,
            )
            preprocess_ms += preprocessed.duration_ms
            output, duration_ms = runner.infer(preprocessed.tensor)
            inference_ms += duration_ms
            predictions = postprocess_classification(
                output,
                labels=bundle.labels,
                top_k=top_k,
            )
            if not predictions or predictions[0][1] < minimum:
                continue
            # Second gate: the predicted ImageNet class must be one this
            # scenario is prepared to defend as a candidate. Anything else is
            # dropped here rather than surfaced in the overlay.
            if not classification_allowed(
                candidate.label,
                predictions[0][0],
                self.scenario.event_rules.get("classify_output_labels", {}),
            ):
                continue
            index = classified_by_id[id(candidate)]
            classified[index] = candidate.with_classification(
                predictions[0][0],
                predictions[0][1],
            )
            classified_count += 1
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        return tuple(classified), StageMetric(
            stage=assignment.stage,
            model_id=assignment.model_id,
            requested_device=assignment.requested_device,
            execution_devices=runner.info.execution_devices,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            output_count=classified_count,
            inference_count=len(candidates),
        )

    def _run_zero_shot(
        self,
        frame: np.ndarray,
        detections: tuple[Detection, ...],
        assignment: StageAssignment,
        runner: OpenVINOSingleRunner,
    ) -> tuple[tuple[Detection, ...], StageMetric]:
        """Name each candidate crop with CLIP against a declared vocabulary.

        Unlike the ImageNet classifier, the output set is bounded by the
        stage's baked ``vocabulary_<name>.json``, which is generated from
        ``tools/clip_vocabulary.py``. CLIP can only return one of the words the
        operator declared for this scenario, so the overlay cannot invent a
        label.
        """

        bundle = self.registry.bundle(assignment.model_id)
        rules = self.scenario.event_rules
        top_k = int(rules.get("classify_top_k", 1))
        minimum = float(rules["zero_shot_min"])
        min_crop = float(rules.get("classify_min_crop_px", 0.0))
        candidates = classification_candidates(
            detections,
            confidence_min=float(rules["confidence_min"]),
            top_k=top_k,
            detector_labels=rules.get("classify_detector_labels", ()),
            min_crop_pixels=min_crop,
            # A zero-shot vocabulary is declared by the operator for this exact
            # scene, so a person crop is a legitimate question here ("is this
            # worker in PPE?"). The ImageNet path keeps the ban.
            allow_person_labels=True,
        )
        if not candidates:
            return detections, _idle_stage_metric(assignment, runner, 0, 0)

        vocabulary_name = next(
            (
                stage.vocabulary
                for stage in self.scenario.stages
                if stage.stage == assignment.stage
            ),
            None,
        )
        embeddings, labels, template_counts = self._zero_shot_vocabulary(
            assignment.model_id, bundle, name=vocabulary_name
        )
        crop_mode = str(rules.get("classify_crop", "box")).lower()
        head_fraction = float(rules.get("classify_head_fraction", 0.38))
        head_widen = float(rules.get("classify_head_widen", 1.3))
        classified = list(detections)
        classified_by_id = {id(item): index for index, item in enumerate(classified)}
        preprocess_ms = 0.0
        inference_ms = 0.0
        classified_count = 0
        postprocess_started = time.perf_counter()
        for candidate in candidates:
            if crop_mode == "head":
                crop = crop_detection_head(
                    frame, candidate, head_fraction, head_widen
                )
            else:
                crop = crop_detection(frame, candidate)
            if crop.size == 0:
                continue
            started = time.perf_counter()
            tensor = preprocess_clip(crop)
            preprocess_ms += (time.perf_counter() - started) * 1000.0
            output, duration_ms = runner.infer(tensor)
            inference_ms += duration_ms
            predictions = postprocess_zero_shot(
                output,
                embeddings,
                labels,
                template_counts,
                top_k,
            )
            if not predictions or predictions[0][1] < minimum:
                continue
            classified[classified_by_id[id(candidate)]] = candidate.with_classification(
                predictions[0][0],
                predictions[0][1],
            )
            classified_count += 1
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        return tuple(classified), StageMetric(
            stage=assignment.stage,
            model_id=assignment.model_id,
            requested_device=assignment.requested_device,
            execution_devices=runner.info.execution_devices,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            output_count=classified_count,
            inference_count=len(candidates),
        )

    def _zero_shot_vocabulary(
        self,
        model_id: str,
        bundle: ModelBundle,
        name: str | None = None,
    ) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
        """Load and memoise the baked CLIP text embeddings for a vocabulary.

        ``name`` is the stage's declared vocabulary; a missing named pair
        falls back to the default one inside :func:`load_zero_shot_vocabulary`,
        so a partially built model directory cannot fail the stream here.
        """

        cached = self._zero_shot_cache.get((model_id, name))
        if cached is None:
            vocabulary_path = bundle.model_path.parent / "vocabulary.json"
            cached = load_zero_shot_vocabulary(vocabulary_path, name=name)
            self._zero_shot_cache[(model_id, name)] = cached
        return cached

    def _run_pose(
        self,
        frame: np.ndarray,
        detections: tuple[Detection, ...],
        assignment: StageAssignment,
        runner: OpenVINOSingleRunner,
    ) -> tuple[
        tuple[Detection, ...],
        tuple[float, ...],
        StageMetric,
    ]:
        bundle = self.registry.bundle(assignment.model_id)
        preprocessed = preprocess_omz_image(frame, runner.input_spec)
        outputs, inference_ms = runner.infer_all(preprocessed.tensor)
        if len(outputs) < 2:
            raise RuntimeError("pose model did not return heatmap and PAF outputs")
        postprocess_started = time.perf_counter()
        keypoints = decode_pose(
            outputs[1],
            outputs[0],
            bundle.keypoint_names,
            preprocessed.transform,
            float(self.scenario.event_rules["pose_keypoint_threshold"]),
        )
        angle = confident_pose_angle(
            keypoints,
            float(self.scenario.event_rules["pose_keypoint_threshold"]),
        )
        updated = list(detections)
        if updated:
            target = max(updated, key=lambda item: item.width * item.height)
            index = updated.index(target)
            updated[index] = target.with_keypoints(keypoints)
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        return (
            tuple(updated),
            (angle,) if angle is not None else (),
            StageMetric(
                stage=assignment.stage,
                model_id=assignment.model_id,
                requested_device=assignment.requested_device,
                execution_devices=runner.info.execution_devices,
                preprocess_ms=preprocessed.duration_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
                output_count=len(keypoints),
                inference_count=1,
            ),
        )

    def _prepare_scenario_runtime(
        self,
        scenario: ScenarioConfig,
        assignments: tuple[StageAssignment, ...],
        *,
        runner_cache: dict[tuple[str, str], OpenVINOSingleRunner],
        capture_cache: dict[
            str,
            tuple[cv2.VideoCapture, bool, float | None],
        ],
        rate_cache: dict[str, _ProcessingRate],
        frame_index_cache: dict[str, int],
        scenario_changed: bool,
    ) -> tuple[
        dict[str, OpenVINOSingleRunner],
        cv2.VideoCapture,
        bool,
        float | None,
    ]:
        self.scenario = scenario
        self.assignments = assignments
        tracker = self._event_trackers.get(scenario.id)
        if tracker is None:
            tracker = EventTracker(scenario, self.stream_index)
            self._event_trackers[scenario.id] = tracker
        elif scenario_changed:
            tracker.reset(scenario)
        self._event_tracker = tracker
        capture_state = capture_cache.get(scenario.id)
        if capture_state is None:
            capture, loop = self._open_capture(scenario)
            source_fps_raw = float(capture.get(cv2.CAP_PROP_FPS))
            source_fps = source_fps_raw if source_fps_raw > 0 else None
            capture_state = (capture, loop, source_fps)
            capture_cache[scenario.id] = capture_state
        capture, loop, source_fps = capture_state
        active_runners: dict[str, OpenVINOSingleRunner] = {}
        placements: dict[str, RunnerInfo] = {}
        self._fallbacks = []
        for assignment in assignments:
            if assignment.stage in LOGICAL_STAGES:
                continue
            key = (scenario.id, assignment.stage)
            runner = runner_cache.get(key)
            needs_compile = (
                runner is None
                or runner.device != assignment.requested_device
                or runner.compile_config != assignment.compile_config
            )
            if (
                needs_compile
                and runner is not None
                and self._cpu_fallback_devices.get(assignment.stage) == assignment.requested_device
            ):
                # This stage already degraded to CPU on this exact device string; retrying it
                # every frame would spam the log and stall the tile. A policy change to a
                # different device clears the match and is retried.
                needs_compile = False
            if needs_compile:
                runner = self._create_runner_or_cpu_fallback(assignment)
                runner_cache[key] = runner
            if runner is None:  # unreachable: needs_compile is True whenever the cache misses
                raise RuntimeError(
                    f"stream {self.stream_index} stage {assignment.stage} has no runner"
                )
            active_runners[assignment.stage] = runner
            placements[assignment.stage] = runner.info
            self._record_placement(assignment, runner)
        self._latest_placements = placements
        rate_cache.setdefault(scenario.id, _ProcessingRate())
        frame_index_cache.setdefault(scenario.id, 0)
        return active_runners, capture, loop, source_fps

    def set_reference_frame(self, frame_index: int) -> None:
        """Publish the display clock's current frame as the sync reference.

        Called from the UI thread for every clock frame. The overlay draws that
        frame, so detections must describe it; the worker seeks to it from its
        own thread when it has fallen too far behind. A plain int assignment is
        sufficient -- the value is republished on the next frame regardless, so a
        torn read costs nothing.
        """
        self._reference_frame = int(frame_index)

    def run(self) -> None:
        capture_cache: dict[
            str,
            tuple[cv2.VideoCapture, bool, float | None],
        ] = {}
        runner_cache: dict[tuple[str, str], OpenVINOSingleRunner] = {}
        rate_cache: dict[str, _ProcessingRate] = {}
        frame_index_cache: dict[str, int] = {}
        event_times: dict[str, deque[float]] = {}
        try:
            active_runners, capture, loop, source_fps = (
                self._prepare_scenario_runtime(
                    self.scenario,
                    self.assignments,
                    runner_cache=runner_cache,
                    capture_cache=capture_cache,
                    rate_cache=rate_cache,
                    frame_index_cache=frame_index_cache,
                    scenario_changed=False,
                )
            )
            event_times[self.scenario.id] = deque(maxlen=600)
            self.placement_ready.emit(
                self.stream_index,
                self._placement_payload(),
                self._policy_sequence,
                (time.monotonic() - self._requested_at) * 1000.0,
            )
            next_deadline = time.perf_counter()
            # Level the wall-clock anchor with the freshly opened capture.
            self._sync_started = time.perf_counter()
            self._sync_frames = 0
            while not self.isInterruptionRequested():
                while True:
                    try:
                        sequence, scenario, assignments, requested_at = (
                            self._runtime_updates.get_nowait()
                        )
                    except queue.Empty:
                        break
                    scenario_changed = scenario.id != self.scenario.id
                    active_runners, capture, loop, source_fps = (
                        self._prepare_scenario_runtime(
                            scenario,
                            assignments,
                            runner_cache=runner_cache,
                            capture_cache=capture_cache,
                            rate_cache=rate_cache,
                            frame_index_cache=frame_index_cache,
                            scenario_changed=scenario_changed,
                        )
                    )
                    event_times.setdefault(
                        scenario.id,
                        deque(maxlen=600),
                    )
                    self.placement_ready.emit(
                        self.stream_index,
                        self._placement_payload(),
                        sequence,
                        (time.monotonic() - requested_at) * 1000.0,
                    )
                    next_deadline = time.perf_counter()
                    # Switching scenario reopens the capture; the anchor must
                    # move with it or the first frame after the switch looks
                    # wildly behind and triggers a spurious catch-up.
                    self._sync_started = time.perf_counter()
                    self._sync_frames = 0

                end_to_end_started = time.perf_counter()
                # The display clock and this worker read the same file through
                # separate captures, so they stay aligned only while inference
                # keeps up with real time. Toggling the NPU and GPU off drops
                # everything to CPU, inference falls behind, and this capture
                # lags -- and it never recovers on its own, because the pacing at
                # the end of this loop resyncs the clock while the capture keeps
                # handing back stale sequential frames. The symptom a viewer sees
                # is boxes drawn on items that have already left the screen.
                # Dropping the frames we have fallen behind by re-levels the two.
                if source_fps is not None:
                    behind = (
                        (time.perf_counter() - self._sync_started) * source_fps
                        - self._sync_frames
                    )
                    if behind > self.CATCH_UP_THRESHOLD_FRAMES:
                        dropped = 0
                        while dropped < int(behind) and capture.grab():
                            dropped += 1
                            self._sync_frames += 1
                        self.dropped_frames += dropped
                    # Rate-matching keeps this worker level with the clock, but it
                    # cannot close an offset already accumulated -- which is what
                    # a device toggle leaves behind. Seek forward to rejoin the
                    # frame the viewer is actually looking at.
                    if self._reference_frame >= 0:
                        try:
                            position = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                        except Exception:  # noqa: BLE001 - backend-dependent
                            position = -1
                        self.last_position = position
                        gap = self._reference_frame - position
                        if position >= 0 and gap > self.RESYNC_THRESHOLD_FRAMES:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, self._reference_frame)
                            self.resynced_frames += gap
                            self._sync_frames += gap
                decode_started = time.perf_counter()
                ok, frame = capture.read()
                self._sync_frames += 1
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
                if not ok or frame is None:
                    if loop:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = capture.read()
                    if not ok or frame is None:
                        if loop:
                            raise RuntimeError("scenario loop could not restart")
                        self.msleep(10)
                        next_deadline = time.perf_counter()
                        continue
                frame_index_cache[self.scenario.id] += 1
                detections: tuple[Detection, ...] = ()
                raw_detections: tuple[Detection, ...] = ()
                stage_metrics: list[StageMetric] = []
                posture_angles: tuple[float, ...] = ()
                for assignment in self.assignments:
                    if assignment.stage in LOGICAL_STAGES:
                        continue
                    runner = active_runners[assignment.stage]
                    if assignment.stage == "classifier":
                        detections, metric = self._run_classifier(
                            frame,
                            detections,
                            assignment,
                            runner,
                        )
                    elif assignment.stage == "product_classifier":
                        detections, metric = self._run_zero_shot(
                            frame,
                            detections,
                            assignment,
                            runner,
                        )
                    elif assignment.stage == "pose":
                        detections, posture_angles, metric = self._run_pose(
                            frame,
                            detections,
                            assignment,
                            runner,
                        )
                    else:
                        stage_detections, metric = self._run_detector(
                            frame,
                            assignment,
                            runner,
                        )
                        detections = detections + stage_detections
                        raw_detections = raw_detections + stage_detections
                    stage_metrics.append(metric)
                postprocess_started = time.perf_counter()
                tracked, events = self._event_tracker.update(
                    detections,
                    posture_angles=posture_angles,
                    source_size=(frame.shape[1], frame.shape[0]),
                )
                event_postprocess_ms = (
                    time.perf_counter() - postprocess_started
                ) * 1000.0
                if event_postprocess_ms > 0.0:
                    event_stage = next(
                        (
                            stage.stage
                            for stage in self.scenario.stages
                            if stage.stage in LOGICAL_STAGES
                        ),
                        "events",
                    )
                    stage_metrics.append(
                        StageMetric(
                            stage=event_stage,
                            model_id="cpu",
                            requested_device="CPU",
                            execution_devices=("CPU",),
                            preprocess_ms=0.0,
                            inference_ms=0.0,
                            postprocess_ms=event_postprocess_ms,
                            output_count=len(events),
                            inference_count=0,
                        )
                    )
                completed_at = time.perf_counter()
                now = time.time()
                scenario_event_times = event_times[self.scenario.id]
                for event in events:
                    scenario_event_times.append(event.ts)
                cutoff = now - 60.0
                while scenario_event_times and scenario_event_times[0] < cutoff:
                    scenario_event_times.popleft()
                elapsed = max(completed_at - end_to_end_started, 1e-9)
                fallbacks = self._fallbacks
                self._fallbacks = []
                metrics = PipelineFrameMetrics(
                    stream_index=self.stream_index,
                    scenario_id=self.scenario.id,
                    frame_index=frame_index_cache[self.scenario.id],
                    captured_at=now,
                    source_fps=source_fps,
                    processing_fps=rate_cache[self.scenario.id].sample(completed_at),
                    inferences_per_second=(
                        sum(metric.inference_count for metric in stage_metrics)
                        / elapsed
                    ),
                    decode_ms=decode_ms,
                    preprocess_ms=sum(metric.preprocess_ms for metric in stage_metrics),
                    inference_ms=sum(metric.inference_ms for metric in stage_metrics),
                    postprocess_ms=sum(metric.postprocess_ms for metric in stage_metrics),
                    end_to_end_ms=elapsed * 1000.0,
                    detection_count=len(tracked),
                    detections_per_second=len(tracked) / elapsed,
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                    stages=tuple(stage_metrics),
                    event_counts_60s=self._event_tracker.rolling_counts(now),
                )
                self.frame_ready.emit(
                    self.stream_index,
                    PipelineFrame(
                        stream_index=self.stream_index,
                        scenario_id=self.scenario.id,
                        detections=tracked,
                        raw_detections=raw_detections,
                        events=events,
                        metrics=metrics,
                        placements=dict(self._latest_placements),
                        zones=self.scenario.zones,
                        fallbacks=tuple(fallbacks),
                    ),
                )
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
            for capture, _, _ in capture_cache.values():
                capture.release()
            runner_cache.clear()



"""Phase 3 scenario graphs, model registry, and multi-stage stream workers."""

from __future__ import annotations

import json
import queue
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
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
    decode_pose,
    load_labels,
    load_preprocess_config,
    postprocess_classification,
    postprocess_ssd,
    postprocess_yolo,
    preprocess_classification,
    preprocess_omz_image,
    preprocess_yolo,
)
from app.scenarios.medical import confident_pose_angle
from app.scenarios.retail import classification_candidates
from app.telemetry.npu_fallback import NpuDutyCycle


LOGICAL_STAGES = {"zone_event", "posture_event", "track_event"}


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
            if entry["source"] == "huggingface":
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
        if stage.device_pref == "NPU" and policy.npu_enabled:
            return "NPU", "NPU"
        if stage.device_pref == "GPU" and policy.gpu_enabled:
            return "GPU", "GPU"
        return "CPU", stage.device_pref

    auxiliary = stage.stage not in {"detector", "perimeter_detector", "plate_detector"}
    if auxiliary:
        if stage.device_pref == "NPU" and not policy.npu_enabled:
            return "CPU", "NPU"
        if stage.device_pref == "GPU" and not policy.gpu_enabled:
            return "CPU", "GPU"
        return stage.device_pref, stage.device_pref
    requested = policy.device_for_stream(stream_index, stage.device_pref)
    return requested, requested


def _fallback_for_model(
    availability: AvailabilityMatrix,
    model_id: str,
    requested: str,
    active_devices: tuple[str, ...],
) -> str:
    if requested.startswith("AUTO:"):
        priorities = tuple(
            device
            for device in requested.split(":", 1)[1].split(",")
            if device
        )
        supported = tuple(
            device for device in priorities if availability.supports(model_id, device)
        )
        return "AUTO:" + ",".join(supported) if supported else "CPU"
    if availability.supports(model_id, requested):
        return requested
    for device in active_devices:
        if device != requested and availability.supports(model_id, device):
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
            if (
                runner is None
                or runner.device != assignment.requested_device
                or runner.compile_config != assignment.compile_config
            ):
                runner = self._create_runner(assignment)
                runner_cache[key] = runner
            active_runners[assignment.stage] = runner
            placements[assignment.stage] = runner.info
            self._record_placement(assignment, runner)
        self._latest_placements = placements
        rate_cache.setdefault(scenario.id, _ProcessingRate())
        frame_index_cache.setdefault(scenario.id, 0)
        return active_runners, capture, loop, source_fps

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

                end_to_end_started = time.perf_counter()
                decode_started = time.perf_counter()
                ok, frame = capture.read()
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
                    stage_metrics.append(metric)
                postprocess_started = time.perf_counter()
                tracked, events = self._event_tracker.update(
                    detections,
                    posture_angles=posture_angles,
                )
                event_postprocess_ms = (
                    time.perf_counter() - postprocess_started
                ) * 1000.0
                if event_postprocess_ms > 0.0:
                    stage_metrics.append(
                        StageMetric(
                            stage="events",
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

"""Regression test for the stream-death fix.

A compile or placement failure *after* the startup availability gate (a driver update
invalidating the NPU blob cache, a transient NPU error, an AUTO placement landing outside the
priority list) used to end the stream for the rest of the session. The spec requires the stage to
degrade to CPU with a labelled badge instead, and to keep the tile live.

``_create_runner_or_cpu_fallback`` is exercised unbound against a stub so the real method body is
under test without needing OpenVINO models on disk.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.engine.pipelines import ScenarioStreamWorker, StageAssignment
from app.engine.runner import RunnerCompileConfig


class _StubWorker:
    stream_index = 0

    def __init__(self, failing_devices: set[str]) -> None:
        self._failing_devices = failing_devices
        self._fallbacks: list[str] = []
        self._cpu_fallback_devices: dict[str, str] = {}
        self.attempts: list[str] = []

    def _create_runner(self, assignment: StageAssignment) -> SimpleNamespace:
        self.attempts.append(assignment.requested_device)
        if assignment.requested_device in self._failing_devices:
            raise RuntimeError(
                f"placement assertion failed: requested {assignment.requested_device}"
            )
        return SimpleNamespace(device=assignment.requested_device)


def _assignment(device: str, stage: str = "detector") -> StageAssignment:
    return StageAssignment(
        stage=stage,
        model_id="yolo11n-fp16",
        intended_device="NPU",
        requested_device=device,
        compile_config=RunnerCompileConfig(performance_hint="LATENCY"),
    )


class StreamCompileFallbackTests(unittest.TestCase):
    def test_npu_compile_failure_degrades_to_cpu_and_is_labelled(self) -> None:
        worker = _StubWorker(failing_devices={"NPU"})

        runner = ScenarioStreamWorker._create_runner_or_cpu_fallback(worker, _assignment("NPU"))  # type: ignore[arg-type]

        self.assertEqual(worker.attempts, ["NPU", "CPU"])
        self.assertEqual(runner.device, "CPU")
        self.assertEqual(
            worker._fallbacks, ["detector ran on CPU (NPU compile failed)"]
        )
        self.assertEqual(worker._cpu_fallback_devices, {"detector": "NPU"})

    def test_successful_compile_is_untouched(self) -> None:
        worker = _StubWorker(failing_devices=set())

        runner = ScenarioStreamWorker._create_runner_or_cpu_fallback(worker, _assignment("GPU"))  # type: ignore[arg-type]

        self.assertEqual(worker.attempts, ["GPU"])
        self.assertEqual(runner.device, "GPU")
        self.assertEqual(worker._fallbacks, [])
        self.assertEqual(worker._cpu_fallback_devices, {})

    def test_cpu_failure_propagates_instead_of_looping(self) -> None:
        worker = _StubWorker(failing_devices={"NPU", "CPU"})

        with self.assertRaises(RuntimeError):
            ScenarioStreamWorker._create_runner_or_cpu_fallback(worker, _assignment("NPU"))  # type: ignore[arg-type]

        # One NPU attempt then one CPU attempt; no further retries.
        self.assertEqual(worker.attempts, ["NPU", "CPU"])

    def test_a_later_policy_change_retries_the_new_device(self) -> None:
        worker = _StubWorker(failing_devices={"NPU"})
        ScenarioStreamWorker._create_runner_or_cpu_fallback(worker, _assignment("NPU"))

        runner = ScenarioStreamWorker._create_runner_or_cpu_fallback(worker, _assignment("GPU"))  # type: ignore[arg-type]

        self.assertEqual(worker.attempts, ["NPU", "CPU", "GPU"])
        self.assertEqual(runner.device, "GPU")


if __name__ == "__main__":
    unittest.main()

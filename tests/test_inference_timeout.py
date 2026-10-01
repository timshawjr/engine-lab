"""Tests for the synchronous inference path in OpenVINOSingleRunner.

The federal hang was a GIL deadlock, not a slow inference. ``AsyncInferQueue`` routes
every call through OpenVINO's *Python* data_dispatcher wrapper, which holds the GIL while it
works. With several streams calling it at once the contention was severe enough that a sibling
thread could not run a plain numpy call: the captured hang showed three workers blocked inside
``data_dispatcher._data_dispatch`` and a fourth starved in ``np.ascontiguousarray``. Windows
then killed the process as an Application Hang.

A synchronous ``InferRequest`` is a direct C call that releases the GIL while it runs, so
concurrent streams cannot deadlock against each other. These tests pin that.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from app.engine.runner import OpenVINOSingleRunner
from app.telemetry.npu_fallback import NpuDutyCycle


class _DutyCycle:
    def record_inference(self, *args, **kwargs) -> None:  # pragma: no cover - not exercised
        pass


class SynchronousInferenceTests(unittest.TestCase):
    """A real model is compiled so the behaviour is exercised against real OpenVINO."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.cache = Path(self._directory.name) / "cache"
        self.runner = OpenVINOSingleRunner(
            Path("models") / "yolo11n-fp16" / "yolo11n.xml",
            "GPU",
            self.cache,
            _DutyCycle(),
        )
        self.addCleanup(self._release)

    def _release(self) -> None:
        # Drop the request without waiting on it.
        self.runner._infer_request = None

    def _tensor(self) -> np.ndarray:
        return np.zeros((1, 3, 640, 640), dtype=np.float32)

    def test_uses_a_synchronous_infer_request(self) -> None:
        """The runner must not use AsyncInferQueue.

        AsyncInferQueue's Python data_dispatcher wrapper is what deadlocked the streams.
        """

        self.assertFalse(hasattr(self.runner, "_queue"))
        self.assertTrue(hasattr(self.runner, "_infer_request"))

    def test_concurrent_streams_do_not_deadlock(self) -> None:
        """Four streams inferring at once must all complete.

        This is the regression. Under AsyncInferQueue the GIL contention between concurrent
        streams was enough to hang the process until Windows killed it.

        Each stream gets its own runner, exactly as the app builds one per StreamWorker, so
        no two threads share an InferRequest.
        """

        # One shared store, exactly as the app builds one per model and every stream draws
        # from it. This is the configuration that deadlocked: concurrent infer() on a
        # CompiledModel whose data_dispatcher is shared by all of its requests.
        from app.engine.runner import CompiledModelStore

        store = CompiledModelStore(
            Path("models") / "yolo11n-fp16" / "yolo11n.xml", self.cache
        )
        runners = [
            OpenVINOSingleRunner(
                Path("models") / "yolo11n-fp16" / "yolo11n.xml",
                "GPU",
                self.cache,
                _DutyCycle(),
                compiled_store=store,
            )
            for _ in range(4)
        ]
        self.addCleanup(lambda: [setattr(r, "_infer_request", None) for r in runners])

        results: list[float] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(4)

        def worker(runner: OpenVINOSingleRunner) -> None:
            try:
                barrier.wait(timeout=10)
                _, duration_ms = runner.infer(self._tensor())
                results.append(duration_ms)
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(runner,)) for runner in runners
        ]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        elapsed = time.perf_counter() - started

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 4, "not all concurrent inferences completed")
        # Four concurrent inferences on one GPU must still finish in a sane time. If the GIL
        # were contended the way it was, this would hang far past this bound.
        self.assertLess(elapsed, 30.0)

    def test_healthy_inference_is_unaffected(self) -> None:
        """The change must not break ordinary single-stream work."""

        outputs, duration_ms = self.runner.infer(self._tensor())
        self.assertIsNotNone(outputs)
        self.assertGreater(outputs.size, 0)
        self.assertGreater(duration_ms, 0.0)


if __name__ == "__main__":
    unittest.main()

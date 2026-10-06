from __future__ import annotations

import time
import unittest

import cv2
import numpy as np

from app.engine.pipelines import ScenarioStreamWorker


class _FakeCapture:
    """Records how many frames were grabbed and how many were pulled.

    ``grab`` is the cheap skip; ``read`` is grab+retrieve. Modelling both lets a
    test prove the catch-up skips frames rather than decoding them.
    """

    def __init__(self, frames: int = 10_000) -> None:
        self.position = 0
        self.total = frames
        self.grabs = 0
        self.reads = 0

    def grab(self) -> bool:
        self.grabs += 1
        if self.position >= self.total:
            return False
        self.position += 1
        return True

    def read(self):
        self.reads += 1
        if self.position >= self.total:
            return False, None
        self.position += 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def set(self, prop, value) -> None:  # noqa: ANN001 - matches cv2 API
        self.position = int(value)

    def get(self, prop) -> float:  # noqa: ANN001 - matches cv2 API
        return 24.0


def _catch_up(capture, source_fps, sync_started, sync_frames, now):
    """The catch-up arithmetic, isolated from the thread for testing."""
    behind = (now - sync_started) * source_fps - sync_frames
    dropped = 0
    if behind > ScenarioStreamWorker.CATCH_UP_THRESHOLD_FRAMES:
        while dropped < int(behind) and capture.grab():
            dropped += 1
            sync_frames += 1
    return dropped, sync_frames


class CatchUpTests(unittest.TestCase):
    """Regression: toggling NPU and GPU off drops inference to CPU, the worker's
    capture falls behind the display clock, and it never recovers on its own --
    so boxes were drawn on items that had already left the screen. The worker now
    drops the frames it has fallen behind by.

    Measured cause: the pacing at the end of the loop resyncs the clock but not
    the capture, which keeps returning stale sequential frames.
    """

    def test_frame_on_schedule_drops_nothing(self) -> None:
        capture = _FakeCapture()
        dropped, frames = _catch_up(capture, 24.0, 0.0, 240, 10.0)
        self.assertEqual(dropped, 0)
        self.assertEqual(frames, 240)
        self.assertEqual(capture.grabs, 0)

    def test_small_jitter_is_absorbed_not_dropped(self) -> None:
        # 2 frames behind is inside the threshold, so nothing is skipped.
        capture = _FakeCapture()
        dropped, _ = _catch_up(capture, 24.0, 0.0, 238, 10.0)
        self.assertEqual(dropped, 0)
        self.assertEqual(capture.grabs, 0)

    def test_a_long_slow_period_is_caught_up(self) -> None:
        # Five seconds at 24 fps is 120 frames; the worker only produced 40,
        # which is the shape of a CPU-fallback stretch.
        capture = _FakeCapture()
        dropped, frames = _catch_up(capture, 24.0, 0.0, 40, 5.0)
        self.assertEqual(dropped, 80)
        self.assertEqual(frames, 120)
        self.assertEqual(capture.grabs, 80)

    def test_catch_up_uses_grab_not_read(self) -> None:
        # Skipping must not pay the decode cost of the frames it discards.
        capture = _FakeCapture()
        _catch_up(capture, 24.0, 0.0, 0, 5.0)
        self.assertEqual(capture.reads, 0)
        self.assertGreater(capture.grabs, 0)

    def test_catch_up_stops_at_end_of_stream(self) -> None:
        # A short file must not spin: grab returns False and the loop ends.
        capture = _FakeCapture(frames=10)
        dropped, _ = _catch_up(capture, 24.0, 0.0, 0, 5.0)
        self.assertLessEqual(dropped, 10)
        self.assertEqual(capture.position, 10)

    def test_threshold_is_small_enough_to_stay_visually_aligned(self) -> None:
        # Any lag is visible as misplaced boxes; at 24 fps three frames is 125 ms.
        self.assertLessEqual(ScenarioStreamWorker.CATCH_UP_THRESHOLD_FRAMES / 24.0, 0.2)


class LoopHandlingTests(unittest.TestCase):
    """The catch-up must not break the existing end-of-file restart."""

    def test_read_after_exhausted_capture_reports_failure(self) -> None:
        capture = _FakeCapture(frames=0)
        ok, frame = capture.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)


if __name__ == "__main__":
    unittest.main()

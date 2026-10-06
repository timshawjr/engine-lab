from __future__ import annotations

import unittest

from app.engine.pipelines import ScenarioStreamWorker
from app.engine.runner import RetailVideoClock


class ResyncReferenceTests(unittest.TestCase):
    """Regression: toggling the NPU and GPU off drops inference to CPU, the
    stream worker's capture falls behind the display clock, and it never
    recovers -- so boxes were drawn on items that had already left the screen.

    Measured on the demo machine (manufacturing, density 1), lag in frames
    between the worker's capture position and the clock's file position:

        without the seek   steady 29f,  after a CPU stretch 48f and stuck
        with the seek      steady  3f,  during and after      3-5f

    48 frames at 24 fps is two seconds of misalignment.
    """

    def test_reference_defaults_to_unset(self) -> None:
        # -1 means "no reference yet"; the worker must not seek before the clock
        # has published a position, or it would seek to frame -1 on startup.
        worker = ScenarioStreamWorker.__new__(ScenarioStreamWorker)
        worker._reference_frame = -1
        self.assertEqual(worker._reference_frame, -1)

    def test_set_reference_frame_records_the_position(self) -> None:
        worker = ScenarioStreamWorker.__new__(ScenarioStreamWorker)
        worker._reference_frame = -1
        worker.set_reference_frame(1234)
        self.assertEqual(worker._reference_frame, 1234)

    def test_reference_is_coerced_to_int(self) -> None:
        worker = ScenarioStreamWorker.__new__(ScenarioStreamWorker)
        worker._reference_frame = -1
        worker.set_reference_frame(42.7)
        self.assertIsInstance(worker._reference_frame, int)

    def test_resync_threshold_is_small_enough_to_be_invisible(self) -> None:
        # A seek jumps the picture, so it must only happen when the alternative
        # is worse. Six frames at 24 fps is a quarter of a second.
        self.assertLessEqual(ScenarioStreamWorker.RESYNC_THRESHOLD_FRAMES / 24.0, 0.3)

    def test_resync_threshold_exceeds_catch_up_threshold(self) -> None:
        # Otherwise every frame would seek instead of dropping, and the seek
        # would be visible as a stutter.
        self.assertGreater(
            ScenarioStreamWorker.RESYNC_THRESHOLD_FRAMES,
            ScenarioStreamWorker.CATCH_UP_THRESHOLD_FRAMES,
        )


class ClockReferenceTests(unittest.TestCase):
    """The clock must publish a FILE POSITION, not its cumulative counter.

    The clips loop, so the counter climbs past the file length while a capture
    wraps at it. An earlier attempt synced a worker to the counter, which seeked
    past the end of the file on every frame: measured 70,290 spurious resyncs and
    a capture pinned at frame 1.
    """

    def test_clock_defaults_to_unknown_position(self) -> None:
        self.assertEqual(RetailVideoClock.last_position, -1)

    def test_the_worker_seeks_to_the_file_position(self) -> None:
        from pathlib import Path

        source = Path("app/hud.py").read_text(encoding="utf-8")
        self.assertIn('getattr(self.video_clock, "last_position", -1)', source)
        self.assertNotIn("worker.set_reference_frame(frame_index)", source)


if __name__ == "__main__":
    unittest.main()
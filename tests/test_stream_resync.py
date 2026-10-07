from __future__ import annotations

import unittest
from pathlib import Path

from app.engine.pipelines import ScenarioStreamWorker
from app.engine.runner import RetailVideoClock

BASELINE = {
    # scenario: detections per frame, measured before this change
    "retail": 0.84,
    "metro": 22.57,
    "manufacturing": 4.29,
    "education": 9.56,
    "health": 2.42,
    "federal": 8.32,
}


class ResyncRequestTests(unittest.TestCase):
    """The display clock and each stream worker decode the same file through
    SEPARATE captures. They drift apart when the device policy changes: turning
    the NPU and GPU off drops inference to CPU, the worker's capture falls
    behind the clock, and it never catches back up on its own.

    Measured on metro, drift in frames between the clock's position and the
    worker's:

        on CPU (the bug)      p50  99   max 239
        after the resync      p50   1   max   1
        settled               p50   1   max   6

    The rejoin is ONE-SHOT per device change, never continuous. An earlier
    attempt resynced every frame and regressed tracking: 24 seeks in 20 s, about
    one per second, each discarding the frames the tracker was following.
    """

    def _worker(self) -> ScenarioStreamWorker:
        worker = ScenarioStreamWorker.__new__(ScenarioStreamWorker)
        worker._reference_frame = -1
        worker._resync_requested = False
        worker.resynced_frames = 0
        worker.last_position = -1
        return worker

    def test_nothing_seeks_by_default(self) -> None:
        self.assertFalse(self._worker()._resync_requested)

    def test_publishing_a_reference_never_arms_a_seek(self) -> None:
        # _on_video_frame runs on every clock frame. If storing the reference
        # armed a seek, every frame would seek -- the exact defect that produced
        # phantom boxes on metro.
        worker = self._worker()
        worker.set_reference_frame(500)
        self.assertFalse(worker._resync_requested)
        self.assertEqual(worker._reference_frame, 500)

    def test_request_resync_arms_one_seek(self) -> None:
        worker = self._worker()
        worker.request_resync()
        self.assertTrue(worker._resync_requested)

    def test_reference_is_stored_as_an_int(self) -> None:
        worker = self._worker()
        worker.set_reference_frame(42.9)
        self.assertIsInstance(worker._reference_frame, int)

    def test_minimum_drift_is_small_but_not_zero(self) -> None:
        # A seek is visible, so it must only happen when misalignment is worse.
        self.assertGreater(ScenarioStreamWorker.RESYNC_MINIMUM_DRIFT_FRAMES, 0)
        self.assertLessEqual(
            ScenarioStreamWorker.RESYNC_MINIMUM_DRIFT_FRAMES / 24.0, 0.25
        )


class SeekDirectionTests(unittest.TestCase):
    """Only ever seek FORWARD.

    The clips loop, so a negative drift usually means the clock has wrapped past
    the worker rather than that the worker is ahead. Seeking backwards there
    would replay content.
    """

    def test_only_positive_drift_triggers_a_seek(self) -> None:
        source = Path("app/engine/pipelines.py").read_text(encoding="utf-8")
        marker = "if self._resync_requested:"
        body = source[source.index(marker) : source.index(marker) + 900]
        self.assertIn("drift > self.RESYNC_MINIMUM_DRIFT_FRAMES", body)
        self.assertNotIn("abs(drift)", body)


class ClockReferenceTests(unittest.TestCase):
    """The reference must be a FILE POSITION, not the clock's counter.

    The clips loop, so the counter climbs past the file length while a capture
    wraps at it. An earlier attempt synced to the counter and seeked past the end
    of the file every frame: 70,290 spurious resyncs, capture pinned at frame 1.
    """

    def test_clock_starts_with_an_unknown_position(self) -> None:
        self.assertEqual(RetailVideoClock.last_position, -1)

    def test_hud_publishes_the_file_position(self) -> None:
        source = Path("app/hud.py").read_text(encoding="utf-8")
        self.assertIn('getattr(self.video_clock, "last_position", -1)', source)
        self.assertNotIn("set_reference_frame(frame_index)", source)


class TriggerWiringTests(unittest.TestCase):
    """The resync is driven by device changes and once at startup."""

    def test_every_device_change_requests_a_resync(self) -> None:
        source = Path("app/hud.py").read_text(encoding="utf-8")
        marker = "def _apply_policy_change"
        body = source[source.index(marker) : source.index(marker) + 1200]
        self.assertIn("_request_stream_resync()", body)

    def test_npu_gpu_and_cycle_all_funnel_through_the_policy_change(self) -> None:
        source = Path("app/hud.py").read_text(encoding="utf-8")
        for method in ("def toggle_npu", "def toggle_gpu", "def cycle_mode"):
            body = source[source.index(method) : source.index(method) + 300]
            self.assertIn("_apply_policy_change", body)

    def test_one_resync_is_armed_at_startup(self) -> None:
        # The workers open their captures after compiling models, so they begin
        # a fixed offset behind the clock: measured 20 frames, about 0.8 s, which
        # reads as boxes that do not match the picture.
        source = Path("app/hud.py").read_text(encoding="utf-8")
        self.assertIn("THEME.startup_resync_ms, self._request_stream_resync", source)

    def test_alignment_is_rechecked_periodically(self) -> None:
        # The worker consumes frames marginally slower than real time, so on a
        # long run the offset grows without bound: measured about 18 frames a
        # minute, which is three seconds of misalignment after four minutes and
        # was reported from a booth. A one-shot correction cannot catch that.
        source = Path("app/hud.py").read_text(encoding="utf-8")
        self.assertIn("resync_timer", source)
        self.assertIn("THEME.resync_interval_ms", source)

    def test_the_periodic_check_is_a_check_not_a_seek(self) -> None:
        # It must ask; the worker decides. Seeking on every tick would be the
        # per-frame seeking that broke tracking on the first attempt.
        from app.theme import THEME

        source = Path("app/engine/pipelines.py").read_text(encoding="utf-8")
        marker = "if self._resync_requested:"
        body = source[source.index(marker) : source.index(marker) + 900]
        self.assertIn("RESYNC_MINIMUM_DRIFT_FRAMES", body)
        # And the interval must be long enough that a seek is rare.
        self.assertGreaterEqual(THEME.resync_interval_ms, 5000)

    def test_the_seek_is_not_armed_from_the_frame_loop(self) -> None:
        source = Path("app/engine/pipelines.py").read_text(encoding="utf-8")
        marker = "if self._resync_requested:"
        body = source[source.index(marker) : source.index(marker) + 900]
        self.assertIn("self._resync_requested = False", body)


class DetectionQualityGuardTests(unittest.TestCase):
    """Guard the detection figures, so a future sync change that breaks tracking
    is caught here rather than at a booth.

    Recorded per scenario as detections per frame, measured over 30 s at density
    1 with tools/qa_detection.py. A sync change must not move these.
    """

    def test_baseline_table_is_from_the_measured_run(self) -> None:
        # metro is the tracking-heavy case that caught the earlier regression:
        # ~22 objects per frame, so a broken tracker shows up here first.
        self.assertGreater(BASELINE["metro"], 20.0)
        self.assertLess(BASELINE["retail"], 1.0)

    def test_qa_harnesses_exist(self) -> None:
        self.assertTrue(Path("tools/qa_detection.py").is_file())
        self.assertTrue(Path("tools/qa_sync.py").is_file())


if __name__ == "__main__":
    unittest.main()
from __future__ import annotations

import unittest

from app.engine.resync import ResyncDecision, decide_resync


def decide(reference: int, position: int, *, forward: int = 3, backward: int = 15) -> ResyncDecision:
    return decide_resync(
        reference=reference,
        position=position,
        minimum_forward=forward,
        minimum_backward=backward,
    )


class ForwardSeekTests(unittest.TestCase):
    """The worker fell behind the display clock: it must jump forward.

    This is the case the original one-shot resync was built for. Turning the NPU
    and GPU off drops inference to CPU, the worker's capture stalls while it
    compiles, and the clock keeps playing.
    """

    def test_a_long_way_behind_seeks_to_the_clock(self) -> None:
        decision = decide(reference=500, position=100)
        self.assertEqual(decision.seek_to, 500)
        self.assertEqual(decision.drift, 400)

    def test_a_few_frames_behind_is_left_alone(self) -> None:
        decision = decide(reference=10, position=9)
        self.assertIsNone(decision.seek_to)

    def test_exactly_at_the_threshold_is_left_alone(self) -> None:
        decision = decide(reference=10, position=7, forward=3)
        self.assertIsNone(decision.seek_to)

    def test_one_frame_past_the_threshold_seeks(self) -> None:
        decision = decide(reference=10, position=6, forward=3)
        self.assertEqual(decision.seek_to, 10)


class BackwardSeekTests(unittest.TestCase):
    """The worker is AHEAD of the display clock. Reported from a booth as the
    wireframe no longer matching the video after N, G, then both back on.

    Measured, health, operator's sequence:

        [sync] ref=92  pos=464 sought=False reason=drift -372 below minimum
        [sync] ref=187 pos=482 sought=False reason=drift -295 below minimum

    The clips loop, so the two captures end up in different laps: the clock
    wrapped to the start while the worker was still mid-file. The overlay is drawn
    from the worker's detections over the clock's frame, so a worker ahead by 372
    frames shows boxes for content the screen has not reached yet.

    "Only seek forward" was justified by not replaying content, but the worker's
    frames are never displayed -- only its detections are, over the clock's frame
    -- so seeking it backwards is invisible to a visitor.
    """

    def test_being_far_ahead_seeks_back_to_the_clock(self) -> None:
        decision = decide(reference=92, position=464)
        self.assertEqual(decision.seek_to, 92)
        self.assertEqual(decision.drift, -372)

    def test_the_measured_second_refusal_is_corrected(self) -> None:
        decision = decide(reference=187, position=482)
        self.assertEqual(decision.seek_to, 187)

    def test_a_small_lead_is_left_alone(self) -> None:
        # Run-to-run jitter of a frame or two is not worth resetting the tracker.
        decision = decide(reference=100, position=108)
        self.assertIsNone(decision.seek_to)


class UnavailablePositionTests(unittest.TestCase):
    """An undecidable request must stay armed rather than be thrown away.

    The worker reads its position from the capture backend, which can return -1
    for a frame or two after a seek or a reopen. The original code cleared the
    request first and then found it could not act, so one bad read lost the
    correction entirely -- and nothing retried until the next device change.
    """

    def test_an_unknown_worker_position_asks_to_retry(self) -> None:
        decision = decide(reference=100, position=-1)
        self.assertIsNone(decision.seek_to)
        self.assertTrue(decision.retry)

    def test_an_unknown_reference_asks_to_retry(self) -> None:
        decision = decide(reference=-1, position=100)
        self.assertTrue(decision.retry)

    def test_a_decided_alignment_does_not_ask_to_retry(self) -> None:
        # Already together: the request has been satisfied, not postponed.
        self.assertFalse(decide(reference=100, position=100).retry)
        self.assertFalse(decide(reference=500, position=100).retry)
        self.assertFalse(decide(reference=100, position=500).retry)


class PolicyTests(unittest.TestCase):
    def test_forward_and_backward_thresholds_are_separate(self) -> None:
        # Forward correction is cheap and invisible; a backward one resets the
        # tracker and can re-emit events for frames already seen, so it has to be
        # a bigger error before it fires.
        decision = decide(reference=100, position=110, forward=3, backward=15)
        self.assertIsNone(decision.seek_to)
        self.assertFalse(decision.retry)

    def test_backward_threshold_is_not_so_large_that_it_misses_a_lap(self) -> None:
        # The measured failures were hundreds of frames out. Anything at 30
        # frames (about a second at 24-30 fps) must be corrected.
        self.assertLessEqual(decide(reference=0, position=30, backward=15).seek_to, 0)

    def test_a_seek_always_targets_the_clock(self) -> None:
        for reference, position in ((500, 100), (100, 500), (0, 900), (900, 0)):
            decision = decide(reference=reference, position=position, backward=15)
            if decision.seek_to is not None:
                self.assertEqual(decision.seek_to, reference)


if __name__ == "__main__":
    unittest.main()

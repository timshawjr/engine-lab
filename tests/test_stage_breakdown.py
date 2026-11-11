from __future__ import annotations

import unittest

from app.engine.placement_share import PlacementShares, normalise_shares

INTERVAL = 0.5
SMOOTHING = 0.25
PRECISION = 2


def shares() -> PlacementShares:
    return PlacementShares(
        interval_s=INTERVAL, smoothing=SMOOTHING, precision=PRECISION
    )


class NormaliseTests(unittest.TestCase):
    def test_shares_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(normalise_shares([0.3, 0.3, 0.4], 2)), 1.0, places=6)

    def test_rounding_drift_lands_on_the_last_segment(self) -> None:
        # Three equal thirds cannot be represented exactly; the last absorbs the
        # drift so it stays flush with the bar's right edge.
        self.assertAlmostEqual(sum(normalise_shares([1.0, 1.0, 1.0], 2)), 1.0, places=6)

    def test_negative_samples_are_clamped(self) -> None:
        result = normalise_shares([-1.0, 1.0], 2)
        self.assertEqual(result[0], 0.0)

    def test_all_zero_falls_back_to_equal_shares(self) -> None:
        result = normalise_shares([0.0, 0.0, 0.0], 2)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(sum(result), 1.0, places=6)

    def test_empty_is_empty(self) -> None:
        self.assertEqual(normalise_shares([], 2), [])

    def test_quantised_to_the_requested_precision(self) -> None:
        for value in normalise_shares([0.123456, 0.876544], 2):
            self.assertEqual(round(value, 2), value)


class PlacementShareTests(unittest.TestCase):
    """The placement bar used to redraw its segments from live inference timings
    on every repaint. Those vary frame to frame, so the segments -- and the labels
    centred inside them -- resized and re-elided continuously. Reported from a
    booth as text-box jitter.
    """

    def test_first_sample_is_adopted_outright(self) -> None:
        bar = shares()
        bar.update([10.0, 30.0], now=0.0)
        self.assertAlmostEqual(bar.shares[0], 0.25, places=2)
        self.assertAlmostEqual(bar.shares[1], 0.75, places=2)

    def test_shares_change_only_at_the_throttled_cadence(self) -> None:
        # 100 samples over one second with alternating durations. With a 0.5 s
        # interval only two updates may land, so the drawn bar must take at most
        # three distinct values in total (the initial one plus two updates).
        bar = shares()
        bar.update([10.0, 10.0], now=0.0)
        seen = {tuple(bar.shares)}
        for step in range(1, 101):
            value = 10.0 if step % 2 else 40.0
            bar.update([value, 10.0], now=step * 0.01)
            seen.add(tuple(bar.shares))
        self.assertLessEqual(
            len(seen),
            3,
            f"the bar changed {len(seen)} times in one second; it should be throttled",
        )

    def test_shares_move_once_the_interval_elapses(self) -> None:
        bar = shares()
        bar.update([10.0, 10.0], now=0.0)
        bar.update([90.0, 10.0], now=INTERVAL + 0.01)
        self.assertGreater(bar.shares[0], 0.5)

    def test_a_single_outlier_does_not_snap_the_bar(self) -> None:
        bar = shares()
        bar.update([10.0, 90.0], now=0.0)
        start = bar.shares[0]
        bar.update([90.0, 10.0], now=INTERVAL + 0.01)
        self.assertGreater(bar.shares[0], start, "it should move toward the sample")
        self.assertLess(bar.shares[0], 0.9, "it must not jump all the way in one step")

    def test_a_changed_stage_count_is_adopted_immediately(self) -> None:
        # Holding stale shares for a different number of stages would misalign
        # the bar against the labels.
        bar = shares()
        bar.update([10.0, 10.0], now=0.0)
        changed = bar.update([10.0, 10.0, 20.0], now=0.01)
        self.assertTrue(changed)
        self.assertEqual(len(bar.shares), 3)

    def test_zero_durations_give_equal_shares_not_a_division(self) -> None:
        bar = shares()
        bar.update([0.0, 0.0], now=0.0)
        self.assertAlmostEqual(bar.shares[0], 0.5, places=2)

    def test_no_stages_means_no_shares(self) -> None:
        bar = shares()
        bar.update([], now=0.0)
        self.assertEqual(bar.shares, [])

    def test_repeated_identical_samples_are_not_reported_as_changes(self) -> None:
        bar = shares()
        bar.update([10.0, 10.0], now=0.0)
        self.assertFalse(bar.update([10.0, 10.0], now=INTERVAL + 0.01))

    def test_interval_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            PlacementShares(interval_s=0, smoothing=0.25, precision=2)

    def test_smoothing_must_be_a_fraction(self) -> None:
        for bad in (0.0, 1.5):
            with self.assertRaises(ValueError):
                PlacementShares(interval_s=1.0, smoothing=bad, precision=2)

    def test_precision_must_not_be_negative(self) -> None:
        with self.assertRaises(ValueError):
            PlacementShares(interval_s=1.0, smoothing=0.25, precision=-1)


if __name__ == "__main__":
    unittest.main()
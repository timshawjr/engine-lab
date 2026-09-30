"""Honest idle-state classification for the engine gauges."""

from __future__ import annotations

import unittest

from app.hud import gauge_state
from app.theme import THEME


class GaugeStateTests(unittest.TestCase):
    def test_engine_with_no_stage_is_not_used(self):
        self.assertEqual(
            gauge_state("GPU", {"NPU", "CPU"}, False, None, None),
            "NOT USED BY THIS VERTICAL",
        )

    def test_used_but_quiet_engine_is_idle(self):
        self.assertEqual(
            gauge_state("GPU", {"NPU", "GPU"}, False, 0.0, 12.0),
            "IDLE",
        )

    def test_live_engine_is_active(self):
        self.assertEqual(
            gauge_state("NPU", {"NPU"}, False, 18.0, 0.2),
            "ACTIVE",
        )

    def test_operator_off_beats_everything(self):
        self.assertEqual(
            gauge_state("NPU", {"NPU"}, True, 18.0, 0.2),
            "OFF BY OPERATOR",
        )

    def test_operator_off_beats_not_used(self):
        self.assertEqual(
            gauge_state("GPU", {"NPU", "CPU"}, True, None, None),
            "OFF BY OPERATOR",
        )

    def test_active_requires_recent_measurement(self):
        threshold = THEME.telemetry_recent_seconds
        # A non-zero measurement exactly at the threshold is still recent.
        self.assertEqual(
            gauge_state("NPU", {"NPU"}, False, 1.0, threshold),
            "ACTIVE",
        )
        # The same measurement a hair past the threshold is stale, not active.
        self.assertEqual(
            gauge_state("NPU", {"NPU"}, False, 1.0, threshold + 0.001),
            "IDLE",
        )

    def test_zero_value_is_never_active(self):
        # A used engine with a fresh but zero reading is idle, never active.
        self.assertEqual(
            gauge_state("NPU", {"NPU"}, False, 0.0, 0.0),
            "IDLE",
        )

    def test_missing_value_is_idle_not_active(self):
        # No numeric reading yet (e.g. "NO COUNTER") is idle, not active.
        self.assertEqual(
            gauge_state("NPU", {"NPU"}, False, None, 0.2),
            "IDLE",
        )


if __name__ == "__main__":
    unittest.main()

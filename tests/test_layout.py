from __future__ import annotations

import unittest

from app.layout import use_compact_layout


class LayoutDecisionTests(unittest.TestCase):
    def test_1080p_uses_full_layout(self) -> None:
        self.assertIs(use_compact_layout(1920, 1080), False)

    def test_ultrawide_uses_full_layout(self) -> None:
        self.assertIs(use_compact_layout(2293, 960), False)

    def test_4k_uses_full_layout(self) -> None:
        self.assertIs(use_compact_layout(3840, 2160), False)

    def test_small_window_goes_compact(self) -> None:
        self.assertIs(use_compact_layout(1280, 720), True)

    def test_exactly_1600x900_is_full(self) -> None:
        self.assertIs(use_compact_layout(1600, 900), False)

from __future__ import annotations

import unittest
from pathlib import Path

from app.theme import THEME


class GaugeColumnFitTests(unittest.TestCase):
    """Three engine tiles share the gauge column, and each needs enough height
    for its own content: header + value line + bar + gap.

    Measured at 1920x1080 before this was fixed, the column offered 137 px per
    tile against a 139 px requirement, so the bottom tile lost its bar and
    sparkline. The CPU readout was visibly truncated -- reported from a booth.

    After tightening the gap between tiles: 141 px per tile, and all three bars
    render at the same thickness.
    """

    def tile_content_height(self) -> int:
        return (
            THEME.gauge_header_height
            + round(THEME.gauge_value_font * THEME.gauge_value_line_ratio)
            + THEME.gauge_bar_min
            + THEME.gauge_gap_min
        )

    def test_the_gap_leaves_room_for_three_tiles(self) -> None:
        # The column is about 429 px tall at 1080p; three tiles plus two gaps
        # must fit inside that.
        available = 429
        needed = 3 * self.tile_content_height() + 2 * THEME.gauge_tile_spacing
        self.assertLessEqual(
            needed,
            available,
            f"three tiles need {needed} px but the column offers {available}",
        )

    def test_the_gap_is_tighter_than_general_panel_spacing(self) -> None:
        # It was deliberately reduced from spacing_sm to make the tiles fit.
        self.assertLess(THEME.gauge_tile_spacing, THEME.spacing_sm)

    def test_the_gap_is_not_so_tight_that_tiles_touch(self) -> None:
        self.assertGreater(THEME.gauge_tile_spacing, 0)

    def test_a_tile_taller_than_its_content_is_not_clipped(self) -> None:
        # 141 px is what the column actually grants at 1080p.
        self.assertGreaterEqual(141, self.tile_content_height())

    def test_the_tile_has_a_hard_floor_below_the_content_height(self) -> None:
        # The floor only matters on a very short window; it must not itself force
        # the column to overflow at a normal size.
        floor = THEME.gauge_min_height
        self.assertLess(floor, self.tile_content_height())

    def test_the_gap_comes_from_the_theme_not_a_literal(self) -> None:
        source = Path("app/hud.py").read_text(encoding="utf-8")
        self.assertIn("THEME.gauge_tile_spacing", source)


if __name__ == "__main__":
    unittest.main()
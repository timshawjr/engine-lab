"""Layout invariants for the booth HUD.

Cheap guards on decisions that are easy to undo by accident. The video size in
particular was measured wrong twice during this work: once by eyeballing a
screenshot, and once with a brightness mask that mistook the placement bar for
footage. `fitted_video_size` is the arithmetic both the canvas and the tests
share, so the picture size can be asserted instead of guessed.
"""

from __future__ import annotations

import unittest

from app.hud import SCENARIO_ORDER, fitted_video_size
from app.theme import THEME


class FittedVideoSizeTests(unittest.TestCase):
    def test_sixteen_by_nine_clip_is_height_limited(self) -> None:
        """The canvas is wider than 16:9, so height is what sizes the picture.

        This is why the layout change had to reclaim vertical space: adding width
        to the canvas would only widen the black bars.
        """
        width, height = fitted_video_size(1101, 482, 768, 432)
        self.assertAlmostEqual(height, 482.0, delta=0.01)
        self.assertAlmostEqual(width, 482 * 16 / 9, delta=0.01)

    def test_measured_area_matches_the_shipped_layout(self) -> None:
        """Guards the measured before/after: 94,044 px2 -> 413,020 px2."""
        short_w, short_h = fitted_video_size(1101, 230, 768, 432)
        tall_w, tall_h = fitted_video_size(1101, 482, 768, 432)
        self.assertGreater(short_w * short_h, 93_000.0)
        self.assertGreater(tall_w * tall_h, 412_000.0)
        # The picture is now more than four times the area it had.
        self.assertGreater((tall_w * tall_h) / (short_w * short_h), 4.0)

    def test_narrow_canvas_is_width_limited(self) -> None:
        width, height = fitted_video_size(400, 900, 768, 432)
        self.assertAlmostEqual(width, 400.0, delta=0.01)
        self.assertAlmostEqual(height, 225.0, delta=0.01)

    def test_aspect_ratio_is_preserved(self) -> None:
        for canvas in ((1101, 482), (400, 900), (1920, 200)):
            width, height = fitted_video_size(*canvas, 1280, 720)
            self.assertAlmostEqual(width / height, 1280 / 720, places=3)

    def test_degenerate_inputs_return_zero(self) -> None:
        self.assertEqual(fitted_video_size(0, 482, 768, 432), (0.0, 0.0))
        self.assertEqual(fitted_video_size(1101, 0, 768, 432), (0.0, 0.0))
        self.assertEqual(fitted_video_size(1101, 482, 0, 432), (0.0, 0.0))
        self.assertEqual(fitted_video_size(1101, 482, 768, 0), (0.0, 0.0))


class OverlayTypographyTests(unittest.TestCase):
    def test_detection_descriptor_font_is_small(self) -> None:
        """Descriptors sit on top of the footage.

        They used to render at the overlay font size, which covered the very
        vehicles and pedestrians the operator is meant to be looking at.
        """
        self.assertLess(THEME.font_detection_label, THEME.font_regular)
        self.assertLessEqual(THEME.font_detection_label, 16)

    def test_label_row_height_tracks_the_descriptor_font(self) -> None:
        """A row shorter than its text smears adjacent labels together."""
        self.assertGreaterEqual(
            THEME.overlay_label_height, THEME.font_detection_label
        )

    def test_overlay_font_is_no_larger_than_the_body_font(self) -> None:
        """The header badge and event strip use the overlay font.

        It was reduced from 22 to 18 so the badge stops covering the top of the
        picture; matching the body font is acceptable, larger is not.
        """
        self.assertLessEqual(THEME.font_overlay, THEME.font_regular)
        self.assertLess(THEME.font_overlay, THEME.font_title)


class HeaderStatTypographyTests(unittest.TestCase):
    def test_header_stat_value_font_is_booth_readable(self) -> None:
        """These three readouts replaced the LIVE badge and the wall clock.

        They have to be legible from a standing distance, so the value font must
        be clearly larger than its caption and no smaller than the body font.
        """
        self.assertGreater(
            THEME.header_stat_value_font, THEME.header_stat_caption_font
        )
        self.assertGreaterEqual(THEME.header_stat_value_font, THEME.font_regular)


class DemoPageStructureTests(unittest.TestCase):
    def test_scenario_keys_cover_every_vertical(self) -> None:
        self.assertEqual(len(SCENARIO_ORDER), 7)
        self.assertEqual(len(set(SCENARIO_ORDER)), 7)


if __name__ == "__main__":
    unittest.main()
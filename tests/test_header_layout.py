from __future__ import annotations

import unittest
from pathlib import Path

from app.theme import THEME


class HeaderStatWidthTests(unittest.TestCase):
    """The header readouts update several times a second, so their boxes must not
    be sized to their text.

    Measured before the fix, sampling widget geometry over 25 s on metro:
    the three stats showed 6, 20 and 10 distinct widths, one slid 63 px
    horizontally, and the title panel beside them cycled through 24 widths. The
    whole strip bounced as digits changed.

    After sizing the value label from a fixed template: one width for every
    widget, x pinned, nothing moved.
    """

    def test_value_template_is_wide_enough_for_real_values(self) -> None:
        # Largest measured values: metro DET/s 651.8 at density 8, EVENTS 1681.
        # The template must fit those, or a fixed label would clip them.
        template = THEME.header_stat_value_template
        self.assertGreaterEqual(len(template), 5)
        self.assertIn(".", template)  # FPS and DET/s carry one decimal place

    def test_value_template_is_not_wastefully_wide(self) -> None:
        # An over-wide box steals horizontal room from the platform panel beside
        # it, which then wraps or clips. Measured: 6 characters is enough for
        # every value any scenario produces at any density.
        self.assertLessEqual(len(THEME.header_stat_value_template), 7)

    def test_stat_value_label_is_given_a_fixed_width(self) -> None:
        source = Path("app/hud.py").read_text(encoding="utf-8")
        marker = "class HeaderStat"
        body = source[source.index(marker) : source.index(marker) + 2200]
        self.assertIn("setFixedWidth", body)
        self.assertIn("header_stat_value_template", body)

    def test_the_width_comes_from_the_theme_not_a_literal(self) -> None:
        # AGENTS.md 10: layout tokens live in theme.py.
        source = Path("app/hud.py").read_text(encoding="utf-8")
        self.assertIn("THEME.header_stat_value_template", source)


if __name__ == "__main__":
    unittest.main()
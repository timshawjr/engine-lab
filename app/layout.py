"""Layout decisions for the booth HUD.

The demo runs on a 16:9 1080p display, while it is developed on a 2293x960
ultrawide. Both must take the full layout; only a genuinely small window goes
compact.
"""

from __future__ import annotations

FULL_LAYOUT_MIN_WIDTH = 1600
FULL_LAYOUT_MIN_HEIGHT = 900


def use_compact_layout(width: int, height: int) -> bool:
    """True only when the window is smaller than the full layout's minimum."""
    return width < FULL_LAYOUT_MIN_WIDTH or height < FULL_LAYOUT_MIN_HEIGHT

"""Central visual tokens for the Engine Lab booth UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    background: str = "#071018"
    panel: str = "#0D1B26"
    panel_alt: str = "#112635"
    border: str = "#28475A"
    text: str = "#F4F8FA"
    text_muted: str = "#9DB1BE"
    npu: str = "#42D9FF"
    gpu: str = "#69F0AE"
    cpu: str = "#FFD166"
    danger: str = "#FF5D73"
    warning: str = "#FFB454"
    overlay_fill: str = "#071018"
    overlay_text: str = "#FFFFFF"

    font_family: str = "Segoe UI"
    font_regular: int = 18
    font_semibold: int = 21
    font_metric: int = 30
    font_metric_compact: int = 24
    font_engine_value: int = 72
    font_title: int = 34
    font_badge: int = 18
    font_overlay: int = 22

    radius_small: int = 8
    radius_panel: int = 14
    spacing_xs: int = 6
    spacing_sm: int = 10
    spacing_md: int = 16
    spacing_lg: int = 24
    spacing_xl: int = 32

    gauge_height: int = 28
    overlay_label_height: int = 30
    overlay_badge_width: int = 300
    overlay_border: int = 3
    minimum_window_width: int = 1280
    minimum_window_height: int = 720
    design_width: int = 1920
    design_height: int = 1080
    tile_display_fps: int = 30
    stream_rate_window_seconds: float = 2.0
    telemetry_recent_seconds: float = 2.0
    streams_real_time_fraction: float = 0.90
    single_tile_row_max_height: int = 150
    multi_tile_row_max_height: int = 270


THEME = Theme()

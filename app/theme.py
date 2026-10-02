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
    font_overlay: int = 18
    # Descriptor drawn on each detection box ("vehicle 100% . car 30%"). Kept
    # deliberately small: these sit on top of the video, and at booth distance a
    # large label hides the very content the operator is meant to be looking at.
    font_detection_label: int = 13
    attract_title_font: int = 72
    attract_subtitle_font: int = 36
    attract_business_font: int = 28

    radius_small: int = 8
    radius_panel: int = 14
    spacing_xs: int = 6
    spacing_sm: int = 10
    spacing_md: int = 16
    spacing_lg: int = 24
    spacing_xl: int = 32

    gauge_height: int = 28
    # Engine-gauge tile layout. The tile renders its header, value and bar
    # from its actual widget height, so the column may compress the tiles
    # below the design height without truncating the value.
    gauge_min_height: int = 76
    gauge_min_height_compact: int = 115
    gauge_header_height: int = 28
    gauge_value_font: int = 56
    gauge_value_font_compact: int = 40
    gauge_value_font_min: int = 12
    gauge_value_font_compact_min: int = 10
    gauge_value_area_min: int = 20
    gauge_bar_min: int = 8
    gauge_gap_min: int = 4
    gauge_value_line_ratio: float = 1.77
    # Row height for on-video overlay text (detection descriptors, zone names,
    # the header badge and the event strip). Tracks font_detection_label so the
    # rows pack tightly instead of overprinting the footage.
    overlay_label_height: int = 18
    overlay_badge_width: int = 300
    overlay_border: int = 3
    minimum_window_width: int = 1280
    minimum_window_height: int = 720
    design_width: int = 1920
    design_height: int = 1080
    tile_display_fps: int = 30
    telemetry_interval_seconds: float = 0.2
    ticker_seconds: int = 8
    attract_scenario_seconds: int = 12
    stream_rate_window_seconds: float = 2.0
    # A telemetry sample counts as "recent" for a gauge's ACTIVE state while its
    # measurement is no older than this many seconds. Ten samples at the 5 Hz
    # telemetry cadence, so a briefly idle engine is not mistaken for a dead one.
    telemetry_recent_seconds: float = 2.0
    metric_window_seconds: float = 10.0
    sparkline_window_seconds: float = 60.0
    rss_sample_seconds: float = 5.0
    gauge_max_percent: float = 100.0
    percentile_p50: float = 50.0
    percentile_p95: float = 95.0
    clock_interval_ms: int = 1000
    operator_log_lines: int = 20
    operator_event_lines: int = 100
    bytes_per_mib: int = 1024 * 1024
    streams_real_time_fraction: float = 0.90
    real_time_equivalent_cap: float = 1.0
    single_tile_row_max_height: int = 280
    multi_tile_row_max_height: int = 390


THEME = Theme()

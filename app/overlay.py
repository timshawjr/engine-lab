"""Qt overlay painter for measured YOLO detections."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from app.engine.stages import Detection
from app.theme import THEME


def source_to_display_rect(
    detection: Detection,
    video_rect: QRectF,
    source_width: int,
    source_height: int,
) -> QRectF:
    if source_width <= 0 or source_height <= 0:
        return QRectF()
    x1 = video_rect.left() + detection.x1 / source_width * video_rect.width()
    y1 = video_rect.top() + detection.y1 / source_height * video_rect.height()
    x2 = video_rect.left() + detection.x2 / source_width * video_rect.width()
    y2 = video_rect.top() + detection.y2 / source_height * video_rect.height()
    return QRectF(x1, y1, x2 - x1, y2 - y1)


def draw_detections(
    painter: QPainter,
    video_rect: QRectF,
    detections: Sequence[Detection],
    source_width: int,
    source_height: int,
) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    font = QFont(THEME.font_family, THEME.font_overlay)
    font.setWeight(QFont.Weight.DemiBold)
    painter.setFont(font)
    pen = QPen(QColor(THEME.npu), THEME.overlay_border)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    for detection in detections:
        rect = source_to_display_rect(
            detection,
            video_rect,
            source_width,
            source_height,
        )
        if rect.isEmpty():
            continue
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        label = f"{detection.label}  {detection.confidence * 100.0:.0f}%"
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(label) + THEME.spacing_sm * 2
        text_height = THEME.overlay_label_height
        label_rect = QRectF(
            rect.left(),
            max(video_rect.top(), rect.top() - text_height),
            text_width,
            text_height,
        )
        painter.fillRect(label_rect, QColor(THEME.overlay_fill))
        painter.setPen(QPen(QColor(THEME.overlay_text)))
        painter.drawText(
            label_rect.adjusted(THEME.spacing_sm, 0, -THEME.spacing_sm, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            label,
        )
    painter.restore()

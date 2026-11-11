"""Throttled, smoothed segment shares for the pipeline-placement bar.

The bar draws one segment per pipeline stage, sized by that stage's share of the
measured duration. Those durations vary frame to frame, and drawing them directly
made the segments -- and the labels centred inside them -- resize and re-elide on
every repaint. On a booth screen that reads as the text box jittering.

Three things keep it still:

* **throttle** -- a new sample is only adopted every ``interval_s``; the bar
  redraws far more often than that.
* **smoothing** -- an adopted sample moves each share a fraction of the way, so
  one slow frame cannot snap a segment to a new width.
* **quantisation** -- shares are rounded before drawing, so a sub-percent wobble
  cannot move a segment edge by a pixel.

This lives outside the widget so it can be tested without a QApplication.
"""

from __future__ import annotations


def normalise_shares(shares: list[float], precision: int) -> list[float]:
    """Clamp, quantise and renormalise shares so they sum to exactly 1.

    The rounding drift is absorbed by the last segment so it stays flush with the
    right edge of the bar.
    """
    if not shares:
        return []
    cleaned = [max(0.0, value) for value in shares]
    total = sum(cleaned)
    if total <= 0.0:
        return [1.0 / len(shares)] * len(shares)
    rounded = [round(value / total, precision) for value in cleaned]
    rounded[-1] = round(rounded[-1] + (1.0 - sum(rounded)), precision)
    return rounded


class PlacementShares:
    """Holds the shares the bar draws, updating them at a fixed cadence."""

    def __init__(self, *, interval_s: float, smoothing: float, precision: int) -> None:
        if interval_s <= 0:
            raise ValueError("interval must be positive")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if precision < 0:
            raise ValueError("precision must not be negative")
        self._interval_s = interval_s
        self._smoothing = smoothing
        self._precision = precision
        self._shares: list[float] = []
        self._last_update: float | None = None

    @property
    def shares(self) -> list[float]:
        return list(self._shares)

    def update(self, durations: list[float], now: float) -> bool:
        """Adopt new durations if the cadence allows. Returns whether it changed.

        A change in stage count is adopted immediately: keeping stale shares for a
        different number of stages would misalign the bar against the labels.
        """
        sample = self._sample(durations)

        if not sample:
            changed = self._shares != []
            self._shares = []
            self._last_update = now
            return changed

        if not self._shares or len(self._shares) != len(sample):
            self._shares = sample
            self._last_update = now
            return True

        if self._last_update is not None and (now - self._last_update) < self._interval_s:
            return False

        weight = self._smoothing
        smoothed = [
            old * (1.0 - weight) + new * weight
            for old, new in zip(self._shares, sample)
        ]
        updated = normalise_shares(smoothed, self._precision)
        self._last_update = now
        # Report a change only when the drawn shares actually differ. After
        # quantisation they often do not, and repainting for an identical bar is
        # the churn this module exists to remove.
        changed = updated != self._shares
        self._shares = updated
        return changed

    def _sample(self, durations: list[float]) -> list[float]:
        cleaned = [max(0.0, value) for value in durations]
        total = sum(cleaned)
        if total <= 0.0:
            if not cleaned:
                return []
            return normalise_shares([1.0] * len(cleaned), self._precision)
        return normalise_shares([value / total for value in cleaned], self._precision)
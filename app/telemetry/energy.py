"""Turn the platform's energy counters into one steady number for the HUD.

The reading itself comes from Windows' ``Energy Meter`` performance counters,
which on this hardware are backed by Intel RAPL -- the processor's own energy
meter. Two things are handled here, away from the widgets so they can be tested:

* **which** counter to believe (see ``pdh.select_package_watts``)
* how to present a fast, fractionally-wobbling measurement as one readable
  number without it flickering

The presentation rule is deliberately boring: **the mean of the last second,
rounded to whole watts, published once a second.** That is what a power meter
shows, it is easy to explain to a visitor, and it cannot be accused of cherry-
picking a flattering sample. Painting the raw counter at 5 Hz onto a label is
the same mistake that made the placement bar jitter on screen.
"""

from __future__ import annotations


class PowerDisplay:
    """Holds the wattage the header tile shows.

    ``None`` means "not measurable here", never zero. A zero would read as
    "measured, and the chip drew nothing", which is a different claim.
    """

    def __init__(self, *, publish_interval_s: float) -> None:
        if publish_interval_s <= 0:
            raise ValueError("publish interval must be positive")
        self._publish_interval_s = publish_interval_s
        self._samples: list[float] = []
        self._value: int | None = None
        # Publish the first reading immediately rather than after a second of
        # blank tile.
        self._window_started = float("-inf")

    @property
    def value(self) -> int | None:
        """Whole watts, or None when no counter is readable."""
        return self._value

    def update(self, watts: float | None, now: float) -> bool:
        """Feed one reading. Returns whether the displayed number changed."""
        if watts is None or watts < 0.0:
            changed = self._value is not None
            self._value = None
            self._samples.clear()
            # Publish the next readable value immediately: a one-second glitch
            # should not leave the tile blank for another interval afterwards.
            self._window_started = float("-inf")
            return changed

        self._samples.append(float(watts))
        if now - self._window_started < self._publish_interval_s:
            return False

        mean = sum(self._samples) / len(self._samples)
        # Half-up, not round(): Python rounds 20.5 to 20, and a tile that reads
        # one watt low at the halfway point is a needless argument.
        published = int(mean + 0.5)
        changed = published != self._value
        self._value = published
        self._samples.clear()
        self._window_started = now
        return changed

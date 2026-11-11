from __future__ import annotations

import unittest

from app.telemetry.energy import PowerDisplay
from app.telemetry.pdh import select_package_watts


class PackageSelectionTests(unittest.TestCase):
    """Windows' ``Energy Meter`` counter set exposes several RAPL domains.

    Measured on the booth NUC:

        RAPL_Package0_DRAM   is memory, not the package
        RAPL_Package0_PKG    is the number we want
        RAPL_Package0_PP0    is the cores only
        _Total               reads 0.00 W unconditionally -- a trap

    Showing ``_Total`` would display zero watts while the demo runs flat out,
    which is worse than showing nothing.
    """

    def test_picks_the_package_domain(self) -> None:
        samples = [
            ("RAPL_Package0_DRAM", 0.11),
            ("RAPL_Package0_PKG", 20.41),
            ("RAPL_Package0_PP0", 11.85),
        ]
        self.assertAlmostEqual(select_package_watts(samples), 20.41)

    def test_never_picks_the_total_instance(self) -> None:
        # _Total is present on this machine and always reads zero.
        samples = [("_Total", 0.0), ("RAPL_Package0_PKG", 20.41)]
        self.assertAlmostEqual(select_package_watts(samples), 20.41)

    def test_only_total_means_unavailable_not_zero(self) -> None:
        # A zero would read as "measured, and the chip drew nothing".
        self.assertIsNone(select_package_watts([("_Total", 0.0)]))

    def test_no_domains_means_unavailable(self) -> None:
        self.assertIsNone(select_package_watts([]))

    def test_memory_and_cores_are_not_mistaken_for_the_package(self) -> None:
        samples = [("RAPL_Package0_DRAM", 0.35), ("RAPL_Package0_PP0", 11.9)]
        self.assertIsNone(select_package_watts(samples))

    def test_two_sockets_are_summed(self) -> None:
        samples = [("RAPL_Package0_PKG", 20.0), ("RAPL_Package1_PKG", 30.0)]
        self.assertAlmostEqual(select_package_watts(samples), 50.0)

    def test_a_negative_reading_is_treated_as_unavailable(self) -> None:
        self.assertIsNone(select_package_watts([("RAPL_Package0_PKG", -1.0)]))


class PowerDisplayTests(unittest.TestCase):
    """The tile must be steady enough to read from booth distance.

    The counter updates faster than once a second and wobbles by a fraction of a
    watt; painting that straight onto a label is the same mistake that made the
    placement bar jitter. The tile shows the mean of the last second, rounded to
    whole watts, published once a second.
    """

    def _display(self) -> PowerDisplay:
        return PowerDisplay(publish_interval_s=1.0)

    def test_first_reading_is_published_immediately(self) -> None:
        display = self._display()
        display.update(20.4, now=0.0)
        self.assertEqual(display.value, 20)

    def test_readings_inside_the_interval_do_not_change_the_display(self) -> None:
        display = self._display()
        display.update(20.0, now=0.0)
        for step in range(1, 5):
            display.update(40.0, now=step * 0.2)
        self.assertEqual(display.value, 20)

    def test_the_published_value_is_the_mean_of_the_window(self) -> None:
        display = self._display()
        display.update(20.0, now=0.0)
        # Five readings over the next interval: 20, 40, 40, 40, 40 -> mean 36.
        for step, value in enumerate((20.0, 40.0, 40.0, 40.0, 40.0), start=1):
            display.update(value, now=step * 0.2)
        self.assertEqual(display.value, 36)

    def test_a_single_spike_cannot_take_the_tile_on_its_own(self) -> None:
        # One 100 W sample among four 20 W samples averages to 36 W, not 100.
        display = self._display()
        display.update(20.0, now=0.0)
        for step, value in enumerate((100.0, 20.0, 20.0, 20.0), start=1):
            display.update(value, now=step * 0.2)
        display.update(20.0, now=1.01)
        self.assertLess(display.value, 60)

    def test_an_unavailable_reading_clears_the_value(self) -> None:
        display = self._display()
        display.update(20.0, now=0.0)
        display.update(None, now=2.0)
        self.assertIsNone(display.value)

    def test_a_transient_dropout_does_not_blank_the_tile_forever(self) -> None:
        display = self._display()
        display.update(20.0, now=0.0)
        display.update(None, now=1.5)
        display.update(21.0, now=2.0)
        self.assertEqual(display.value, 21)

    def test_a_zero_reading_is_measurement_not_absence(self) -> None:
        # Zero is a legitimate value; only None means "no counter".
        display = self._display()
        display.update(0.0, now=0.0)
        self.assertEqual(display.value, 0)

    def test_negative_readings_are_treated_as_unavailable(self) -> None:
        display = self._display()
        display.update(-5.0, now=0.0)
        self.assertIsNone(display.value)

    def test_rounding_is_half_up_not_bankers(self) -> None:
        display = self._display()
        display.update(20.5, now=0.0)
        self.assertEqual(display.value, 21)

    def test_publish_interval_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            PowerDisplay(publish_interval_s=0.0)


if __name__ == "__main__":
    unittest.main()

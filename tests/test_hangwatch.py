"""Tests for the hang watchdog.

The watchdog exists because a Windows Application Hang left no evidence: the process was killed
without a traceback. These tests pin the two behaviours that matter - a stall produces a stack
dump in the log, and a healthy heartbeat produces nothing.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from app import hangwatch


class HangWatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.log_path = Path(self._directory.name) / "session.log"
        self.stream = self.log_path.open("w+", encoding="utf-8")
        self.addCleanup(self.stream.close)
        self.addCleanup(hangwatch.disarm)

    def _log_text(self) -> str:
        self.stream.flush()
        return self.log_path.read_text(encoding="utf-8")

    def test_stalled_ui_thread_dumps_all_stacks(self) -> None:
        hangwatch.arm(self.stream, stall_seconds=0.05, poll_seconds=0.02)
        hangwatch.mark_ui_tick()

        # Never tick again: the watchdog must notice and write the stacks itself.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and "Thread 0x" not in self._log_text():
            time.sleep(0.05)

        text = self._log_text()
        # faulthandler prints thread ids and file/line, not thread names.
        self.assertIn("Thread 0x", text)
        self.assertIn("hangwatch.py", text)
        self.assertIn("_watch", text)

    def test_healthy_heartbeat_writes_nothing(self) -> None:
        hangwatch.arm(self.stream, stall_seconds=0.30, poll_seconds=0.02)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            hangwatch.mark_ui_tick()
            time.sleep(0.02)

        self.assertEqual(self._log_text(), "")

    def test_arm_without_a_usable_stream_is_disabled(self) -> None:
        self.assertIsNone(hangwatch.arm(None))

    def test_stall_is_measured_from_the_last_tick(self) -> None:
        hangwatch.mark_ui_tick()
        # Sleep well past the assertion threshold. Windows timer resolution means
        # time.sleep(0.05) can return after ~0.047s, so asserting >= 0.05 against a
        # 0.05s sleep is a coin flip that failed 5 runs in 6.
        time.sleep(0.15)

        self.assertGreaterEqual(hangwatch.ui_stall_seconds(), 0.05)


if __name__ == "__main__":
    unittest.main()

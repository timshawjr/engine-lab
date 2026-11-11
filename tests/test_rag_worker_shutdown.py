from __future__ import annotations

import unittest

from app.rag.worker import RagWorker


class _FakeThread:
    """Stands in for QThread so the shutdown paths are testable without a GPU."""

    def __init__(self, stops: bool) -> None:
        self.stops = stops
        self.quit_called = False
        self.waited_ms: int | None = None

    def quit(self) -> None:
        self.quit_called = True

    def wait(self, ms: int) -> bool:
        self.waited_ms = ms
        return self.stops


def _worker_with(thread: _FakeThread) -> RagWorker:
    worker = RagWorker.__new__(RagWorker)
    worker._shutdown = False
    worker._thread = thread
    return worker


class ShutdownSafetyTests(unittest.TestCase):
    """A QThread destroyed while still running aborts the process with
    STATUS_STACK_BUFFER_OVERRUN -- observed earlier in this project as exit code
    -1073740791. shutdown() used to drop its reference to the thread even when
    wait() timed out, which is exactly that condition.

    Generation measures ~2.5 s with the video running, so a 5 s wait is normally
    enough; a slower machine, or a request that starts just as the window closes,
    makes the timeout reachable.
    """

    def test_thread_is_stopped_then_released(self) -> None:
        thread = _FakeThread(stops=True)
        worker = _worker_with(thread)
        worker.shutdown()
        self.assertTrue(thread.quit_called)
        self.assertIsNone(worker._thread)

    def test_thread_reference_is_kept_when_it_does_not_stop(self) -> None:
        # The critical assertion: dropping the reference here would let Python
        # garbage-collect a running QThread and abort the process.
        thread = _FakeThread(stops=False)
        worker = _worker_with(thread)
        with self.assertLogs("app.rag.worker", level="WARNING"):
            worker.shutdown()
        self.assertIs(
            worker._thread,
            thread,
            "a thread that did not stop must not be released",
        )

    def test_it_warns_rather_than_claiming_a_clean_stop(self) -> None:
        thread = _FakeThread(stops=False)
        worker = _worker_with(thread)
        with self.assertLogs("app.rag.worker", level="WARNING") as captured:
            worker.shutdown()
        joined = " ".join(captured.output).lower()
        self.assertIn("did not stop", joined)

    def test_shutdown_is_idempotent_and_keeps_retrying(self) -> None:
        # If the first wait timed out, a second call must be able to stop it
        # rather than returning early because _shutdown is already True.
        thread = _FakeThread(stops=False)
        worker = _worker_with(thread)
        with self.assertLogs("app.rag.worker", level="WARNING"):
            worker.shutdown()
        thread.stops = True
        worker.shutdown()
        self.assertIsNone(worker._thread)

    def test_the_wait_is_bounded(self) -> None:
        thread = _FakeThread(stops=True)
        _worker_with(thread).shutdown()
        self.assertIsNotNone(thread.waited_ms)
        self.assertLessEqual(thread.waited_ms, 10_000)

    def test_a_worker_with_no_thread_shuts_down_quietly(self) -> None:
        worker = RagWorker.__new__(RagWorker)
        worker._shutdown = False
        worker._thread = None
        worker.shutdown()
        self.assertIsNone(worker._thread)


if __name__ == "__main__":
    unittest.main()
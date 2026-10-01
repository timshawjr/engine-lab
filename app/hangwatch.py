"""Capture the app's own thread stacks when the UI thread stops being serviced.

A Windows "Application Hang" kills the process without a Python traceback, so a reproducible
federal hang left no evidence at all - only an Application Hang event in the Windows event
log. This module makes the app record where it was stuck *before* Windows closes it:

* ``faulthandler`` is enabled against the session log, so a hard crash (for example a segfault
  inside a graphics driver) also leaves a stack.
* A daemon watchdog thread watches a heartbeat stamped by the UI thread. When the event loop
  stops being serviced for longer than ``stall_seconds``, every thread's stack is dumped into the
  session log and the stall is logged. A stall is reported once; it is reported again only after
  the UI recovers and stalls anew, so a healthy 10-minute run writes nothing.
"""

from __future__ import annotations

import faulthandler
import logging
import threading
import time
from typing import IO, Any, cast

LOGGER = logging.getLogger("engine_lab")

DEFAULT_STALL_SECONDS = 30.0
DEFAULT_POLL_SECONDS = 5.0

_lock = threading.Lock()
_stop = threading.Event()
_last_tick = time.monotonic()
_thread: threading.Thread | None = None
# The session-log stream, retained so the C-level timer can be re-armed from the UI
# heartbeat without threading the handle through every caller.
_stream: IO[str] | None = None


def mark_ui_tick() -> None:
    """Stamp the heartbeat. Called from the UI thread while the event loop is being serviced."""

    global _last_tick
    with _lock:
        _last_tick = time.monotonic()


def ui_stall_seconds() -> float:
    """Seconds since the UI thread last stamped the heartbeat."""

    with _lock:
        return time.monotonic() - _last_tick


def disarm() -> None:
    """Stop the watchdog. Used on shutdown and by tests."""

    _stop.set()


def arm(
    stream: IO[str] | None,
    stall_seconds: float = DEFAULT_STALL_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> threading.Thread | None:
    """Enable crash reporting and start the stall watchdog.

    ``stream`` must be the open session-log file object (anything with a ``fileno``). Returns the
    watchdog thread, or ``None`` when the stream cannot be used for faulthandler output.
    """

    global _thread, _stream
    if stream is None or not hasattr(stream, "fileno"):
        LOGGER.warning("hangwatch: session log stream is not usable; stall dumps are disabled")
        return None

    _stream = stream
    target = cast(Any, stream)
    try:
        faulthandler.enable(file=target, all_threads=True)
    except (ValueError, OSError) as exc:  # pragma: no cover - depends on the stream type
        LOGGER.warning("hangwatch: faulthandler could not be enabled: %s", exc)

    mark_ui_tick()
    _stop.clear()
    if _thread is not None and _thread.is_alive():
        return _thread

    def _watch() -> None:
        reported = False
        while not _stop.wait(poll_seconds):
            stalled = ui_stall_seconds()
            if stalled < stall_seconds:
                reported = False
                continue
            if reported:
                continue
            reported = True
            LOGGER.error(
                "UI thread stalled for %.1fs - dumping all thread stacks to the session log",
                stalled,
            )
            try:
                faulthandler.dump_traceback(file=target, all_threads=True)
            except (ValueError, OSError) as exc:  # pragma: no cover
                LOGGER.error("hangwatch: stack dump failed: %s", exc)

    _thread = threading.Thread(target=_watch, name="engine-lab-hangwatch", daemon=True)
    _thread.start()
    return _thread


def arm_dump_later(stall_seconds: float = DEFAULT_STALL_SECONDS) -> None:
    """Arm a C-level timer that dumps every thread's stack if the UI stops ticking.

    The watchdog thread above cannot observe the hang that actually happens. When the
    main thread blocks inside a C call that holds the GIL - an OpenVINO or driver call,
    for example - no other Python thread is scheduled, so the watchdog never reaches its
    own poll and never dumps. Six hangs across two sessions produced zero stacks for
    exactly this reason.

    ``faulthandler.dump_traceback_later`` is implemented in C and does not depend on
    another Python thread being scheduled, so it still fires. Calling it again cancels
    and restarts the timer, so re-arming it from the UI heartbeat gives "dump if the
    heartbeat has been silent for ``stall_seconds``" with no log spam on a healthy run.
    """

    if _stream is None:
        return
    try:
        faulthandler.dump_traceback_later(
            stall_seconds, file=cast(Any, _stream), repeat=False
        )
    except (ValueError, OSError) as exc:  # pragma: no cover - depends on the stream type
        LOGGER.warning("hangwatch: dump_traceback_later could not be armed: %s", exc)

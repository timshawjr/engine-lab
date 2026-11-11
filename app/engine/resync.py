"""Decide whether an armed resync should move a worker's capture.

The display clock and each stream worker decode the same clip through SEPARATE
captures. The overlay is drawn from the worker's detections over the clock's
current frame, so the two must be at the same file position or the boxes describe
a different moment from the picture.

Two failures were measured at a booth and in the harness:

1. **The two captures end up in different laps.** The clips loop, so the clock can
   wrap to the start while the worker is still mid-file. The worker is then AHEAD
   and the drift is negative. The rule "only ever seek forward" declined every
   one of those, so the misalignment persisted:

       [sync] ref=92  pos=464 sought=False reason=drift -372 below minimum
       [sync] ref=187 pos=482 sought=False reason=drift -295 below minimum

2. **A declined request was consumed anyway.** The flag was cleared before the
   code knew whether it could act, so a single unreadable position threw the
   correction away and nothing tried again until the next device change.

Seeking a worker backwards does not replay anything a visitor can see: the
worker's frames are never displayed, only its detections are, drawn over the
clock's frame. That is why a backward seek is the right correction and not a
visible rewind.

The two thresholds differ on purpose. A forward seek is invisible and cheap. A
backward seek resets the tracker and can re-emit events for frames already seen,
so it must be a bigger error (a lap-scale desync, or over a second) before it is
worth it; run-to-run jitter of a frame or two is left alone.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResyncDecision:
    """The outcome of one armed resync check.

    ``seek_to`` is the capture position the worker should jump to, or None to
    leave it where it is. ``retry`` means the decision could not be made at all
    and the request must stay armed for the next frame.
    """

    seek_to: int | None
    drift: int
    reason: str
    retry: bool = False


def decide_resync(
    *,
    reference: int,
    position: int,
    minimum_forward: int,
    minimum_backward: int,
) -> ResyncDecision:
    """Compare the clock's file position with the worker's and decide.

    Both positions are file positions in the same clip, not the clock's
    cumulative counter: the counter climbs past the file length while a capture
    wraps at it, and syncing to it once seeked past the end every frame.
    """
    if minimum_forward < 0 or minimum_backward < 0:
        raise ValueError("resync thresholds must not be negative")

    if reference < 0 or position < 0:
        # The backend reports -1 for a frame or two after a seek or reopen.
        # Postpone rather than lose the correction.
        return ResyncDecision(
            seek_to=None,
            drift=0,
            reason="position unavailable",
            retry=True,
        )

    drift = reference - position

    if drift > minimum_forward:
        return ResyncDecision(reference, drift, f"worker behind by {drift} frames")

    if drift < -minimum_backward:
        return ResyncDecision(reference, drift, f"worker ahead by {-drift} frames")

    return ResyncDecision(None, drift, f"within tolerance ({drift} frames)")

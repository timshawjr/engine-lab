"""The in-flight policy transition, as one record instead of three maps.

The HUD used to track a device/scenario change across three dicts keyed by the
same sequence number -- ``pending_acks``, ``policy_requested_at`` and
``policy_requested_monotonic`` -- plus a free-standing counter. Three failure
modes followed from that shape:

* **no timeout** -- a worker that never acknowledged left the sequence pending
  forever, so the status line never confirmed the change;
* **acks discarded on a scenario switch** -- ``_on_placement_ready`` ignored any
  acknowledgement whose ``scenario_id`` no longer matched, so the previous
  sequence's entry was never cleared and leaked;
* **partial writes** -- a record could exist in one map and not the others, and
  the completion handler then read ``None`` for the request time.

This module keeps one :class:`PendingTransition` per sequence, and makes the
three outcomes explicit: completed, timed out, or discarded because the operator
moved on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field


@dataclass
class PendingTransition:
    """One requested policy or scenario change, awaiting per-stream acks."""

    sequence: int
    scenario: str
    policy_label: str
    requested_at: float          # wall clock, for the operator log
    requested_monotonic: float   # for elapsed-time maths, immune to clock changes
    awaiting: set[int] = field(default_factory=set)

    def elapsed_ms(self, now_monotonic: float) -> float:
        return (now_monotonic - self.requested_monotonic) * 1000.0


class TransitionLedger:
    """Tracks the transitions that have been requested but not yet confirmed.

    Exactly one transition is normally in flight. Older ones are not deleted on
    sight, though: an operator pressing `N` twice quickly produces two, and the
    second must not erase the first's record before it is acknowledged.
    """

    LOGGER = logging.getLogger("app.engine.transitions")

    def __init__(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        self._timeout_seconds = timeout_seconds
        self._pending: dict[int, PendingTransition] = {}

    def __len__(self) -> int:
        return len(self._pending)

    def __contains__(self, sequence: int) -> bool:
        return sequence in self._pending

    def begin(
        self,
        *,
        sequence: int,
        scenario: str,
        policy_label: str,
        streams: set[int],
        requested_at: float,
        requested_monotonic: float,
    ) -> PendingTransition:
        transition = PendingTransition(
            sequence=sequence,
            scenario=scenario,
            policy_label=policy_label,
            requested_at=requested_at,
            requested_monotonic=requested_monotonic,
            awaiting=set(streams),
        )
        if sequence in self._pending:
            # Two different counters currently feed this key -- the policy
            # sequence and the scenario runtime sequence -- so a collision is
            # possible if an operator changes both quickly. Replacing silently
            # would strand the earlier transition without a record, so say so.
            self.LOGGER.warning(
                "Transition %d is still pending and is being replaced; the earlier "
                "one (%s) will never be recorded",
                sequence,
                self._pending[sequence].scenario,
            )
        self._pending[sequence] = transition
        return transition

    def acknowledge(self, sequence: int, stream_index: int) -> PendingTransition | None:
        """Record one stream's acknowledgement.

        Returns the transition when this call completed it (every stream acked),
        otherwise ``None``. Acknowledging an unknown sequence is ignored rather
        than raising: late acks from a superseded transition are expected.
        """
        transition = self._pending.get(sequence)
        if transition is None:
            return None
        transition.awaiting.discard(stream_index)
        if transition.awaiting:
            return None
        del self._pending[sequence]
        return transition

    def discard_stale(self, keep_sequence: int) -> list[PendingTransition]:
        """Drop transitions older than ``keep_sequence``.

        Called when the operator changes scenario. Their acknowledgements can no
        longer arrive -- the worker filters on scenario id -- so keeping the entry
        would leak it and leave the strip waiting for an ack that is never coming.
        """
        stale = [t for seq, t in self._pending.items() if seq < keep_sequence]
        for transition in stale:
            self._pending.pop(transition.sequence, None)
        return stale

    def expire(self, now_monotonic: float) -> list[PendingTransition]:
        """Return and drop transitions that have waited longer than the timeout.

        A transition that times out is a real operational fact, not a silent
        success: the caller is expected to say so rather than leave the strip
        claiming a change is still in progress.
        """
        timed_out = [
            t
            for t in self._pending.values()
            if (now_monotonic - t.requested_monotonic) > self._timeout_seconds
        ]
        for transition in timed_out:
            self._pending.pop(transition.sequence, None)
        return timed_out

    def snapshot(self) -> list[PendingTransition]:
        return sorted(self._pending.values(), key=lambda t: t.sequence)


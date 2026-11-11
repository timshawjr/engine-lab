from __future__ import annotations

import unittest

from app.engine.transitions import PendingTransition, TransitionLedger


def ledger(timeout: float = 5.0) -> TransitionLedger:
    return TransitionLedger(timeout_seconds=timeout)


def begin(l: TransitionLedger, sequence: int = 1, scenario: str = "retail",
          streams: set[int] | None = None, at: float = 0.0) -> PendingTransition:
    return l.begin(
        sequence=sequence,
        scenario=scenario,
        policy_label="SPREAD",
        streams={0, 1} if streams is None else streams,
        requested_at=1_700_000_000.0 + at,
        requested_monotonic=at,
    )


class TransitionLedgerTests(unittest.TestCase):
    """These replace three parallel dicts and a free-standing counter.

    Each test below corresponds to a failure mode that shape allowed:
    no timeout, acks discarded across a scenario switch, and partial writes.
    """

    def test_a_transition_completes_only_when_every_stream_acks(self) -> None:
        l = ledger()
        begin(l, streams={0, 1})
        self.assertIsNone(l.acknowledge(1, 0))
        completed = l.acknowledge(1, 1)
        self.assertIsNotNone(completed)
        self.assertEqual(completed.sequence, 1)
        self.assertEqual(len(l), 0)

    def test_completion_carries_the_request_time_it_started_with(self) -> None:
        # The old code read requested_at from a separate dict and could get None.
        l = ledger()
        begin(l, at=12.5)
        completed = l.acknowledge(1, 0) or l.acknowledge(1, 1)
        self.assertEqual(completed.requested_monotonic, 12.5)
        self.assertIsNotNone(completed.requested_at)

    def test_elapsed_is_measured_from_monotonic_not_wall_clock(self) -> None:
        l = ledger()
        begin(l, at=100.0)
        self.assertAlmostEqual(l.snapshot()[0].elapsed_ms(102.5), 2500.0, places=3)

    def test_an_unacknowledged_transition_times_out(self) -> None:
        l = ledger(timeout=5.0)
        begin(l, at=0.0)
        self.assertEqual(l.expire(4.0), [])
        expired = l.expire(6.0)
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].sequence, 1)
        self.assertEqual(len(l), 0, "an expired transition must not linger")

    def test_a_late_ack_after_timeout_is_ignored_not_raised(self) -> None:
        l = ledger(timeout=5.0)
        begin(l, at=0.0)
        l.expire(6.0)
        self.assertIsNone(l.acknowledge(1, 0))
        self.assertIsNone(l.acknowledge(1, 1))

    def test_a_scenario_switch_discards_older_transitions(self) -> None:
        # Acks for a superseded scenario are filtered out by the worker, so the
        # entry can never complete and must not be kept.
        l = ledger()
        begin(l, sequence=1, scenario="retail")
        begin(l, sequence=2, scenario="metro")
        l.discard_stale(keep_sequence=2)
        self.assertNotIn(1, l)
        self.assertIn(2, l)

    def test_discard_stale_returns_what_it_dropped(self) -> None:
        l = ledger()
        begin(l, sequence=1)
        begin(l, sequence=2)
        dropped = l.discard_stale(keep_sequence=2)
        self.assertEqual([t.sequence for t in dropped], [1])

    def test_two_quick_presses_are_tracked_separately(self) -> None:
        # An operator pressing N twice must not have the second erase the first
        # before it is acknowledged.
        l = ledger()
        begin(l, sequence=1)
        begin(l, sequence=2)
        self.assertEqual(len(l), 2)
        self.assertIsNotNone(l.acknowledge(1, 0) or l.acknowledge(1, 1))
        self.assertEqual(len(l), 1)
        self.assertIn(2, l)

    def test_acknowledging_an_unknown_sequence_is_a_no_op(self) -> None:
        l = ledger()
        self.assertIsNone(l.acknowledge(99, 0))
        self.assertEqual(len(l), 0)

    def test_a_duplicate_ack_does_not_complete_early(self) -> None:
        l = ledger()
        begin(l, streams={0, 1})
        l.acknowledge(0, 0)
        self.assertIsNone(l.acknowledge(0, 0), "the same stream twice must not finish it")
        self.assertEqual(len(l), 1)

    def test_a_transition_with_no_streams_completes_immediately(self) -> None:
        l = ledger()
        begin(l, streams=set())
        # No streams to wait for: the caller should treat it as already done.
        self.assertEqual(l.snapshot()[0].awaiting, set())

    def test_timeout_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            TransitionLedger(timeout_seconds=0)

    def test_snapshot_is_ordered_by_sequence(self) -> None:
        l = ledger()
        begin(l, sequence=3)
        begin(l, sequence=1)
        begin(l, sequence=2)
        self.assertEqual([t.sequence for t in l.snapshot()], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
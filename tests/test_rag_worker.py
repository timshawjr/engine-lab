from __future__ import annotations

import logging
import unittest

import numpy as np

from app.rag.retrieval import Passage
from app.rag.worker import RagAnswer, RagBackends, RagWorker


DIM = 1024


class FakeEmbedding:
    """Stands in for the GenAI embedding pipeline.

    Returns a full 1024-dimension vector because retrieval rejects any other
    width — that check is deliberate and this fake honours it.
    """

    def __init__(self, calls: list[str] | None = None) -> None:
        self.calls = calls if calls is not None else []

    def embed_query(self, text: str) -> np.ndarray:
        self.calls.append(text)
        vector = np.zeros(DIM, dtype=np.float32)
        vector[0] = 1.0
        vector[1] = 1.0
        return vector


class FakeLlm:
    """Stands in for the GenAI generation pipeline."""

    def __init__(self, answer: str = "An answer.", calls: list[str] | None = None) -> None:
        self.answer = answer
        self.calls = calls if calls is not None else []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.answer


def _identity_index(rows: int) -> np.ndarray:
    """A ``(rows, DIM)`` unit-length-row matrix for `rows` fake passages.

    ``retrieve`` requires the index row count to equal the passage count, so a
    square identity matrix of DIM rows would be rejected.
    """
    matrix = np.zeros((rows, DIM), dtype=np.float32)
    for i in range(rows):
        matrix[i, i] = 1.0
    return matrix


def fake_backends(answer: str = "An answer.") -> RagBackends:
    return RagBackends(
        embedder=FakeEmbedding(),
        llm=FakeLlm(answer),
        passages=[Passage(38, "alpha", 0.0), Passage(131, "beta", 0.0)],
        index=_identity_index(2),
        query_instruction="Instruct: ",
        system_prompt="Answer only from the passages.",
        abstention_marker="NOT IN PROVIDED PAGES",
        top_k=2,
        max_new_tokens=64,
    )


class RagWorkerTests(unittest.TestCase):
    """The worker runs on its own thread because a measured answer takes 1.3 s
    mean (max 2.0 s); blocking the UI thread for that would freeze the booth UI.

    Backends are injected and execution is synchronous in tests, so these
    assertions are deterministic. The threading itself is exercised in Task 6,
    where the page is driven end to end.
    """

    def _worker(self, backends: RagBackends | None = None) -> RagWorker:
        return RagWorker(backends or fake_backends(), synchronous=True)

    def test_empty_question_is_rejected_without_inference(self) -> None:
        backends = fake_backends()
        worker = RagWorker(backends, synchronous=True)
        answers: list[RagAnswer] = []
        worker.answered.connect(answers.append)
        worker.ask("   ")
        self.assertEqual(backends.embedder.calls, [])
        self.assertEqual(backends.llm.calls, [])
        self.assertEqual(answers, [])

    def test_second_request_while_busy_is_dropped(self) -> None:
        backends = fake_backends()
        worker = RagWorker(backends, synchronous=True)

        # Simulate an in-flight request, then ask again the way a booth visitor
        # double-tapping Ask would.
        worker._busy = True
        answers: list[RagAnswer] = []
        worker.answered.connect(answers.append)
        worker.ask("What is an SIS?")
        worker._busy = False

        self.assertEqual(backends.embedder.calls, [])
        self.assertEqual(answers, [])

    def test_backend_exception_emits_failed_not_raises(self) -> None:
        class Exploding:
            def embed_query(self, text: str) -> np.ndarray:
                raise RuntimeError("boom")

        backends = fake_backends()
        backends = RagBackends(
            embedder=Exploding(),
            llm=FakeLlm(),
            passages=[Passage(1, "a", 0.0)],
            index=_identity_index(1),
            query_instruction="Instruct: ",
            system_prompt="s",
            abstention_marker="NOT IN PROVIDED PAGES",
            top_k=1,
            max_new_tokens=64,
        )
        worker = RagWorker(backends, synchronous=True)
        failures: list[str] = []
        worker.failed.connect(failures.append)

        with self.assertLogs("app.rag.worker", level=logging.ERROR):
            worker.ask("What is an SIS?")

        self.assertEqual(len(failures), 1)
        self.assertIn("boom", failures[0])

    def test_answer_carries_question_passages_and_timings(self) -> None:
        backends = fake_backends()
        worker = RagWorker(backends, synchronous=True)
        answers: list[RagAnswer] = []
        worker.answered.connect(answers.append)

        worker.ask("What is a safety instrumented system?")

        self.assertEqual(len(answers), 1)
        result = answers[0]
        self.assertEqual(result.question, "What is a safety instrumented system?")
        self.assertEqual(result.answer, "An answer.")
        self.assertEqual(len(result.passages), 2)
        self.assertEqual([p.page for p in result.passages], [38, 131])
        self.assertGreaterEqual(result.embed_ms, 0.0)
        self.assertGreaterEqual(result.generate_ms, 0.0)

    def test_abstention_is_flagged_on_the_answer(self) -> None:
        backends = fake_backends(answer="NOT IN PROVIDED PAGES")
        worker = RagWorker(backends, synchronous=True)
        answers: list[RagAnswer] = []
        worker.answered.connect(answers.append)
        worker.ask("unanswerable?")
        self.assertTrue(answers[0].abstained)

    def test_normal_answer_is_not_flagged_as_abstention(self) -> None:
        worker = RagWorker(fake_backends(), synchronous=True)
        answers: list[RagAnswer] = []
        worker.answered.connect(answers.append)
        worker.ask("q")
        self.assertFalse(answers[0].abstained)

    def test_busy_is_cleared_after_a_successful_answer(self) -> None:
        worker = RagWorker(fake_backends(), synchronous=True)
        states: list[bool] = []
        worker.busy_changed.connect(states.append)
        worker.ask("q")
        self.assertFalse(worker._busy)
        self.assertEqual(states, [True, False])

    def test_busy_is_cleared_after_a_failure(self) -> None:
        class Exploding:
            def embed_query(self, text: str) -> np.ndarray:
                raise RuntimeError("boom")

        backends = fake_backends()
        backends.embedder = Exploding()
        worker = RagWorker(backends, synchronous=True)
        with self.assertLogs("app.rag.worker", level=logging.ERROR):
            worker.ask("q")
        self.assertFalse(worker._busy)

    def test_worker_can_answer_again_after_completing(self) -> None:
        backends = fake_backends()
        worker = RagWorker(backends, synchronous=True)
        answers: list[RagAnswer] = []
        worker.answered.connect(answers.append)
        worker.ask("first")
        worker.ask("second")
        self.assertEqual([a.question for a in answers], ["first", "second"])

    def test_shutdown_is_idempotent(self) -> None:
        worker = RagWorker(fake_backends(), synchronous=True)
        worker.shutdown()
        worker.shutdown()

    def test_ask_after_shutdown_is_ignored(self) -> None:
        backends = fake_backends()
        worker = RagWorker(backends, synchronous=True)
        worker.shutdown()
        answers: list[RagAnswer] = []
        worker.answered.connect(answers.append)
        worker.ask("q")
        self.assertEqual(answers, [])

    def test_answer_is_immutable(self) -> None:
        answer = RagAnswer("q", "a", [], False, 1.0, 2.0)
        with self.assertRaises(Exception):
            answer.answer = "changed"  # type: ignore[misc]
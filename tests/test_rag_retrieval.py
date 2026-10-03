from __future__ import annotations

import unittest

import numpy as np
import openvino_genai as genai

from app.rag.retrieval import (
    Passage,
    build_embedding_config,
    normalise,
    retrieve,
)

INSTRUCTION = (
    "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "
)


class PoolingConfigTests(unittest.TestCase):
    """These pin the single most dangerous setting in the feature.

    openvino_genai defaults TextEmbeddingPipeline to PoolingType.CLS. On a causal
    decoder such as Qwen3-Embedding that fails silently and raises nothing:
    measured on the demo machine, three of four DIFFERENT questions produced
    byte-identical vectors (query-query cosine 1.0000) and retrieval returned the
    title page for every one of them. LAST_TOKEN moved the score spread from
    0.0025 to 0.1145, a 45x improvement.

    If a future edit drops these two settings, retrieval breaks with no error and
    these tests are the only thing that notices.
    """

    def test_pooling_is_last_token_not_the_genai_default(self) -> None:
        cfg = build_embedding_config(INSTRUCTION)
        self.assertEqual(
            cfg.pooling_type,
            genai.TextEmbeddingPipeline.PoolingType.LAST_TOKEN,
        )

    def test_genai_default_would_be_cls(self) -> None:
        """Documents why the assertion above is not vacuous."""
        self.assertEqual(
            genai.TextEmbeddingPipeline.Config().pooling_type,
            genai.TextEmbeddingPipeline.PoolingType.CLS,
        )

    def test_left_padding_is_required(self) -> None:
        self.assertEqual(build_embedding_config(INSTRUCTION).padding_side, "left")

    def test_query_instruction_is_passed_through(self) -> None:
        self.assertEqual(
            build_embedding_config(INSTRUCTION).query_instruction, INSTRUCTION
        )


class NormaliseTests(unittest.TestCase):
    def test_unit_vector(self) -> None:
        vector = normalise(np.array([3.0, 4.0], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_zero_vector_is_rejected(self) -> None:
        # Measured: the NPU returns all-zero vectors under left padding. Left
        # alone, a zero vector yields uniform scores that look like a working
        # search while returning nonsense.
        with self.assertRaises(ValueError):
            normalise(np.zeros(8, dtype=np.float32))


class RetrieveTests(unittest.TestCase):
    def _passages(self, count: int) -> list[Passage]:
        return [Passage(page=i + 1, text=f"chunk {i}", score=0.0) for i in range(count)]

    def test_query_matrix_against_index(self) -> None:
        index = np.eye(3, dtype=np.float32)
        top = retrieve(
            index,
            [Passage(1, "a", 0.0), Passage(2, "b", 0.0), Passage(3, "c", 0.0)],
            np.array([0, 1, 0], dtype=np.float32),
            1,
        )
        self.assertEqual(top[0].page, 2)

    def test_identical_vectors_rank_together(self) -> None:
        index = np.eye(4, dtype=np.float32)
        query = np.array([0, 1, 0, 1], dtype=np.float32)
        first = retrieve(index, self._passages(4), query, 2)
        second = retrieve(index, self._passages(4), query, 2)
        self.assertEqual([p.page for p in first], [p.page for p in second])

    def test_results_are_sorted_by_descending_score(self) -> None:
        index = np.array(
            [[0.1, 0.0], [0.9, 0.0], [0.5, 0.0]], dtype=np.float32
        )
        top = retrieve(index, self._passages(3), np.array([1.0, 0.0], dtype=np.float32), 3)
        scores = [p.score for p in top]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_scores_are_the_raw_cosine(self) -> None:
        index = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        top = retrieve(
            index,
            [Passage(1, "a", 0.0), Passage(2, "b", 0.0)],
            np.array([1.0, 0.0], dtype=np.float32),
            1,
        )
        self.assertAlmostEqual(top[0].score, 1.0, places=6)

    def test_top_k_larger_than_corpus_returns_everything(self) -> None:
        index = np.eye(2, dtype=np.float32)
        top = retrieve(index, self._passages(2), np.array([1.0, 1.0], dtype=np.float32), 5)
        self.assertEqual(len(top), 2)

    def test_passage_is_immutable(self) -> None:
        # The worker hands these to a Qt signal; a caller mutating one in place
        # would silently corrupt a displayed answer.
        passage = Passage(1, "text", 0.5)
        with self.assertRaises(Exception):
            passage.page = 2  # type: ignore[misc]
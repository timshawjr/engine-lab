from __future__ import annotations

import unittest

from app.rag.generation import build_generation_config, is_abstention
from app.rag.retrieval import Passage
from app.rag import generation as gen
from app.rag.config import FALLBACK, load_rag_config


class LatencyBudgetTests(unittest.TestCase):
    """The pause a booth visitor sees is the answer budget, not the machinery.

    Measured on the demo machine for a typical question
    ("How does this document describe network segmentation between IT and OT?"):

        220 tokens -> 3237 ms
         96 tokens -> 1861 ms
         48 tokens -> 1010 ms
         24 tokens ->  570 ms   (already a complete answer)

    A 220-token budget buys a tail nobody reads, and the wait reads as a hang.
    """

    def test_answer_budget_is_short_enough_to_avoid_a_visible_pause(self) -> None:
        budget = FALLBACK["max_new_tokens"]
        self.assertLessEqual(
            budget, 64,
            f"max_new_tokens={budget} produces a multi-second pause at the booth",
        )

    def test_repo_config_matches_the_latency_budget(self) -> None:
        config = load_rag_config("config/rag.json")
        self.assertLessEqual(int(config["max_new_tokens"]), 64)

    def test_generation_config_honours_the_budget(self) -> None:
        self.assertEqual(build_generation_config(64).max_new_tokens, 64)


class WorkingStateTests(unittest.TestCase):
    """A wait must look like progress, not a freeze."""

    def test_progress_text_names_the_retrieval_step(self) -> None:
        self.assertIn("passage", gen.RETRIEVING_STATUS.lower())

    def test_progress_text_names_the_generation_step(self) -> None:
        self.assertIn("answer", gen.GENERATING_STATUS.lower())

    def test_progress_steps_are_distinct(self) -> None:
        self.assertNotEqual(gen.RETRIEVING_STATUS, gen.GENERATING_STATUS)


class AbstentionStillWorksTests(unittest.TestCase):
    """Shortening the answer must not weaken the refusal behaviour.

    Measured with the long budget: the explicit system prompt abstained on 2 of 3
    unsupported questions and wrongly abstained on 0 of 3 supported ones. The
    marker is what makes a refusal legible, so it must survive truncation.
    """

    def test_marker_still_detected(self) -> None:
        self.assertTrue(is_abstention("NOT IN PROVIDED PAGES", "NOT IN PROVIDED PAGES"))

    def test_truncated_refusal_still_detected(self) -> None:
        # A tight budget can cut the refusal mid-phrase.
        self.assertTrue(is_abstention("NOT IN PROVIDED", "NOT IN PROVIDED PAGES"))

    def test_real_answer_still_not_a_refusal(self) -> None:
        self.assertFalse(
            is_abstention("It separates the OT network.", "NOT IN PROVIDED PAGES")
        )


class PassageEvidenceTests(unittest.TestCase):
    """The evidence panel is the part that carries trust when the answer is terse.

    With a shorter answer the passages matter more, not less, so the excerpt must
    keep showing whole passages: at 700 characters only 78 of 595 chunks fitted,
    and a measured case hid the sentence that supported a correct answer.
    """

    def test_passage_excerpt_shows_whole_passages(self) -> None:
        from app.theme import THEME

        self.assertGreaterEqual(THEME.rag_passage_excerpt_chars, 5000)

    def test_passages_are_always_rendered(self) -> None:
        source = (
            Path := __import__("pathlib").Path
        )("app/hud.py").read_text(encoding="utf-8")
        self.assertIn("self.rag_sources.setText", source)


if __name__ == "__main__":
    unittest.main()
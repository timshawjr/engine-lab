from __future__ import annotations

import unittest

import openvino_genai as genai

from app.rag.generation import (
    build_generation_config,
    build_prompt,
    is_abstention,
)
from app.rag.retrieval import Passage

SYSTEM = (
    "You answer questions about NIST SP 800-82r4 using ONLY the numbered passages "
    "provided. Cite the page as [p.N]. If the passages do not contain the answer, "
    "reply with exactly NOT IN PROVIDED PAGES and nothing else."
)
MARKER = "NOT IN PROVIDED PAGES"


class PromptTests(unittest.TestCase):
    def test_prompt_disables_reasoning_mode(self) -> None:
        # Qwen3 is a reasoning model: with the chat template applied it spends the
        # whole token budget inside <think> and returns NO answer at all.
        prompt = build_prompt(SYSTEM, [Passage(38, "text", 0.7)], "What is an SIS?")
        self.assertIn("<think>\n\n</think>", prompt)

    def test_prompt_starts_the_assistant_turn_directly(self) -> None:
        # The prompt ends with an EMPTY think block: reasoning opens and closes
        # immediately, so the budget is spent on the answer.
        prompt = build_prompt(SYSTEM, [Passage(38, "text", 0.7)], "What is an SIS?")
        self.assertTrue(
            prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            f"prompt ended with {prompt[-60:]!r}",
        )
        self.assertNotIn("<think>\n\n</think>\n\nassistant", prompt)

    def test_every_passage_is_labelled_with_its_page(self) -> None:
        prompt = build_prompt(
            SYSTEM,
            [Passage(38, "alpha", 0.1), Passage(131, "beta", 0.2)],
            "q",
        )
        self.assertIn("[p.38]", prompt)
        self.assertIn("[p.131]", prompt)

    def test_passage_text_is_not_truncated(self) -> None:
        # Measured: truncating each chunk to 110 words cut off the very sentence
        # being asked about, and both models then answered NOT IN PROVIDED PAGES
        # for questions whose answers had been retrieved correctly.
        long_text = "word " * 400
        prompt = build_prompt(SYSTEM, [Passage(1, long_text.strip(), 0.5)], "q")
        self.assertIn(long_text.strip(), prompt)

    def test_whitespace_in_passage_is_collapsed(self) -> None:
        prompt = build_prompt(SYSTEM, [Passage(1, "a\n\n  b\tc", 0.5)], "q")
        self.assertIn("a b c", prompt)

    def test_question_is_included(self) -> None:
        prompt = build_prompt(SYSTEM, [Passage(1, "text", 0.5)], "What is an SIS?")
        self.assertIn("What is an SIS?", prompt)

    def test_no_passages_still_builds_a_prompt(self) -> None:
        # Retrieval returning nothing must not raise here; the page shows the
        # abstention marker rather than crashing.
        prompt = build_prompt(SYSTEM, [], "q")
        self.assertIn("q", prompt)


class AbstentionTests(unittest.TestCase):
    def test_marker_is_detected(self) -> None:
        self.assertTrue(is_abstention(MARKER, MARKER))

    def test_marker_with_trailing_period_is_detected(self) -> None:
        self.assertTrue(is_abstention("NOT IN PROVIDED PAGES.", MARKER))

    def test_detection_is_case_insensitive(self) -> None:
        self.assertTrue(is_abstention("not in provided pages", MARKER))

    def test_real_answer_is_not_an_abstention(self) -> None:
        self.assertFalse(
            is_abstention("It separates the OT network from the enterprise.", MARKER)
        )

    def test_empty_answer_is_not_an_abstention(self) -> None:
        # An empty answer is a failure, not a refusal; reporting it as an
        # abstention would show "not in pages" for a question that was answered
        # with nothing at all.
        self.assertFalse(is_abstention("", MARKER))


class GenerationConfigTests(unittest.TestCase):
    def test_decoding_is_greedy(self) -> None:
        # A booth visitor asking the same question twice must get the same
        # answer, so sampling must be off.
        cfg = build_generation_config(220)
        self.assertFalse(cfg.do_sample)

    def test_chat_template_is_not_applied(self) -> None:
        # With the template applied, GenAI lets Qwen3 enter reasoning mode and
        # burns the budget without answering.
        self.assertFalse(build_generation_config(220).apply_chat_template)

    def test_max_new_tokens_is_honoured(self) -> None:
        self.assertEqual(build_generation_config(220).max_new_tokens, 220)

    def test_config_is_a_genai_generation_config(self) -> None:
        self.assertIsInstance(build_generation_config(220), genai.GenerationConfig)
"""Cleaning invariants for the RAG corpus extractor.

These tests run under the pinned ``.venv``, which deliberately has no
``pypdf`` and no ``openvino_genai``. They must therefore exercise only the
standard-library cleaning helpers, never the bake itself (a bake is ~28 s on
the GPU and is a one-time build step, not a unit test).
"""

from __future__ import annotations

import unittest

from tools.build_rag_corpus import clean_page, marginal_line_numbers


class CleaningTests(unittest.TestCase):
    def test_running_header_is_removed(self) -> None:
        page = "NIST SP 800-82r4 ipd (Initial Public Draft)  Guide to OT Security\nSeptember 2026\n\n113\nBody text here."
        self.assertNotIn("NIST SP", clean_page(page))
        self.assertNotIn("113", clean_page(page))
        self.assertIn("Body text here.", clean_page(page))

    def test_every_marginal_run_is_detected(self) -> None:
        # Regression: only the LONGEST run used to be returned, which left
        # earlier line numbers embedded mid-sentence. Two runs, each three or
        # more long, so the correct answer is their union -- a longest-only
        # implementation returns just {4066..4069} and misses {4074..4076}.
        lines = ["Alpha 4066", "Beta 4067", "Gamma 4068", "Delta 4069",
                 "Epsilon", "Zeta 4074", "Eta 4075", "Theta 4076"]
        self.assertEqual(marginal_line_numbers(lines),
                         {4066, 4067, 4068, 4069, 4074, 4075, 4076})

    def test_justified_text_is_not_mangled(self) -> None:
        # Layout-mode extraction produced "n et work bu t n ot allow ed".
        page = "4070 permitted on the enterprise network but not allowed on OT networks."
        self.assertIn("enterprise network but not allowed on OT networks",
                      clean_page("May also be " + page))
        # Also pin real cleaning behaviour on the same input: the running
        # header and the bare folio are removed, so an identity clean_page
        # would fail this test.
        cleaned = clean_page(
            "NIST SP 800-82r4 ipd (Initial Public Draft)  Guide to OT Security\n"
            "September 2026\n\n4070\n"
            "May also be permitted on the enterprise network but not allowed on OT networks."
        )
        self.assertNotIn("NIST SP", cleaned)
        self.assertNotIn("September 2026", cleaned)
        self.assertNotIn("4070", cleaned)
        self.assertIn("May also be permitted", cleaned)

    def test_common_word_fragment_keeps_its_space(self) -> None:
        # "his" is a complete word, so it must NOT be glued to the next
        # line. Without the stop-list this produced "hisdocument provides".
        cleaned = clean_page("his\ndocument provides guidelines")
        self.assertIn("his document provides guidelines", cleaned)
        self.assertNotIn("hisdocument", cleaned)

    def test_genuine_fragment_still_glues(self) -> None:
        # "mec" is a mid-word fragment, so it must still glue to "hanisms".
        cleaned = clean_page("mec\nhanisms")
        self.assertIn("mechanisms", cleaned)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from app.rag.availability import RagAvailability, RagState, describe_failure


class AvailabilityStateTests(unittest.TestCase):
    """`rag_worker is None` used to mean four different things, and the page
    printed the raw exception into the status line -- so a customer could read
    `FileNotFoundError: ...openvino_model.bin` on the booth screen.
    """

    def test_ready_can_answer(self) -> None:
        availability = RagAvailability.ready()
        self.assertIs(availability.state, RagState.READY)
        self.assertTrue(availability.can_answer)

    def test_unavailable_cannot_answer(self) -> None:
        availability = RagAvailability.unavailable("the model files are missing.")
        self.assertIs(availability.state, RagState.UNAVAILABLE)
        self.assertFalse(availability.can_answer)

    def test_ready_status_does_not_claim_a_problem(self) -> None:
        self.assertNotIn("unavailable", RagAvailability.ready().status_line().lower())

    def test_unavailable_status_carries_the_reason(self) -> None:
        line = RagAvailability.unavailable("the model files are missing.").status_line()
        self.assertIn("unavailable", line.lower())
        self.assertIn("model files are missing", line)

    def test_the_status_line_is_one_sentence_a_visitor_can_read(self) -> None:
        line = RagAvailability.unavailable("the configuration could not be read.").status_line()
        self.assertNotIn("\n", line)
        self.assertLessEqual(len(line), 110)


class FailureDescriptionTests(unittest.TestCase):
    """Exception text must go to the log, not the screen.

    These messages are chosen by exception type rather than by echoing `str(exc)`,
    because `str(exc)` frequently contains a filesystem path.
    """

    def test_missing_model_file_is_described_without_the_path(self) -> None:
        exc = FileNotFoundError("C:\\secret\\path\\openvino_model.bin")
        described = describe_failure(exc)
        self.assertIn("missing", described)
        self.assertNotIn("secret", described)
        self.assertNotIn("C:", described)

    def test_permission_failure_is_described(self) -> None:
        self.assertIn("could not be opened", describe_failure(PermissionError("x")))

    def test_bad_config_is_described(self) -> None:
        import json

        try:
            json.loads("{not json")
        except json.JSONDecodeError as exc:
            self.assertIn("configuration", describe_failure(exc))

    def test_an_unknown_failure_still_reads_as_a_sentence(self) -> None:
        described = describe_failure(RuntimeError("vpux compiler exploded"))
        self.assertNotIn("vpux", described)
        self.assertTrue(described.endswith("."))

    def test_no_description_leaks_a_path_or_traceback(self) -> None:
        for exc in (
            FileNotFoundError("/a/b/c"),
            PermissionError("/a/b/c"),
            RuntimeError("/a/b/c"),
            MemoryError(),
        ):
            described = describe_failure(exc)
            self.assertNotIn("/a/", described)
            self.assertNotIn("Error", described)


if __name__ == "__main__":
    unittest.main()
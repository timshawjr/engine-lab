"""Regression tests for benchmark reporting honesty.

A full acceptance run produced one FAIL row whose Evidence column pointed at a JSON file left
behind by an earlier, unrelated run: the killed run wrote nothing of its own. A report that cites
evidence it does not have is worse than one with an empty cell, so both halves of that are pinned
here - the row must not cite a file the run did not produce, and a timeout must carry the output
tail so the failure is diagnosable at all.
"""

from __future__ import annotations

import subprocess
import unittest

from tools.benchmark_matrix import LOGS, RunResult, RunSpec, _markdown, _output_tail


def _result(name: str, passed: bool, detail: str) -> RunResult:
    return RunResult(RunSpec(name=name, scenario="retail", density=1, seconds=60.0), passed, detail, {})


class EvidenceCellTests(unittest.TestCase):
    def test_failed_run_without_diagnostics_does_not_cite_a_file(self) -> None:
        diagnostics = LOGS / "phase3-benchmark-retail-d1.json"
        diagnostics.unlink(missing_ok=True)
        self.addCleanup(diagnostics.unlink, True)

        row = [
            line
            for line in _markdown([_result("retail-d1", False, "timeout after 180s")]).splitlines()
            if line.startswith("| retail-d1 ")
        ]

        self.assertEqual(len(row), 1)
        self.assertIn("no diagnostics written", row[0])
        self.assertNotIn("phase3-benchmark-retail-d1.json", row[0])

    def test_run_that_wrote_diagnostics_cites_it(self) -> None:
        diagnostics = LOGS / "phase3-benchmark-smart_city-d1.json"
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.write_text("{}", encoding="utf-8")
        self.addCleanup(diagnostics.unlink, True)

        row = [
            line
            for line in _markdown([_result("smart_city-d1", True, "60.0s")]).splitlines()
            if line.startswith("| smart_city-d1 ")
        ]

        self.assertIn("`logs/phase3-benchmark-smart_city-d1.json`", row[0])


class TimeoutTailTests(unittest.TestCase):
    def test_carries_the_child_output(self) -> None:
        exc = subprocess.TimeoutExpired(cmd=["python"], timeout=1.0, output=b"partial output\n")

        self.assertEqual(_output_tail(exc), "partial output")

    def test_empty_when_the_child_said_nothing(self) -> None:
        exc = subprocess.TimeoutExpired(cmd=["python"], timeout=1.0)

        self.assertEqual(_output_tail(exc), "")

    def test_keeps_the_tail_not_the_head(self) -> None:
        exc = subprocess.TimeoutExpired(cmd=["python"], timeout=1.0, output="x" * 5000 + "END")

        tail = _output_tail(exc)

        self.assertTrue(tail.endswith("END"))
        self.assertLessEqual(len(tail), 1000)


if __name__ == "__main__":
    unittest.main()

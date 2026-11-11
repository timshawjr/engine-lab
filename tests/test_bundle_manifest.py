from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bundle_manifest import (
    CHECKSUM_NAME,
    MANIFEST_NAME,
    build,
    collect,
    verify,
)


class BundleManifestTests(unittest.TestCase):
    """A 2.6 GB bundle is handed to other people as a zip. Without a manifest they
    cannot tell a complete copy from a truncated one, and a half-extracted folder
    fails far from the cause.

    The manifest must also never imply the artefact is trusted: there is no
    Authenticode certificate, so it records ``signed: false`` rather than leaving
    a reader to assume otherwise.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.bundle = Path(self._tmp.name) / "engine-lab-portable"
        self.bundle.mkdir()
        (self.bundle / "Start Engine Lab.bat").write_text("@echo off\n", encoding="utf-8")
        nested = self.bundle / "app"
        nested.mkdir()
        (nested / "main.py").write_text("print(1)\n", encoding="utf-8")
        (self.bundle / "models.bin").write_bytes(b"\x00" * 1024)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_build_records_every_payload_file(self) -> None:
        manifest = build(self.bundle, "abc1234")
        self.assertEqual(manifest["file_count"], 3)
        self.assertEqual(
            sorted(manifest["files"]),
            ["Start Engine Lab.bat", "app/main.py", "models.bin"],
        )

    def test_manifest_declares_itself_unsigned(self) -> None:
        # Nothing may fabricate authenticity.
        manifest = build(self.bundle, None)
        self.assertIs(manifest["signed"], False)
        self.assertIn("Unsigned", manifest["signature_note"])
        self.assertIn("SmartScreen", manifest["signature_note"])

    def test_build_writes_both_manifest_forms(self) -> None:
        build(self.bundle, "abc1234")
        self.assertTrue((self.bundle / MANIFEST_NAME).is_file())
        # A conventional sha256sum file lets a recipient verify with standard
        # tools rather than only with this script.
        self.assertTrue((self.bundle / CHECKSUM_NAME).is_file())
        first = (self.bundle / CHECKSUM_NAME).read_text(encoding="utf-8").splitlines()[0]
        self.assertRegex(first, r"^[0-9a-f]{64}  ")

    def test_manifest_files_are_not_themselves_hashed(self) -> None:
        build(self.bundle, None)
        rows = collect(self.bundle)
        names = {rel for rel, _, _ in rows}
        self.assertNotIn(MANIFEST_NAME, names)
        self.assertNotIn(CHECKSUM_NAME, names)

    def test_output_is_deterministic_for_identical_content(self) -> None:
        first = build(self.bundle, "abc1234")["files"]
        second = build(self.bundle, "abc1234")["files"]
        self.assertEqual(first, second)

    def test_verify_passes_on_an_untouched_bundle(self) -> None:
        build(self.bundle, None)
        self.assertEqual(verify(self.bundle), 0)

    def test_verify_detects_a_changed_file(self) -> None:
        build(self.bundle, None)
        (self.bundle / "app" / "main.py").write_text("print(2)\n", encoding="utf-8")
        self.assertEqual(verify(self.bundle), 1)

    def test_verify_detects_a_missing_file(self) -> None:
        build(self.bundle, None)
        (self.bundle / "models.bin").unlink()
        self.assertEqual(verify(self.bundle), 1)

    def test_verify_detects_an_extra_file(self) -> None:
        build(self.bundle, None)
        (self.bundle / "smuggled.txt").write_text("x", encoding="utf-8")
        self.assertEqual(verify(self.bundle), 1)

    def test_verify_fails_loudly_without_a_manifest(self) -> None:
        self.assertEqual(verify(self.bundle), 2)

    def test_a_truncated_file_is_caught_by_size_before_hashing(self) -> None:
        # The realistic failure: an interrupted copy.
        build(self.bundle, None)
        data = (self.bundle / "models.bin").read_bytes()
        (self.bundle / "models.bin").write_bytes(data[:512])
        self.assertEqual(verify(self.bundle), 1)


class GeneratedBytecodeTests(unittest.TestCase):
    """The bundle carries a Python interpreter. The first run of that
    interpreter writes ``__pycache__`` trees -- including a run of the manifest
    verifier itself. Hashing them made ``verify`` report dozens of EXTRA files on
    every bundle a recipient had opened, which reads as "your copy is corrupt".
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.bundle = Path(self._tmp.name) / "engine-lab-portable"
        (self.bundle / "app").mkdir(parents=True)
        (self.bundle / "app" / "main.py").write_text("print(1)\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_bytecode_is_not_recorded_in_the_manifest(self) -> None:
        cache = self.bundle / "app" / "__pycache__"
        cache.mkdir()
        (cache / "main.cpython-312.pyc").write_bytes(b"\x00" * 32)

        manifest = build(self.bundle, None)

        self.assertEqual(sorted(manifest["files"]), ["app/main.py"])

    def test_bytecode_written_after_a_build_does_not_fail_verify(self) -> None:
        build(self.bundle, None)
        # Exactly what the first launch does.
        for directory in (self.bundle / "app" / "__pycache__", self.bundle / "__pycache__"):
            directory.mkdir(exist_ok=True)
            (directory / "x.cpython-312.pyc").write_bytes(b"\x00" * 16)
        self.assertEqual(verify(self.bundle), 0)

    def test_a_stray_pyc_outside_a_cache_directory_is_ignored(self) -> None:
        build(self.bundle, None)
        (self.bundle / "loose.pyc").write_bytes(b"\x00" * 8)
        self.assertEqual(verify(self.bundle), 0)

    def test_a_real_extra_file_is_still_caught(self) -> None:
        # Excluding bytecode must not blunt the truncation/copy check.
        build(self.bundle, None)
        cache = self.bundle / "__pycache__"
        cache.mkdir()
        (cache / "x.cpython-312.pyc").write_bytes(b"\x00" * 16)
        (self.bundle / "unexpected.dll").write_bytes(b"\x00" * 16)
        self.assertEqual(verify(self.bundle), 1)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the Windows telemetry helpers that can be exercised anywhere.

Both defects these cover were silent: one produced a truncated PDH instance list, the other a
one-character device name. Neither raised, which is exactly why they only show up as a blank or
wrong gauge on the demo machine.
"""

from __future__ import annotations

import ctypes
import unittest

from app.telemetry.devices_win import _decode_utf16_registry_string
from app.telemetry.pdh import _split_double_nul

INSTANCES = (
    "pid_1234_luid_0x00000000_0x000128A6_phys_0_eng_0_engtype_3D",
    "pid_5678_luid_0x00000000_0x000124A5_phys_0_eng_0_engtype_Compute",
    "pid_9012_luid_0x00000000_0x000124A5_phys_0_eng_0_engtype_Neural",
)


class SplitDoubleNulTests(unittest.TestCase):
    """``PdhEnumObjectItemsW`` returns a double-NUL-terminated list of TCHAR strings."""

    def test_parses_every_instance_name(self) -> None:
        payload = "\x00".join(INSTANCES) + "\x00\x00"
        # Build the buffer as explicit UTF-16LE bytes: that is the layout the wide PDH API fills
        # on Windows. ``create_unicode_buffer`` cannot be used here because ``c_wchar`` is 4 bytes
        # on this host (2 on Windows), which would silently change what the helper is reading.
        buffer = ctypes.create_string_buffer(payload.encode("utf-16-le"))

        self.assertEqual(_split_double_nul(buffer, len(payload)), list(INSTANCES))

    def test_tchar_count_is_not_a_byte_count(self) -> None:
        # Documents the defect being guarded: reading ``char_count`` *bytes* from a wide buffer
        # parses only half the list, so instances after the first one disappear.
        payload = "\x00".join(INSTANCES) + "\x00\x00"
        buffer = ctypes.create_string_buffer(payload.encode("utf-16-le"))
        truncated = ctypes.string_at(buffer, len(payload)).decode("utf-16-le", errors="replace")

        self.assertNotEqual([value for value in truncated.split("\x00") if value], list(INSTANCES))

    def test_empty_buffer(self) -> None:
        buffer = ctypes.create_string_buffer("\x00\x00".encode("utf-16-le"))
        self.assertEqual(_split_double_nul(buffer, 0), [])


class RegistryStringTests(unittest.TestCase):
    """``SetupDiGetDeviceRegistryPropertyW`` fills a UTF-16LE buffer."""

    def test_decodes_the_whole_name(self) -> None:
        text = "Intel(R) AI Boost"
        raw = (text + "\x00").encode("utf-16-le")
        self.assertEqual(_decode_utf16_registry_string(raw, len(raw)), text)

    def test_bytes_value_would_truncate_to_one_character(self) -> None:
        # Documents the defect being guarded: bytes.value stops at the first NUL *byte*, which in
        # UTF-16LE is half of the first character, and a one-character name fails is_npu().
        text = "Intel(R) AI Boost"
        raw = (text + "\x00").encode("utf-16-le")
        self.assertEqual(ctypes.create_string_buffer(raw).value, b"I")
        self.assertEqual(_decode_utf16_registry_string(raw, len(raw)), text)

    def test_zero_size_is_empty(self) -> None:
        self.assertEqual(_decode_utf16_registry_string(b"", 0), "")


if __name__ == "__main__":
    unittest.main()

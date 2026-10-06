from __future__ import annotations

import unittest

from PySide6.QtCore import Qt

from app.hud import rag_key_closes_panel


class RagPanelCloseTests(unittest.TestCase):
    """The question box takes focus when the panel opens, so keypresses go to the
    QLineEdit and never reach MainWindow.keyPressEvent. Without a rule of its own
    the panel could be opened with R but never closed -- the reported bug.

    Escape is the close key. No printable key may close the panel: an earlier
    attempt let R close while the box was empty, and that made "Risk assessment"
    impossible to type because the leading "r" dismissed the panel.
    """

    def test_escape_closes_an_empty_box(self) -> None:
        self.assertTrue(
            rag_key_closes_panel(Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier, "")
        )

    def test_escape_closes_mid_question(self) -> None:
        # The dependable escape hatch: works with text typed.
        self.assertTrue(
            rag_key_closes_panel(
                Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier, "half-typed question"
            )
        )

    def test_r_never_closes_so_questions_can_start_with_r(self) -> None:
        # "Risk assessment..." and "Retail..." must be typeable from an empty box.
        self.assertFalse(
            rag_key_closes_panel(Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier, "")
        )
        self.assertFalse(
            rag_key_closes_panel(
                Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier, "Risk"
            )
        )

    def test_no_printable_key_closes(self) -> None:
        for key in (
            Qt.Key.Key_R,
            Qt.Key.Key_A,
            Qt.Key.Key_N,
            Qt.Key.Key_Q,
            Qt.Key.Key_Space,
            Qt.Key.Key_Return,
        ):
            self.assertFalse(
                rag_key_closes_panel(key, Qt.KeyboardModifier.NoModifier, ""),
                f"{key} must remain typeable in the question box",
            )

    def test_shift_modified_keys_never_close(self) -> None:
        self.assertFalse(
            rag_key_closes_panel(
                Qt.Key.Key_R, Qt.KeyboardModifier.ShiftModifier, ""
            )
        )


class CloseAffordanceTests(unittest.TestCase):
    """A booth operator must be able to dismiss the panel without a keyboard
    shortcut, since the shortcut is not discoverable on its own."""

    def test_close_button_exists_and_is_wired(self) -> None:
        from pathlib import Path

        source = Path("app/hud.py").read_text(encoding="utf-8")
        self.assertIn("rag_close_button", source)
        self.assertIn("rag_close_button.clicked.connect(self._close_rag_panel)", source)

    def test_escape_is_caught_while_the_box_has_focus(self) -> None:
        from pathlib import Path

        source = Path("app/hud.py").read_text(encoding="utf-8")
        self.assertIn("rag_input.installEventFilter(self)", source)
        self.assertIn("def eventFilter(", source)

    def test_closing_returns_focus_to_the_window(self) -> None:
        # Otherwise focus stays on a hidden text box and the scenario keys
        # (1-6), N/G/C and R stop responding.
        from pathlib import Path

        source = Path("app/hud.py").read_text(encoding="utf-8")
        marker = "def _close_rag_panel"
        body = source[source.index(marker) : source.index(marker) + 700]
        self.assertIn("setFocus()", body)

    def test_the_panel_says_how_to_close_it(self) -> None:
        from pathlib import Path

        source = Path("app/hud.py").read_text(encoding="utf-8")
        self.assertIn("Esc", source)


if __name__ == "__main__":
    unittest.main()
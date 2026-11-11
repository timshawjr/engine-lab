"""What the document Q&A page is able to do, stated in one place.

``rag_worker is None`` used to mean any of four different things: no config path
supplied, config unreadable, pipelines failed to load, and not yet initialised.
The page then printed the raw exception into the status line, so a customer could
read ``FileNotFoundError: ...openvino_model.bin`` on the booth screen.

This module separates the two questions a caller actually asks:

* **can it answer?** -- one value of :class:`RagState`
* **why not?** -- a short human sentence, with the exception kept in the log
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RagState(str, Enum):
    """Whether the page can answer a question right now."""

    #: Pipelines loaded; a question can be asked.
    READY = "ready"
    #: Nothing is wrong with the app, but this page cannot be used. Carries a
    #: reason suitable for showing a visitor.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RagAvailability:
    """The page's capability, plus a reason when it is not usable.

    ``reason`` is always a short sentence fit for the status line. Exception text
    belongs in the log, not on a booth screen.
    """

    state: RagState
    reason: str = ""

    @property
    def can_answer(self) -> bool:
        return self.state is RagState.READY

    @classmethod
    def ready(cls) -> "RagAvailability":
        return cls(RagState.READY, "")

    @classmethod
    def unavailable(cls, reason: str) -> "RagAvailability":
        return cls(RagState.UNAVAILABLE, reason)

    def status_line(self) -> str:
        """What the page shows. Never contains a traceback or a file path."""
        if self.can_answer:
            return "Ask a question. Esc or Close puts this panel away."
        return f"Document Q&A unavailable — {self.reason}"


def describe_failure(exc: BaseException) -> str:
    """Turn an exception into a sentence a visitor can read.

    Deliberately drops the exception message: it is often a filesystem path or a
    driver string, which leaks implementation detail and reads as a crash.
    """
    name = type(exc).__name__
    if name in {"FileNotFoundError", "IsADirectoryError"}:
        return "the model files are missing from this installation."
    if name in {"PermissionError"}:
        return "the model files could not be opened."
    if name in {"JSONDecodeError", "ValueError", "KeyError"}:
        return "the configuration could not be read."
    if name in {"MemoryError"}:
        return "there was not enough memory to load the models."
    return "the assistant could not start on this machine."

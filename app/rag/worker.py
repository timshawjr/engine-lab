"""Runs one retrieval-plus-generation request off the UI thread.

A measured answer takes 1.3 s mean and up to 2.0 s on this machine, so it must
not run on the Qt UI thread — that would freeze the booth, including the hang
watch that exists to catch exactly this (AGENTS.md §11).

The worker owns nothing but the two GenAI pipelines. It never touches a widget:
it emits Qt signals, and the page connects to them (AGENTS.md §11).

Two guards, both measured or observed rather than theoretical:

* A second request arriving while one is in flight is DROPPED, not queued. A
  booth visitor double-tapping Ask would otherwise get two answers racing into
  the same label.
* Any backend exception is caught, logged with a traceback, and surfaced as
  ``failed`` (AGENTS.md §12). A missing model file must not take the app down in
  front of a customer.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal

from .generation import build_generation_config, build_prompt, generate, is_abstention
from .retrieval import Passage, embed_query, retrieve

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagAnswer:
    """One completed answer, with the evidence that produced it."""

    question: str
    answer: str
    passages: list[Passage]
    abstained: bool
    embed_ms: float
    generate_ms: float


@dataclass
class RagBackends:
    """Everything the worker needs, so tests can inject fakes with no GPU."""

    embedder: object
    llm: object
    passages: list[Passage]
    index: np.ndarray
    query_instruction: str
    system_prompt: str
    abstention_marker: str
    top_k: int
    max_new_tokens: int


class RagWorker(QObject):
    """Executes one question at a time and reports via signals."""

    answered = Signal(object)
    failed = Signal(str)
    busy_changed = Signal(bool)

    # Internal hand-off from the UI thread to the worker thread. Declared as a
    # signal rather than a direct call because a plain method call would run on
    # the caller's thread and block the UI for the measured 1.3 s.
    _request = Signal(str)

    def __init__(
        self,
        backends: RagBackends,
        *,
        synchronous: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._backends = backends
        self._busy = False
        self._shutdown = False
        self._synchronous = synchronous
        self._thread: QThread | None = None
        if not synchronous:
            self._thread = QThread()
            self._thread.setObjectName("rag-worker")
            self.moveToThread(self._thread)
            # A queued connection: emitted on the UI thread, delivered in this
            # object's own thread, so the inference never runs on the UI thread.
            self._request.connect(self._run_request, Qt.ConnectionType.QueuedConnection)
            self._thread.start()

    def is_busy(self) -> bool:
        """Whether a request is in flight. Safe to read from the UI thread."""
        return self._busy

    def ask(self, question: str) -> None:
        """Queue one question. Returns immediately.

        Empty or whitespace-only questions are dropped before any inference, so
        a stray Return key costs nothing rather than the measured 1.3 s.
        """
        if self._shutdown:
            return
        if not (question or "").strip():
            return
        if self._busy:
            LOGGER.info("Dropping request while another is in flight")
            return

        self._busy = True
        self.busy_changed.emit(True)
        if self._synchronous:
            self._run_request(question)
            return
        self._start_threaded(question)

    def _start_threaded(self, question: str) -> None:
        # Queued across the thread boundary established in __init__.
        self._request.emit(question)

    def _run_request(self, question: str) -> None:
        """Do the work. Runs on the worker thread in production."""
        try:
            backends = self._backends

            started = time.perf_counter()
            query_vec = embed_query(
                backends.embedder, question, backends.query_instruction
            )
            passages = retrieve(
                backends.index, backends.passages, query_vec, backends.top_k
            )
            embed_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            prompt = build_prompt(backends.system_prompt, passages, question)
            answer = generate(backends.llm, prompt)
            generate_ms = (time.perf_counter() - started) * 1000.0

            self.answered.emit(
                RagAnswer(
                    question=question,
                    answer=answer,
                    passages=passages,
                    abstained=is_abstention(answer, backends.abstention_marker),
                    embed_ms=embed_ms,
                    generate_ms=generate_ms,
                )
            )
        except Exception:  # noqa: BLE001 - must never propagate to the booth UI
            LOGGER.error("RAG request failed: %s", traceback.format_exc())
            self.failed.emit(traceback.format_exc(limit=3))
        finally:
            self._busy = False
            self.busy_changed.emit(False)

    def shutdown(self) -> None:
        """Stop accepting work and tear the thread down. Safe to call twice.

        The thread must be stopped before this QObject is destroyed. Leaving a
        QThread running and letting Python garbage-collect it aborts the process
        with STATUS_STACK_BUFFER_OVERRUN, so the page calls this from its own
        shutdown path.
        """
        if self._shutdown:
            return
        self._shutdown = True
        thread = self._thread
        if thread is not None:
            thread.quit()
            # A request already in flight holds the pipeline for its measured
            # 1.3-2.0 s, so allow it to drain rather than cutting the model off
            # mid-generation. Only after that is it safe to discard.
            if not thread.wait(5000):
                LOGGER.warning("RAG worker thread did not stop in time")
            self._thread = None


def load_backends(config: dict, root) -> RagBackends:
    """Construct the real pipelines from ``config/rag.json``."""
    from pathlib import Path

    import openvino_genai as genai

    from .generation import build_generation_config as _build_gen_cfg
    from .retrieval import (
        build_embedding_config,
        load_corpus,
        load_index,
    )

    root = Path(root)
    device = str(config.get("device", "GPU"))

    embedder = genai.TextEmbeddingPipeline(
        str(root / config["embedding_model"]),
        device,
        build_embedding_config(config["query_instruction"]),
    )
    llm = genai.LLMPipeline(str(root / config["generation_model"]), device)
    llm.set_generation_config(
        _build_gen_cfg(int(config.get("max_new_tokens", 220)))
    )

    return RagBackends(
        embedder=embedder,
        llm=llm,
        passages=load_corpus(root / config["corpus_path"]),
        index=load_index(root / config["index_path"]),
        query_instruction=str(config["query_instruction"]),
        system_prompt=str(config["system_prompt"]),
        abstention_marker=str(config["abstention_marker"]),
        top_k=int(config.get("top_k", 3)),
        max_new_tokens=int(config.get("max_new_tokens", 220)),
    )
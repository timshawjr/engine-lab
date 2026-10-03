"""Dense retrieval over the baked NIST SP 800-82r4 corpus.

The embedding model is Qwen3-Embedding-0.6B, a causal decoder, loaded through
openvino-genai's TextEmbeddingPipeline.

POOLING IS LOAD-BEARING. openvino_genai defaults that pipeline to
``PoolingType.CLS``, which is meaningless for a causal decoder and fails
silently -- no exception, no warning. Measured on the demo machine:

  * three of four DIFFERENT questions returned byte-identical vectors
    (query-to-query cosine 1.0000);
  * retrieval returned the title page for every question;
  * the score spread between the best and third-best passage was 0.0025.

Setting ``LAST_TOKEN`` moved that spread to 0.1145, a 45x improvement, and made
each question's ranking distinct. ``padding_side="left"`` is required alongside
it so that last-token pooling reads a real token rather than padding.

``tests/test_rag_retrieval.py`` pins both settings, because nothing else would
notice their loss.

Never run this path on the NPU with left padding: measured, the NPU returns
all-zero vectors under it. That is why ``normalise`` rejects a zero vector
rather than returning one -- a zero vector produces uniform scores that look
like a working search while returning nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openvino_genai as genai

#: Fallback width when config/rag.json does not state one. config/rag.json's
#: ``embedding_dim`` is authoritative; this only keeps a hand-built config from
#: failing on a missing key.
EMBEDDING_DIM = 1024


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, with the dense score that selected it."""

    page: int
    text: str
    score: float


def build_embedding_config(query_instruction: str) -> genai.TextEmbeddingPipeline.Config:
    """Build the embedding pipeline config this model actually requires.

    ``LAST_TOKEN`` and left padding are the two settings that silently break
    retrieval when omitted; see the module docstring for the measurements.
    """
    config = genai.TextEmbeddingPipeline.Config()
    config.pooling_type = genai.TextEmbeddingPipeline.PoolingType.LAST_TOKEN
    config.padding_side = "left"
    config.query_instruction = query_instruction
    return config


def normalise(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` scaled to unit length.

    Raises:
        ValueError: if the vector has zero length. GenAI normally normalises for
            us, but a zero vector does not throw on its own and would score
            uniformly against the whole index -- indistinguishable, from the
            outside, from a search that genuinely found nothing relevant.
    """
    array = np.asarray(vector, dtype=np.float32).ravel()
    magnitude = float(np.linalg.norm(array))
    if magnitude == 0.0 or not np.isfinite(magnitude):
        raise ValueError(
            "embedding has zero or non-finite magnitude; refusing to normalise"
        )
    return array / magnitude


def _as_matrix(result: object, count: int) -> np.ndarray:
    """Coerce a GenAI embedding result to a ``(count, dim)`` float32 matrix.

    GenAI's return shape is not stable across calls and configurations:
    an object exposing ``to_vecs()``, a list of vectors, or a flat list of
    floats. Measured here: ``embed_documents`` over the whole corpus returned a
    flat list, and iterating it as if it were a list of vectors raises
    ``TypeError: 'float' object is not iterable``.
    """
    vectors = result.to_vecs() if hasattr(result, "to_vecs") else result
    if isinstance(vectors, (list, tuple)) and vectors and hasattr(vectors[0], "to_vecs"):
        collected: list[object] = []
        for item in vectors:
            collected.extend(item.to_vecs())
        vectors = collected
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(count, -1)
    return array


def embed_query(
    pipeline: object, query: str, query_instruction: str, expected_dim: int = EMBEDDING_DIM
) -> np.ndarray:
    """Embed one question and return a unit-length 1-D vector.

    ``expected_dim`` is checked rather than assumed: a wrong-width embedding
    would fail later as a confusing shape error inside the matmul instead of
    here, where the cause is obvious.
    """
    result = pipeline.embed_query(query_instruction + query)
    vector = normalise(_as_matrix(result, 1)[0])
    if vector.size != expected_dim:
        raise ValueError(
            f"expected {expected_dim}-dimension embedding, got {vector.size}"
        )
    return vector


def retrieve(
    index: np.ndarray,
    passages: list[Passage],
    query_vec: np.ndarray,
    top_k: int,
) -> list[Passage]:
    """Rank ``passages`` by cosine similarity to ``query_vec``.

    The index is assumed to hold unit-length rows, as baked by
    ``tools/build_rag_corpus.py``, so a plain dot product is the cosine.
    """
    if index.shape[0] != len(passages):
        raise ValueError(
            f"index has {index.shape[0]} rows but {len(passages)} passages were supplied"
        )
    if index.shape[0] == 0:
        return []

    scores = np.asarray(index, dtype=np.float32) @ normalise(query_vec)
    if not np.isfinite(scores).all():
        raise ValueError("retrieval produced non-finite scores")

    limit = min(max(top_k, 0), len(passages))
    # Sort by score, breaking ties on page so the ordering is deterministic --
    # a booth visitor asking the same question twice must see the same passages.
    order = sorted(range(len(passages)), key=lambda i: (-scores[i], passages[i].page))
    return [
        Passage(
            page=passages[i].page,
            text=passages[i].text,
            score=float(scores[i]),
        )
        for i in order[:limit]
    ]


def load_corpus(corpus_path) -> list[Passage]:
    """Load the baked corpus into ``Passage`` objects with zero scores."""
    import json
    from pathlib import Path

    data = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    return [Passage(page=int(c["page"]), text=str(c["text"]), score=0.0) for c in data["chunks"]]


def load_index(index_path) -> np.ndarray:
    """Load the baked embedding matrix as float32."""
    from pathlib import Path

    matrix = np.load(Path(index_path))
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D index, got shape {matrix.shape}")
    return np.asarray(matrix, dtype=np.float32)
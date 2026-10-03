#!/usr/bin/env python3
"""Bake the NIST SP 800-82r4 retrieval corpus and its embedding index.

This is a DEVELOPMENT-ONLY, one-time build step (~28 s on the GPU). It needs
``pypdf``, which is deliberately absent from the production ``requirements.txt``
and from the pinned ``.venv``; run it under the dev venv, like
``tools/build_clip_zero_shot.py``:

    <dev-venv>\\Scripts\\python tools\\build_rag_corpus.py

It writes:

``models/rag/corpus.json``
    ``{"document": ..., "chunks": [{"page": int, "text": str}, ...]}``
``models/rag/index.npy``
    float32 array of shape ``(len(chunks), 1024)``, row-aligned with the chunks.

Why the text handling looks the way it does (measured during the spike, see
the reference extractor this was ported from):

  plain pypdf   words split across lines ("mec" / "hanisms", "O" /
                "rganizations") but body text otherwise clean
  layout pypdf  justified text mangled: "n et work bu t n ot allow ed"
  pdfminer      lost line ordering entirely on the two-column pages

So: plain extraction, then repair. ``pypdf`` is imported inside ``main()``
and never at module level, because the pinned ``.venv`` has no ``pypdf`` and
the unit tests import the cleaning helpers from that venv.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = Path(r"C:\Users\Intel Demo\Downloads\NIST.SP.800-82r4.ipd.pdf")
OUT_DIR = ROOT / "models" / "rag"
CORPUS_PATH = OUT_DIR / "corpus.json"
INDEX_PATH = OUT_DIR / "index.npy"

EMBEDDING_MODEL = ROOT / "models" / "Qwen3-Embedding-0.6B-int8-ov"
DEVICE = "GPU"
BATCH_SIZE = 16  # one embed_documents call over all 595 chunks exhausts GPU memory

# Measured counts from the spike. A mismatch means the PDF or the cleaning
# changed; the bake refuses to proceed rather than shipping a silent surprise.
EXPECTED_PAGES = 321
EXPECTED_CHUNKS = 595
EXPECTED_WORDS = 107_496

HEADER = re.compile(
    r"NIST\s*SP\s*800[-\s]*82r|Guide\s*to\s*OT\s*Security|^September\s*2026\s*$",
    re.IGNORECASE,
)
ROMAN = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
TRAILING_NUM = re.compile(r"^(.*?)\s+(\d{1,5})\s*$")
FOLIO = re.compile(r"\d{1,4}")


def marginal_line_numbers(lines: list[str]) -> set[int]:
    """Trailing integers that increment by one are the marginal line numbers.

    A page's numbering is split into several runs wherever a line carries no
    marginal number (a figure caption, a wrapped heading). Keeping only the
    longest run left the earlier numbers in the text, so every run of three or
    more is collected instead.
    """
    found: list[int] = []
    for line in lines:
        m = TRAILING_NUM.match(line.rstrip())
        if m and m.group(2):
            found.append(int(m.group(2)))
    if len(found) < 3:
        return set()

    keep: set[int] = set()
    run_start, run_len = 0, 1
    for i in range(1, len(found) + 1):
        if i < len(found) and found[i] == found[i - 1] + 1:
            run_len += 1
            continue
        if run_len >= 3:
            keep.update(found[run_start:i])
        run_start, run_len = i, 1
    return keep


def clean_page(text: str) -> str:
    """Strip the running header, folios and marginal line numbers, then reflow."""
    raw = [line.rstrip() for line in text.splitlines()]
    numbers = marginal_line_numbers(raw)

    kept: list[str] = []
    for line in raw:
        stripped = line.strip()
        if not stripped:
            continue
        if HEADER.search(stripped) or ROMAN.match(stripped):
            continue
        # The folio (page number) sits alone on its line.
        if FOLIO.fullmatch(stripped):
            continue
        m = TRAILING_NUM.match(stripped)
        if m and int(m.group(2)) in numbers:
            stripped = m.group(1).strip()
            if not stripped:
                continue
        kept.append(stripped)

    # Rejoin the orphan fragments plain mode leaves when a word is split
    # ("mec" + "hanisms", "end-of-s" + "upport", "O" + "rganizations").
    out: list[str] = []
    for line in kept:
        if (
            out
            and len(out[-1]) <= 8
            and re.search(r"[A-Za-z\-]$", out[-1])
            and re.match(r"^[a-z]", line)
        ):
            out[-1] = out[-1] + line
        else:
            out.append(line)

    # Reflow: a line that does not end a sentence continues into the next.
    reflowed = ""
    for line in out:
        if (
            reflowed
            and not reflowed.endswith(("-", " "))
            and reflowed[-1] not in ".!?:;"
            and re.match(r"^[a-z(]", line)
        ):
            reflowed += " " + line
        else:
            reflowed += (" " if reflowed else "") + line
    return re.sub(r"\s{2,}", " ", reflowed).strip()


def chunk(text: str, target: int = 240, overlap: int = 40) -> list[str]:
    """Split cleaned page text into overlapping word windows.

    A window ends early at the last sentence boundary that still leaves at
    least half the target behind, so chunks do not cut mid-sentence when the
    text cooperates.
    """
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + target, len(words))
        if end < len(words):
            for stop in range(end, max(start + target // 2, start), -1):
                if words[stop - 1].endswith((".", "!", "?")):
                    end = stop
                    break
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return pieces


def _import_genai():
    """Import openvino_genai, falling back to the repo's own .venv site-packages.

    The dev venv carries pypdf/torch/transformers but not openvino_genai;
    the pinned .venv carries openvino_genai but not pypdf. Both are Python
    3.12, so the compiled extension from .venv imports cleanly under the dev
    interpreter when its site-packages is on sys.path.
    """
    try:
        import openvino_genai  # noqa: F401

        return openvino_genai
    except ImportError:
        fallback = ROOT / ".venv" / "Lib" / "site-packages"
        if fallback.is_dir():
            sys.path.insert(0, str(fallback))
            import openvino_genai  # noqa: F401

            return openvino_genai
        raise


def build_corpus() -> tuple[int, list[dict[str, object]]]:
    """Extract, clean and chunk the PDF. Returns (page count, chunks)."""
    from pypdf import PdfReader  # imported here: the pinned venv has no pypdf

    reader = PdfReader(str(SRC))
    chunks: list[dict[str, object]] = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_page(page.extract_text() or "")
        if not text:
            continue
        for piece in chunk(text):
            chunks.append({"page": number, "text": piece})
    return len(reader.pages), chunks


def embed_chunks(chunks: list[dict[str, object]]) -> "object":
    """Embed every chunk in batches of BATCH_SIZE; returns the float32 matrix."""
    import numpy as np

    genai = _import_genai()
    config = genai.TextEmbeddingPipeline.Config()
    config.pooling_type = genai.TextEmbeddingPipeline.PoolingType.LAST_TOKEN
    config.padding_side = "left"
    config.batch_size = BATCH_SIZE

    print(f"loading embedding model from {EMBEDDING_MODEL} on {DEVICE}")
    pipeline = genai.TextEmbeddingPipeline(str(EMBEDDING_MODEL), DEVICE, config)

    texts = [str(c["text"]) for c in chunks]
    rows: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        # The pipeline requires exactly batch_size texts; pad a short final
        # batch by repeating its last entry and discard the extra rows.
        if len(batch) < BATCH_SIZE:
            batch = batch + [batch[-1]] * (BATCH_SIZE - len(batch))
        rows.extend(pipeline.embed_documents(batch)[: len(texts[start:])])
        print(f"  embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    return np.asarray(rows, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="write corpus.json only; skip the GPU embedding pass",
    )
    args = parser.parse_args(argv)

    if not SRC.is_file():
        print(f"ERROR: source PDF not found: {SRC}", file=sys.stderr)
        return 1

    print(f"extracting {SRC}")
    page_count, chunks = build_corpus()
    corpus_text = " ".join(str(c["text"]) for c in chunks)
    word_count = len(corpus_text.split())

    print(f"pages            : {page_count}")
    print(f"chunks           : {len(chunks)}")
    print(f"words            : {word_count:,}")
    if chunks:
        print(
            f"mean words/chunk : "
            f"{sum(len(str(c['text']).split()) for c in chunks) / len(chunks):.0f}"
        )

    mismatches = []
    if page_count != EXPECTED_PAGES:
        mismatches.append(f"pages {page_count} != expected {EXPECTED_PAGES}")
    if len(chunks) != EXPECTED_CHUNKS:
        mismatches.append(f"chunks {len(chunks)} != expected {EXPECTED_CHUNKS}")
    if word_count != EXPECTED_WORDS:
        mismatches.append(f"words {word_count} != expected {EXPECTED_WORDS}")
    if mismatches:
        print()
        print("ERROR: measured counts disagree with the spike:", file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(
            "Refusing to bake: the PDF or the cleaning changed. Fix the "
            "extractor or update the expected counts with a measurement.",
            file=sys.stderr,
        )
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    document = {
        "title": "NIST SP 800-82r4 ipd \u2014 Guide to Operational Technology (OT) Security",
        "source_pdf": SRC.name,
        "pages": page_count,
        "note": "Initial public draft. Extracted from the PDF supplied for this demo.",
    }
    CORPUS_PATH.write_text(
        json.dumps({"document": document, "chunks": chunks}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {CORPUS_PATH}")

    if args.skip_embed:
        return 0

    import numpy as np

    matrix = embed_chunks(chunks)
    print(f"index shape      : {matrix.shape}")
    if matrix.shape != (len(chunks), 1024):
        print(
            f"ERROR: index shape {matrix.shape} != {(len(chunks), 1024)}",
            file=sys.stderr,
        )
        return 1
    np.save(INDEX_PATH, matrix)
    print(f"wrote {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

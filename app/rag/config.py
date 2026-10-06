"""Loading ``config/rag.json`` and building the real pipelines from it.

Kept separate from :mod:`app.rag.worker` so the page can read the config (to
label the engine honestly) without paying the ~7 s pipeline load, and so a
missing or malformed config fails in one obvious place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config/rag.json")

# The page shows these even if config/rag.json omits them, because they are
# measured facts about this feature rather than preferences.
FALLBACK: dict[str, Any] = {
    "top_k": 3,
    # Answer budget. Measured on the demo machine for a typical question:
    #   220 tokens -> 3237 ms,  96 -> 1861 ms,  48 -> 1010 ms,  24 -> 570 ms,
    # and the 24-token answer was already complete. The long budget bought a tail
    # nobody read while the wait read as a hang, so it is deliberately short.
    "max_new_tokens": 64,
    "device": "GPU",
    "device_label": "GPU",
    "abstention_marker": "NOT IN PROVIDED PAGES",
    "embedding_dim": 1024,
}


def load_rag_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read ``config/rag.json``, filling in measured defaults.

    Raises:
        FileNotFoundError: if the config is absent. The caller disables the
            page rather than guessing paths.
    """
    config = dict(FALLBACK)
    config.update(json.loads(Path(path).read_text(encoding="utf-8")))

    for required in ("corpus_path", "index_path", "embedding_model",
                     "generation_model", "query_instruction", "system_prompt"):
        if not config.get(required):
            raise ValueError(f"config/rag.json is missing {required!r}")
    return config


def load_rag_backends(config: dict[str, Any], root: Path | str):
    """Build the GenAI pipelines described by ``config``.

    Kept as a thin re-export so the page and the tests share one construction
    path; the actual work lives in :mod:`app.rag.worker`.
    """
    from .worker import load_backends

    return load_backends(config, root)
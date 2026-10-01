"""Zero-shot vocabularies, one per scenario.

The vocabulary is the contract: CLIP may only ever return one of these names, so
the overlay cannot invent a label the operator did not declare. Every entry is a
plain, concrete noun phrase a person standing in that scene would recognise.

This replaces an earlier ImageNet-1k stage, which produced ``Chihuahua``,
``crash_helmet`` and ``king_penguin`` on a stationary cookware item and flipped
its label 239 times in 40 seconds. ImageNet has no retail taxonomy, so no
threshold could make that output trustworthy.

One vocabulary per vertical, because the footage differs: the retail clip is an
overhead self-checkout of groceries, and the smart-city clip is a multi-lane
intersection. A single shared vocabulary named groceries with kitchenware words,
so the vocabulary became per-scenario in ``config/scenarios.json``. A stage whose
named pair (``vocabulary_<name>.json`` / ``text_embeddings_<name>.npy``) is
missing falls back to the baked default pair, which this module's default build
path still writes.
"""

from __future__ import annotations

#: scenario id -> label -> prompt templates. The score is averaged over all
#: templates, which makes CLIP less sensitive to a single awkward phrasing.
#:
#: These names are derived from what the footage actually shows, not from what
#: the detector head reports. The retail list is the checkout belt itself: a
#: hand-checked run on this clip read the loaf ``bread``, the bunch ``bananas``
#: and the tins ``jar`` at 40-49% confidence, so the declared categories follow
#: the footage rather than the detector's class names. The list is deliberately a
#: *category* list, not a SKU database. ``bananas`` is a true statement about the
#: crop; it is not a product code, and the app does not claim it is one.
VOCABULARIES: dict[str, dict[str, tuple[str, ...]]] = {
    "retail": {
        "bananas": (
            "a bunch of bananas",
            "ripe yellow bananas",
        ),
        "apple": (
            "a red apple",
            "a green apple",
        ),
        "packaged snack": (
            "a packaged snack product",
            "a wrapped snack bar",
        ),
        "bottle": (
            "a plastic bottle",
            "a glass bottle",
        ),
        "canned good": (
            "a tin can of food",
            "a jar of food",
        ),
        "carton": (
            "a beverage carton",
            "a milk carton",
        ),
        "bread": (
            "a loaf of bread",
            "a sliced loaf of bread",
        ),
        "bag of chips": (
            "a bag of potato chips",
            "a bag of corn chips",
        ),
    },
    "metro": {
        # The intersection clip is a top-down view, so each category carries one
        # overhead phrasing alongside a plain one. Measured on the vehicle crops
        # this footage actually yields, either phrasing reads the cars as car
        # at 0.25-0.40 (van/truck runner-up); the overhead phrasing is kept
        # because it is faithful to the framing and kept those reads on car
        # rather than flipping to van.
        "car": (
            "a passenger car",
            "a car seen from above on a road",
        ),
        "van": (
            "a passenger van",
            "a van seen from above",
        ),
        "truck": (
            "a delivery truck",
            "a truck seen from above",
        ),
        "bus": (
            "a city bus",
            "a bus seen from above",
        ),
        "motorcycle": (
            "a motorcycle",
            "a motorcycle seen from above",
        ),
        "bicycle": (
            "a bicycle",
            "a bicycle seen from above",
        ),
        "person": (
            "a person walking",
            "a person seen from above",
        ),
    },
}

#: Alias kept for existing imports. Retail is the scenario CLIP was first baked
#: for, so the builder's default (no ``--scenario``) path still uses this entry.
RETAIL_VOCABULARY: dict[str, tuple[str, ...]] = VOCABULARIES["retail"]


def prompt_texts(vocabulary: dict[str, tuple[str, ...]]) -> list[str]:
    """Return one prompt per template, in a stable order."""

    texts: list[str] = []
    for templates in vocabulary.values():
        texts.extend(templates)
    return texts


def vocabulary_labels(vocabulary: dict[str, tuple[str, ...]]) -> list[str]:
    """Return the vocabulary labels, in the same stable order as the prompts."""

    return list(vocabulary)

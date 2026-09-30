"""Zero-shot vocabulary for the retail scenario.

The vocabulary is the contract: CLIP may only ever return one of these names, so
the overlay cannot invent a label the operator did not declare. Every entry is a
plain, concrete noun phrase a person standing at that shelf would recognise.

This replaces an earlier ImageNet-1k stage, which produced ``Chihuahua``,
``crash_helmet`` and ``king_penguin`` on a stationary cookware item and flipped
its label 239 times in 40 seconds. ImageNet has no retail taxonomy, so no
threshold could make that output trustworthy.
"""

from __future__ import annotations

#: label -> prompt templates. The score is averaged over all templates, which
#: makes CLIP less sensitive to a single awkward phrasing.
#:
#: These names are derived from what the footage actually shows, not from what
#: the 80-class COCO head reports. On the store-aisle clip YOLO calls the hero
#: object ``bowl`` on every pass, but the crop is a stainless steel saucepan with
#: a rim and handle, and CLIP independently names it ``pot`` - 75 of 75 confident
#: calls agreed. COCO's ``bowl`` is the wrong name for it, so the declared
#: vocabulary follows the image.
#:
#: The list is deliberately a *category* list, not a SKU database. ``pot`` is a
#: true statement about the crop; it is not a product code, and the app does not
#: claim it is one.
RETAIL_VOCABULARY: dict[str, tuple[str, ...]] = {
    "pot": (
        "a stainless steel cooking pot",
        "a metal saucepan with a handle",
    ),
    "mixing bowl": (
        "a deep glass mixing bowl",
        "a large salad mixing bowl",
    ),
    "soup bowl": (
        "a ceramic soup bowl",
        "a small bowl of soup",
    ),
    "plate": (
        "a ceramic dinner plate",
        "a flat white plate",
    ),
    "cup": (
        "a ceramic drinking mug",
        "a coffee cup",
    ),
    "glass": (
        "a drinking glass",
        "a glass tumbler",
    ),
    "bottle": (
        "a tall glass bottle",
        "a plastic water bottle",
    ),
    "vase": (
        "a decorative vase",
        "a flower vase",
    ),
    "kettle": (
        "a stovetop kettle",
        "a metal teapot",
    ),
    "storage container": (
        "a plastic food storage container",
        "a lunch box with a lid",
    ),
}


def prompt_texts(vocabulary: dict[str, tuple[str, ...]]) -> list[str]:
    """Return one prompt per template, in a stable order."""

    texts: list[str] = []
    for templates in vocabulary.values():
        texts.extend(templates)
    return texts


def vocabulary_labels(vocabulary: dict[str, tuple[str, ...]]) -> list[str]:
    """Return the vocabulary labels, in the same stable order as the prompts."""

    return list(vocabulary)

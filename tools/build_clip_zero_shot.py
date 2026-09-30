#!/usr/bin/env python3
"""Build the OpenVINO CLIP zero-shot assets used by the retail scenario.

This is a DEVELOPMENT-ONLY tool. It needs ``torch`` and ``transformers``, which
are deliberately absent from the production ``requirements.txt``: the app must
never import a deep-learning framework at runtime.

It emits the vision tower once, plus one text-embedding pair per vocabulary:

``vision_model.onnx``
    The CLIP ViT-B/32 vision tower, exported at a static 1x3x224x224 input.
    The tower is shared by every scenario, so a ``--scenario`` run does not
    re-export it.
``text_embeddings.npy`` + ``vocabulary.json``
    The default pair: unit-length embeddings for ``RETAIL_VOCABULARY`` in
    ``tools/clip_vocabulary.py``, computed once, here.
``text_embeddings_<scenario>.npy`` + ``vocabulary_<scenario>.json``
    A per-scenario pair for ``VOCABULARIES[<scenario>]``, written by
    ``--scenario <scenario>``. The vocabulary is fixed configuration, so the
    text tower never has to run again on the booth machine; the app only needs
    the vision tower per crop.

Run it with a throwaway environment, not the app venv:

    python -m venv <dev-venv>
    <dev-venv>\\Scripts\\pip install torch transformers numpy pillow
    <dev-venv>\\Scripts\\python tools\\build_clip_zero_shot.py --out models/clip-vision-patch32
    <dev-venv>\\Scripts\\python tools\\build_clip_zero_shot.py --out models/clip-vision-patch32 --scenario retail
    <dev-venv>\\Scripts\\python tools\\build_clip_zero_shot.py --out models/clip-vision-patch32 --scenario smart_city

Then convert the ONNX to IR with the app venv:

    .venv\\Scripts\\python tools\\convert_clip_onnx.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.clip_vocabulary import (  # noqa: E402
    RETAIL_VOCABULARY,
    VOCABULARIES,
    prompt_texts,
    vocabulary_labels,
)

CHECKPOINT = "openai/clip-vit-base-patch32"


class VisionTower(nn.Module):
    """CLIP image encoder plus its visual projection, as one graph.

    ``get_image_features`` is ``visual_projection(vision_model(x).pooler_output)``.
    Exporting both together means the ONNX graph ends at the joint-space
    embedding, so the app never has to reimplement HF's pooling.
    """

    def __init__(self, clip: CLIPModel) -> None:
        super().__init__()
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.vision_model(pixel_values=pixel_values).pooler_output
        return self.visual_projection(pooled)


def build_text_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    vocabulary: dict[str, tuple[str, ...]],
) -> tuple[np.ndarray, list[str], list[int]]:
    labels = vocabulary_labels(vocabulary)
    texts = prompt_texts(vocabulary)
    with torch.no_grad():
        tokens = processor(text=texts, return_tensors="pt", padding=True)
        features = model.get_text_features(**tokens)
        features = torch.nn.functional.normalize(features, dim=-1)
    # Prompt index -> owning label, so the app can average templates per label.
    template_counts = [len(vocabulary[label]) for label in labels]
    return features.numpy().astype(np.float32), labels, template_counts


def write_vocabulary_index(
    path: Path,
    labels: list[str],
    template_counts: list[int],
) -> None:
    # A tiny human-readable index so the app never hardcodes vocabulary order.
    path.write_text(
        "{\n"
        f'  "checkpoint": "{CHECKPOINT}",\n'
        f'  "labels": {labels!r},\n'.replace("'", '"')
        + "".join(
            f'  "template_counts": {template_counts!r},\n'.replace("'", '"')
        )
        + '  "templates_per_label": true\n}\n',
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "clip-vision-patch32",
        help="output directory for the ONNX vision tower and text embeddings",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(VOCABULARIES),
        default=None,
        help="bake only this scenario's vocabulary pair "
        "(text_embeddings_<name>.npy and vocabulary_<name>.json) beside the "
        "default assets; the shared vision tower is not re-exported",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {CHECKPOINT}")
    model = CLIPModel.from_pretrained(CHECKPOINT).eval()
    processor = CLIPProcessor.from_pretrained(CHECKPOINT)

    if args.scenario:
        vocabulary = VOCABULARIES[args.scenario]
        embeddings_path = out_dir / f"text_embeddings_{args.scenario}.npy"
        index_path = out_dir / f"vocabulary_{args.scenario}.json"
    else:
        vocabulary = RETAIL_VOCABULARY
        embeddings_path = out_dir / "text_embeddings.npy"
        index_path = out_dir / "vocabulary.json"

    embeddings, labels, template_counts = build_text_embeddings(
        model, processor, vocabulary
    )
    np.save(embeddings_path, embeddings)
    print(
        f"wrote {embeddings_path} "
        f"shape={embeddings.shape} labels={len(labels)}"
    )
    write_vocabulary_index(index_path, labels, template_counts)
    print(f"wrote {index_path}")

    if args.scenario:
        # The vision tower is shared by every scenario, so a per-scenario run
        # bakes only the text pair.
        return 0

    tower = VisionTower(model).eval()
    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    onnx_path = out_dir / "vision_model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            tower,
            dummy,
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_embedding"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

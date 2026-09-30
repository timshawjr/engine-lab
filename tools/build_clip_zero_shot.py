#!/usr/bin/env python3
"""Build the OpenVINO CLIP zero-shot assets used by the retail scenario.

This is a DEVELOPMENT-ONLY tool. It needs ``torch`` and ``transformers``, which
are deliberately absent from the production ``requirements.txt``: the app must
never import a deep-learning framework at runtime.

It emits exactly two artifacts:

``vision_model.onnx``
    The CLIP ViT-B/32 vision tower, exported at a static 1x3x224x224 input.
``text_embeddings.npy``
    Unit-length embeddings for the declared vocabulary in
    ``tools/clip_vocabulary.py``, computed once, here. The vocabulary is fixed
    configuration, so the text tower never has to run again on the booth
    machine; the app only needs the vision tower per crop.

Run it with a throwaway environment, not the app venv:

    python -m venv <dev-venv>
    <dev-venv>\\Scripts\\pip install torch transformers numpy pillow
    <dev-venv>\\Scripts\\python tools\\build_clip_zero_shot.py --out models/clip-vision-patch32

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
) -> tuple[np.ndarray, list[str], list[int]]:
    labels = vocabulary_labels(RETAIL_VOCABULARY)
    texts = prompt_texts(RETAIL_VOCABULARY)
    with torch.no_grad():
        tokens = processor(text=texts, return_tensors="pt", padding=True)
        features = model.get_text_features(**tokens)
        features = torch.nn.functional.normalize(features, dim=-1)
    # Prompt index -> owning label, so the app can average templates per label.
    template_counts = [len(RETAIL_VOCABULARY[label]) for label in labels]
    return features.numpy().astype(np.float32), labels, template_counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "clip-vision-patch32",
        help="output directory for the ONNX vision tower and text embeddings",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {CHECKPOINT}")
    model = CLIPModel.from_pretrained(CHECKPOINT).eval()
    processor = CLIPProcessor.from_pretrained(CHECKPOINT)

    embeddings, labels, template_counts = build_text_embeddings(model, processor)
    np.save(out_dir / "text_embeddings.npy", embeddings)
    print(
        f"wrote {out_dir / 'text_embeddings.npy'} "
        f"shape={embeddings.shape} labels={len(labels)}"
    )

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

    # A tiny human-readable index so the app never hardcodes vocabulary order.
    (out_dir / "vocabulary.json").write_text(
        "{\n"
        f'  "checkpoint": "{CHECKPOINT}",\n'
        f'  "labels": {labels!r},\n'.replace("'", '"')
        + "".join(
            f'  "template_counts": {template_counts!r},\n'.replace("'", '"')
        )
        + '  "templates_per_label": true\n}\n',
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'vocabulary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

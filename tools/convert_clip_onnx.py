#!/usr/bin/env python3
"""Convert the CLIP vision ONNX tower into OpenVINO IR.

Run with the APP venv (openvino only, no torch):

    .venv\\Scripts\\python tools\\convert_clip_onnx.py

``tools/build_clip_zero_shot.py`` produces the ONNX; this step turns it into the
same ``.xml`` / ``.bin`` pair the rest of the app already loads, so CLIP is
compiled, cached, and placed exactly like every other model in the demo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import openvino as ov

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "clip-vision-patch32",
    )
    parser.add_argument("--fp16", action="store_true", help="compress weights to FP16")
    args = parser.parse_args(argv)

    model_dir: Path = args.model_dir
    onnx_path = model_dir / "vision_model.onnx"
    if not onnx_path.is_file():
        raise SystemExit(f"missing {onnx_path}; run tools/build_clip_zero_shot.py first")

    print(f"reading {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    model = ov.convert_model(str(onnx_path))

    xml_path = model_dir / "clip-vision.xml"
    bin_path = model_dir / "clip-vision.bin"
    ov.save_model(model, xml_path, compress_to_fp16=args.fp16)
    print(f"wrote {xml_path.name} ({xml_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {bin_path.name} ({bin_path.stat().st_size / 1e6:.1f} MB)")

    reloaded = ov.Core().read_model(str(xml_path))
    print("verified input shape:", reloaded.input(0).partial_shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

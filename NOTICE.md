# Notices and attribution

This repository contains third-party models, footage and a third-party document.
Their licences are **not** the same as this project's own source, and several carry
obligations that apply as soon as the repository — or a build produced from it — is
distributed.

**Status: recorded, not evaluated.** This file records what each upstream source
declares. It does not claim that every obligation has been assessed or discharged.
Anything a licence *requires* of a redistributor is called out, but whether this project
meets it is a legal question that has not been answered here.

Every licence below was read from the upstream source on 2026-11-11, not assumed. The
`Verified from` column says where.

---

## Models

| Model | Licence | Verified from |
|---|---|---|
| `yolo11n-fp16` | **AGPL-3.0** | Hugging Face `cardData.license` + `license:agpl-3.0` tag |
| `yolo11n-int8` | **AGPL-3.0** | Hugging Face `cardData.license` + `license:agpl-3.0` tag |
| `efficientnet-b0-fp16` | BSD-3-Clause | Hugging Face `cardData.license` |
| `efficientnet-b0-int8` | BSD-3-Clause | Hugging Face `cardData.license` |
| `clip-vision-patch32` | MIT | `api.github.com/repos/openai/CLIP/license` |
| `person-detection-retail-0013` | Apache-2.0 | Open Model Zoo repository licence |
| `product-detection-0001` | Apache-2.0 | Open Model Zoo repository licence |
| `person-vehicle-bike-detection-crossroad-1016` | Apache-2.0 | Open Model Zoo repository licence |
| `vehicle-license-plate-detection-barrier-0106` | Apache-2.0 | Open Model Zoo repository licence |
| `age-gender-recognition-retail-0013` | Apache-2.0 | Open Model Zoo repository licence |
| `human-pose-estimation-0001` | Apache-2.0 | Open Model Zoo repository licence |
| `Qwen3-Embedding-0.6B-int8-ov` | Apache-2.0 | Hugging Face `cardData.license` |
| `Qwen3-1.7B-int4-ov` | Apache-2.0 | Hugging Face `cardData.license` |

The Open Model Zoo model READMEs do not state a licence of their own; the figure above is
the licence of the repository that publishes them.

### `yolo11n` / `yolo11n-int8` — AGPL-3.0

The detector weights, and the OpenVINO IR derived from them, are licensed **GNU Affero
General Public License v3.0**.

**This obligation has not been evaluated.** AGPL-3.0 is a strong copyleft licence.
Distributing the model — together with application code that loads and uses it — can
carry source-disclosure requirements. Anyone redistributing this repository, or a package
built from it, must make that assessment themselves before doing so.

This is the only model in this project carrying a strong copyleft licence, and it is the
detector every scenario uses. It is recorded first here for exactly that reason.

- Model: YOLO11n, exported to OpenVINO IR, static 640×640, batch 1, no embedded NMS.
- Upstream: `https://huggingface.co/OpenVINO/YOLO11n-fp16-ov` (and the `-int8-ov` sibling).

---

## Media

The candidate clips under `video-assets/candidates/` are stock footage. Per-file
provenance and licence is recorded in `video-assets/candidates/README.md`; attribution is
not required by either licence but is recorded there anyway so the provenance is not lost.

| Source | Licence | Where recorded |
|---|---|---|
| Pexels | Pexels License | `video-assets/candidates/README.md` |
| Mixkit | Mixkit Free License | `video-assets/candidates/README.md` |

Those free licences cover redistribution. They do **not** grant any right to imply that
the people or businesses appearing in the footage endorse this demo.

---

## Document corpus

The document Q&A answers from **NIST SP 800-82 Rev. 4, Guide to Operational Technology
(OT) Security**, an Initial Public Draft.

NIST publications are works of the US Government and are not subject to copyright in the
United States (17 U.S.C. §105). The baked corpus under `models/rag/` is derived from that
document. The source PDF is **not** redistributed with this repository; only the extracted,
chunked text and its embedding index are, and both retain the document's own attribution.

`README-PORTABLE.md` and the Q&A panel both name the document, so a viewer is never left
guessing what the answers came from.

---

## Application source

No `LICENSE` file is present at the repository root, which means **the project's own
source has no declared licence**. This is a gap, not a decision: it was noticed while
writing this file.

Without a licence, the default position is that all rights are reserved, so nobody may
reuse this code even though the repository is public. If reuse is intended, a licence
should be chosen and added deliberately.

---

## Outstanding

Recorded honestly rather than resolved:

1. **The AGPL-3.0 obligation on `yolo11n` has not been assessed.** This is the open item
   with real consequences for distribution.
2. **No licence for this project's own source.** See above.
3. **Distributed artefacts are unsigned.** The portable bundle and any built package carry
   no Authenticode signature. They will trigger Windows SmartScreen. Nothing here
   fabricates a signature.
4. **The `openai/clip-vit-base-patch32` Hugging Face card declares no licence tag.** The
   MIT licence above is read from the model's own GitHub repository, which is the
   authoritative source for that project. It has not been cross-checked against the
   weights actually downloaded.
5. **Third-party notices are not bundled with the portable build.** `models/` ships inside
   it, but this `NOTICE.md` should be copied alongside when the bundle is distributed. It
   currently is not.

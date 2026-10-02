# Demo footage library

Candidate clips for the scenario videos, kept in git so the demo machine can fetch them without
anyone re-searching stock libraries or re-verifying licences.

**This is not the app's runtime media folder.** The application reads `media/`, which is
git-ignored and populated by `tools\download_models.py` from the URLs recorded in
`config\models.json`. Nothing here is read at runtime; these are the source clips the
config entries point at, plus the frame checks behind them.

## Why a separate folder

`media/` is ignored on purpose: the app's media store is machine-local and large. Committing stock
footage under an ignored path would be confusing, and putting it anywhere the app reads risks a
clipped/normalized version being mistaken for the original. These files are the originals as
downloaded, trimmed only to their usable span.

## Licences

Every clip here comes from a stock library that permits **commercial use with no attribution**:

| clip | source | licence |
|---|---|---|
| `mfg-warehouse-ppe-1080p.mp4` | Pexels 10817415 | Pexels License |
| `mfg-corridor-hardhats-720p.mp4` | Mixkit 23378 | Mixkit Free License |
| `robot-cell-workers-720p.mp4` | Pexels 6450803 | Pexels License |
| `edu-campus-walking-720p.mp4` | Mixkit 4519 | Mixkit Free License |
| `edu-hallway-walking-720p.mp4` | Pexels 8198509 | Pexels License |

Attribution is not required by either licence. It is recorded here anyway so the provenance of
every byte in this repository is traceable. If this repo is ever shown publicly or its footage
used in Intel marketing material, re-check both licence terms at that time — free stock licences
can change, and a repo that outlives the demo should not assume a licence from 2026 still holds.

Provenance for each clip — source URL, what was trimmed, and what the frame check actually
showed — is in `media-entries-to-add.json` and `docs/STATE.md` section 8.

## Verification standard applied

Each delivered file was checked by extracting frames across its full length, not by trusting the
title or the source listing. The bar was the same for every clip:

- landscape, fixed or near-fixed camera, no cuts
- no drone or aerial framing
- no heavy motion blur
- **a person visible in every sampled frame** — the rule that exists because an empty booth
  screenshot is worse than no demo
- 15-30s of usable loop is sufficient; 768x432-class resolution is sufficient

`verification-frames.jpg` is that evidence: 5 rows x 4 chronological frames, one row per clip.

## Known limitations, stated up front

- **No free manufacturing clip combines a worker in hi-vis/hard hat at 5-15 m with a forklift in
  frame.** Every forklift clip found put the operator 1-2 m from the camera, which defeats
  detection. The two clips here trade off: the warehouse one has PPE, the corridor one has the
  clearest hard hats, neither has both.
- **`robot-cell-workers-720p.mp4` has workers partly behind safety fencing**, so person detection
  will be intermittent. It is the only candidate found with robots *and* people in a fixed wide
  shot.
- **The lecture-hall and seated-classroom clips were rejected on purpose.** A person model trained
  on standing and walking people detects seated students poorly, which would have produced a booth
  screenshot with no boxes on it.

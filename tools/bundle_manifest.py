"""Write and verify a SHA-256 manifest for the portable booth bundle.

Why this exists: the bundle is handed to other people as a 2.6 GB zip. Without a
per-file manifest they cannot tell a complete copy from a truncated one, and a
half-extracted folder fails in confusing ways far from the cause. This records
the SHA-256 of every payload file, plus what the artefact is and is not.

It deliberately mirrors the honesty of the roadshow's bundle builder:

* the manifest states ``signed: false`` and carries a ``signature_note``. There
  is no Authenticode certificate for this project, so nothing here fabricates a
  signature or implies authenticity.
* files are hashed from bytes actually read off disk. If a file cannot be read
  the whole run fails rather than emitting a manifest with a gap in it.
* the manifest is written to a temporary file and renamed into place, so an
  interrupted run cannot leave a half-written manifest that looks authoritative.

Usage::

    python tools/bundle_manifest.py build  <bundle-dir>
    python tools/bundle_manifest.py verify <bundle-dir>

``verify`` exits non-zero and names every missing, changed or unexpected file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "bundle-manifest.json"
CHECKSUM_NAME = "MANIFEST.sha256"

#: Written into the manifest so a reader is never misled about what this is.
SIGNATURE_NOTE = (
    "Unsigned. This bundle carries no Authenticode signature: no certificate was "
    "available to produce one. The SHA-256 values below prove the files are the "
    "ones recorded here, not that this software is trusted or from a verified "
    "publisher. Windows SmartScreen will warn on first run."
)

#: Never hash the manifest files themselves, or a rebuild could never reproduce.
_SKIP = {MANIFEST_NAME, CHECKSUM_NAME}


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect(bundle: Path) -> list[tuple[str, int, str]]:
    """Return (relative posix path, bytes, sha256) for every payload file.

    Sorted by path so two builds of identical content produce identical output.
    """
    rows: list[tuple[str, int, str]] = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        if path.name in _SKIP:
            continue
        rel = path.relative_to(bundle).as_posix()
        rows.append((rel, path.stat().st_size, _digest(path)))
    return rows


def build(bundle: Path, source_commit: str | None) -> dict:
    rows = collect(bundle)
    if not rows:
        raise SystemExit(f"no files found under {bundle}; refusing to write an empty manifest")

    total_bytes = sum(size for _, size, _ in rows)
    manifest = {
        "bundle": bundle.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": source_commit,
        "signed": False,
        "signature_note": SIGNATURE_NOTE,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "algorithm": "sha256",
        "files": {rel: {"bytes": size, "sha256": digest} for rel, size, digest in rows},
    }

    # Checksum file in the conventional `sha256sum` format, so a recipient can
    # verify with standard tools and not only with this script.
    checksum_text = "".join(f"{digest}  {rel}\n" for rel, _, digest in rows)

    for name, text in ((MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n"), (CHECKSUM_NAME, checksum_text)):
        target = bundle / name
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temp, target)

    return manifest


def verify(bundle: Path) -> int:
    manifest_path = bundle / MANIFEST_NAME
    if not manifest_path.is_file():
        print(f"  FAIL  no {MANIFEST_NAME} in {bundle}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded: dict[str, dict] = manifest.get("files") or {}

    problems = 0
    for rel, entry in sorted(recorded.items()):
        path = bundle / rel
        if not path.is_file():
            print(f"  MISSING  {rel}")
            problems += 1
            continue
        if path.stat().st_size != entry["bytes"]:
            print(f"  SIZE     {rel}  expected {entry['bytes']}, found {path.stat().st_size}")
            problems += 1
            continue
        actual = _digest(path)
        if actual != entry["sha256"]:
            print(f"  CHANGED  {rel}\n           expected {entry['sha256']}\n           found    {actual}")
            problems += 1

    for rel, size, _ in collect(bundle):
        if rel not in recorded:
            print(f"  EXTRA    {rel}  ({size} bytes, not in the manifest)")
            problems += 1

    if problems:
        print(f"\n  {problems} problem(s). This copy does NOT match {MANIFEST_NAME}.")
        return 1

    print(f"  OK  {len(recorded)} files match {MANIFEST_NAME}")
    print(f"      signed: {manifest.get('signed')} — this proves the files are the recorded ones, not that they are trusted")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("bundle", type=Path, help="the bundle directory")
    parser.add_argument("--source-commit", default=None, help="git commit the bundle was built from")
    args = parser.parse_args(argv)

    if not args.bundle.is_dir():
        print(f"not a directory: {args.bundle}", file=sys.stderr)
        return 2

    if args.mode == "build":
        manifest = build(args.bundle, args.source_commit)
        print(f"  wrote {MANIFEST_NAME} and {CHECKSUM_NAME}")
        print(f"  {manifest['file_count']} files, {manifest['total_bytes'] / 1e9:.2f} GB")
        print(f"  signed: {manifest['signed']}")
        return 0

    return verify(args.bundle)


if __name__ == "__main__":
    raise SystemExit(main())

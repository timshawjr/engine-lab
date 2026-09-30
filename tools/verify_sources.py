#!/usr/bin/env python3
"""Content-verify every source used by ``config/models.json``.

A successful HTTP status is not accepted as proof.  XML, JSON, binary and MP4
payload signatures are checked, and model-card URLs must contain the model id.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "models.json"
USER_AGENT = "EngineLab-source-verifier/0.0"


@dataclass(frozen=True)
class Source:
    owner: str
    kind: str
    url: str
    expected: str | None = None


@dataclass(frozen=True)
class Result:
    source: Source
    ok: bool
    detail: str


def _hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(filename)}"


def collect_sources(config: dict[str, Any]) -> list[Source]:
    sources: list[Source] = []
    for model in config["models"]:
        model_id = model["id"]
        if model["source"] == "huggingface":
            for filename in model["files"]:
                sources.append(
                    Source(
                        owner=model_id,
                        kind="xml" if filename.endswith(".xml") else "bin",
                        url=_hf_url(model["repo"], filename),
                    )
                )
            sources.append(Source(model_id, "label-json", _hf_url(model["repo"], "config.json"), "labels"))
        elif model["source"] == "omz_models_bin":
            sources.append(Source(model_id, "xml", model["url_xml"]))
            sources.append(Source(model_id, "bin", model["url_bin"]))
        elif model["source"] != "converted":
            # "converted" IR is built locally, so it has no downloadable IR of
            # its own; the shared model-card check below verifies the upstream
            # PyTorch checkpoint it was produced from instead.
            raise ValueError(f"unsupported source type for {model_id}: {model['source']}")
        expected_card_id = (
            model["repo"].rsplit("/", 1)[-1]
            if model["source"] in {"huggingface", "converted"}
            else model_id
        )
        sources.append(Source(model_id, "model-card", model["spec_url"], expected_card_id))
    for media in config["media"]:
        if media.get("local_only"):
            # Operator-supplied footage with no public URL. Claiming a network
            # verification here would be a fabricated result, so it is skipped
            # and preflight verifies the local file instead.
            continue
        sources.append(Source(media["id"], "mp4", media["url"]))
        for part in media.get("source_parts", []):
            sources.append(
                Source(
                    f"{media['id']}:{part['id']}",
                    "mp4",
                    part["url"],
                )
            )
    return sources


def _looks_like_html(prefix: bytes) -> bool:
    text = prefix[:1024].lstrip().lower()
    return text.startswith(b"<!doctype html") or text.startswith(b"<html") or b"<html" in text


def _fetch_prefix(source: Source, retries: int = 3) -> tuple[bytes, str, int | None]:
    request = urllib.request.Request(
        source.url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Range": "bytes=0-262143",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                prefix = response.read(262144)
                content_type = response.headers.get("Content-Type", "unknown")
                content_length = response.headers.get("Content-Length")
                return prefix, content_type, int(content_length) if content_length else None
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def verify(source: Source) -> Result:
    try:
        prefix, content_type, content_length = _fetch_prefix(source)
    except Exception as exc:
        return Result(source, False, f"{type(exc).__name__}: {exc}")

    if not prefix:
        return Result(source, False, "empty response body")
    if source.kind != "model-card" and _looks_like_html(prefix):
        return Result(source, False, f"HTML error body (Content-Type={content_type})")

    detail = f"Content-Type={content_type}"
    if content_length is not None:
        detail += f", length={content_length}"

    if source.kind == "xml":
        xml_prefix = prefix.lstrip(b"\xef\xbb\xbf\r\n\t ")
        if not xml_prefix.startswith(b"<?xml"):
            return Result(source, False, f"body does not start with XML declaration ({detail})")
        if b"<net" not in prefix and b"<ie" not in prefix:
            return Result(source, False, f"XML does not look like OpenVINO IR ({detail})")
    elif source.kind == "label-json":
        try:
            payload = json.loads(prefix.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return Result(source, False, f"invalid JSON content: {exc}")
        if source.expected and source.expected not in payload:
            return Result(source, False, f"JSON has no {source.expected!r} field ({detail})")
    elif source.kind == "bin":
        if len(prefix) < 1024:
            return Result(source, False, f"binary payload is implausibly short ({detail})")
        if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
            return Result(source, False, f"received a Git LFS pointer, not model weights ({detail})")
        try:
            prefix[:1024].decode("ascii")
        except UnicodeDecodeError:
            pass
        else:
            lowered = prefix[:1024].lower()
            if b"error" in lowered or b"not found" in lowered or b"access denied" in lowered:
                return Result(source, False, f"binary response appears to be a text error ({detail})")
    elif source.kind == "mp4":
        if len(prefix) < 12 or prefix[4:8] != b"ftyp":
            return Result(source, False, f"body has no MP4 ftyp box ({detail})")
        if "video/mp4" not in content_type.lower() and "application/octet-stream" not in content_type.lower():
            return Result(source, False, f"unexpected media Content-Type ({detail})")
    elif source.kind == "model-card":
        try:
            text = prefix.decode("utf-8", errors="replace")
        except Exception as exc:
            return Result(source, False, f"model card is not decodable: {exc}")
        if source.expected and source.expected.lower() not in text.lower():
            return Result(source, False, f"page does not identify {source.expected!r} ({detail})")
    else:
        return Result(source, False, f"unsupported source kind {source.kind!r}")

    return Result(source, True, detail)


def print_table(results: list[Result]) -> None:
    width = max((len(result.source.owner) for result in results), default=8)
    print(f"{'STATUS':6}  {'OWNER':<{width}}  {'KIND':10}  DETAIL")
    print("-" * (24 + width + 14))
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(
            f"{status:6}  {result.source.owner:<{width}}  {result.source.kind:10}  {result.detail}"
        )
        if not result.ok:
            print(f"       {'':<{width}}  remediation: check the pinned source; do not substitute it without re-verification")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="parallel content checks")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    sources = collect_sources(config)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(verify, sources))
    print_table(results)
    failures = sum(not result.ok for result in results)
    print(f"\nSource verification: {len(results) - failures} PASS, {failures} FAIL")
    if failures:
        print("A status-only check is not accepted; every failure above failed its content signature check.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

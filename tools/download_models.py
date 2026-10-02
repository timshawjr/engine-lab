#!/usr/bin/env python3
"""Download every Engine Lab model, label source and prerecorded video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "models.json"
USER_AGENT = "EngineLab/0.0 (+https://localhost.invalid/engine-lab)"

OMZ_LABELS: dict[str, dict[str, Any]] = {
    "person-detection-retail-0013": {
        "class_ids": {"1": "person"},
    },
    "product-detection-0001": {
        "class_ids": {
            "0": "background_label",
            "1": "undefined",
            "2": "sprite",
            "3": "kool-aid",
            "4": "extra",
            "5": "ocelo",
            "6": "finish",
            "7": "mtn_dew",
            "8": "best_foods",
            "9": "gatorade",
            "10": "heinz",
            "11": "ruffles",
            "12": "pringles",
            "13": "del_monte",
        }
    },
    "person-vehicle-bike-detection-crossroad-1016": {
        "class_ids": {"0": "non-vehicle", "1": "vehicle", "2": "person"},
    },
    "vehicle-license-plate-detection-barrier-0106": {
        "class_ids": {"1": "vehicle", "2": "license plate"},
    },
    "age-gender-recognition-retail-0013": {
        "class_ids": {"0": "female", "1": "male"},
        "continuous_outputs": {"fc3_a": "age years = output * 100"},
    },
    "human-pose-estimation-0001": {
        "keypoints": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "neck",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
        "skeleton": [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
        ],
    },
}

POSE_DEMO_URL = (
    "https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/"
    "demos/human_pose_estimation_demo/python/human_pose_estimation_demo.py"
)


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_xml(path: Path) -> bool:
    try:
        prefix = path.read_bytes()[:256].lstrip(b"\xef\xbb\xbf\r\n\t ")
    except OSError:
        return False
    return prefix.startswith(b"<?xml")


def _valid_mp4(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
    except OSError:
        return False
    return len(header) >= 8 and header[4:8] == b"ftyp"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _vendor_local(media: dict[str, Any], destination: Path, force: bool) -> str:
    """Copy a clip that is committed in this repository instead of fetched.

    Some footage has no public download URL, so the bytes under
    ``video-assets/candidates/`` are the source of record. Copying them keeps
    ``media/`` machine-local and regenerable, exactly like a download, and the
    recorded sha256 proves the copy is byte-identical to what was reviewed.
    """
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return "cached"
    source = ROOT / media["local_source"]
    if not source.is_file():
        raise FileNotFoundError(
            f"{media['id']}: local_only media is missing its committed source {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    expected = media.get("sha256")
    if expected:
        digest = _sha256(destination)
        if digest != expected:
            destination.unlink(missing_ok=True)
            raise ValueError(
                f"{media['id']}: vendored copy sha256 {digest} does not match the "
                f"reviewed {expected}"
            )
    return "vendored"


def _download_http(url: str, destination: Path, force: bool, retries: int = 3) -> str:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                content_type = response.headers.get_content_type()
                if content_type == "text/html":
                    raise ValueError("storage returned text/html instead of model/media content")
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if temporary.stat().st_size <= 0:
                raise ValueError("server returned an empty body")
            os.replace(temporary, destination)
            return "downloaded"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_error}")


def _hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(filename)}"


def _download_hf_file(
    repo: str,
    filename: str,
    destination: Path,
    force: bool,
) -> tuple[str, str]:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return "cached", _hf_url(repo, filename)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download model repositories") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(destination.parent),
            force_download=force,
        )
    )
    if downloaded.resolve() != destination.resolve():
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copy2(downloaded, temporary)
        os.replace(temporary, destination)
    if destination.stat().st_size <= 0:
        raise RuntimeError(f"Hugging Face returned an empty file: {repo}/{filename}")
    return "downloaded", _hf_url(repo, filename)


def _write_hf_labels(model: dict[str, Any], model_dir: Path, force: bool) -> dict[str, Any]:
    source_config = model_dir / "source_config.json"
    status, source_url = _download_hf_file(model["repo"], "config.json", source_config, force)
    try:
        payload = json.loads(source_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Hugging Face config for {model['id']}: {exc}") from exc
    labels = payload.get("labels")
    if not isinstance(labels, str) or not labels.strip():
        raise RuntimeError(f"Hugging Face config for {model['id']} has no labels string")
    label_values = labels.split()
    label_path = model_dir / "labels.txt"
    if force or not label_path.exists() or label_path.read_text(encoding="utf-8").splitlines() != label_values:
        label_path.write_text("\n".join(label_values) + "\n", encoding="utf-8")
    labels_json = {
        "source_url": source_url,
        "labels": label_values,
        "preprocess": {
            key: payload[key]
            for key in ("mean_values", "scale_values", "reverse_input_channels", "resize_type", "pad_value")
            if key in payload
        },
    }
    _atomic_json(model_dir / "labels.json", labels_json)
    return {"status": status, "source_url": source_url, "count": len(label_values)}


def _write_omz_labels(model: dict[str, Any], model_dir: Path, force: bool) -> dict[str, Any]:
    try:
        payload = OMZ_LABELS[model["id"]]
    except KeyError as exc:
        raise RuntimeError(f"no sourced label definition for {model['id']}") from exc
    value = {
        "source_url": model["spec_url"],
        **payload,
    }
    if model["id"] == "human-pose-estimation-0001":
        value["skeleton_source_url"] = POSE_DEMO_URL
    path = model_dir / "labels.json"
    if force or not path.exists() or json.loads(path.read_text(encoding="utf-8")) != value:
        _atomic_json(path, value)
    return {"status": "generated from verified model documentation", "source_url": model["spec_url"]}


def _model_files(model: dict[str, Any]) -> list[tuple[str, str]]:
    if model["source"] == "huggingface":
        return [(filename, _hf_url(model["repo"], filename)) for filename in model["files"]]
    if model["source"] == "omz_models_bin":
        return [
            ("xml", model["url_xml"]),
            ("bin", model["url_bin"]),
        ]
    raise ValueError(f"unsupported model source {model['source']!r} for {model['id']}")


def _target_name(model: dict[str, Any], kind_or_filename: str, url: str) -> str:
    if model["source"] == "huggingface":
        return kind_or_filename
    if kind_or_filename == "xml":
        return Path(urllib.parse.urlparse(url).path).name
    return Path(urllib.parse.urlparse(url).path).name


def download_models(config: dict[str, Any], force: bool, include_media: bool) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "downloaded_on": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_verified_on": config.get("verified_on"),
        "models": {},
        "media": {},
    }
    failures: list[str] = []

    for model in config["models"]:
        model_id = model["id"]
        model_dir = ROOT / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"MODEL {model_id}")
        entry: dict[str, Any] = {"files": {}}
        if model.get("source") == "converted":
            # Some models are built locally rather than downloaded (the CLIP vision tower is
            # exported and FP16-compressed on a dev machine). That is not a download failure, so
            # it must not fail Phase 0 - but it has to say exactly what to run instead.
            entry["status"] = "requires_local_build"
            entry["files_expected"] = list(model.get("files", ()))
            entry["instruction"] = (
                "Build locally before preflight: "
                f"{model.get('conversion_tool', 'see config/models.json')}"
            )
            manifest["models"][model_id] = entry
            print(f"  SKIP: source 'converted' - build locally: {model.get('conversion_tool')}")
            continue
        try:
            for kind, url in _model_files(model):
                filename = _target_name(model, kind, url)
                destination = model_dir / filename
                if model["source"] == "huggingface":
                    status, actual_url = _download_hf_file(model["repo"], filename, destination, force)
                else:
                    status = _download_http(url, destination, force)
                    actual_url = url
                if filename.endswith(".xml") and not _valid_xml(destination):
                    raise RuntimeError(f"downloaded XML has invalid signature: {destination}")
                if destination.stat().st_size <= 0:
                    raise RuntimeError(f"downloaded file is empty: {destination}")
                digest = _sha256(destination)
                entry["files"][filename] = {
                    "status": status,
                    "source_url": actual_url,
                    "bytes": destination.stat().st_size,
                    "sha256": digest,
                }
                print(f"  {filename}: {status} ({destination.stat().st_size} bytes, sha256={digest})")

            if model["source"] == "huggingface":
                label_info = _write_hf_labels(model, model_dir, force)
            else:
                label_info = _write_omz_labels(model, model_dir, force)
            entry["labels"] = label_info
            print(f"  labels: {label_info['status']} ({label_info['source_url']})")
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(f"model {model_id}: {exc}")
            print(f"  FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        manifest["models"][model_id] = entry

    if include_media:
        for media in config["media"]:
            media_id = media["id"]
            destination = ROOT / "media" / media["file"]
            print(f"VIDEO {media_id}")
            try:
                if media.get("local_only"):
                    status = _vendor_local(media, destination, force)
                    origin = f"repo:{media['local_source']}"
                else:
                    status = _download_http(media["url"], destination, force)
                    origin = media["url"]
                if not _valid_mp4(destination):
                    raise RuntimeError(f"downloaded media has no MP4 ftyp signature: {destination}")
                digest = _sha256(destination)
                manifest["media"][media_id] = {
                    "status": status,
                    "source_url": None if media.get("local_only") else media["url"],
                    "provenance": origin,
                    "file": str(destination.relative_to(ROOT)),
                    "bytes": destination.stat().st_size,
                    "sha256": digest,
                }
                print(f"  {media['file']}: {status} ({destination.stat().st_size} bytes, sha256={digest})")
            except Exception as exc:
                manifest["media"][media_id] = {"error": f"{type(exc).__name__}: {exc}"}
                failures.append(f"media {media_id}: {exc}")
                print(f"  FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)

    manifest["failures"] = failures
    manifest_path = ROOT / "models" / "download_manifest.json"
    _atomic_json(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download even when a valid local file exists")
    parser.add_argument("--models-only", action="store_true", help="skip prerecorded videos")
    parser.add_argument("--media-only", action="store_true", help="skip model repositories")
    args = parser.parse_args(argv)
    if args.models_only and args.media_only:
        parser.error("--models-only and --media-only are mutually exclusive")

    config = _load_config()
    if args.media_only:
        manifest: dict[str, Any] = {
            "downloaded_on": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_verified_on": config.get("verified_on"),
            "models": {},
            "media": {},
        }
        failures: list[str] = []
        for media in config["media"]:
            destination = ROOT / "media" / media["file"]
            try:
                if media.get("local_only"):
                    status = _vendor_local(media, destination, args.force)
                    origin = f"repo:{media['local_source']}"
                else:
                    status = _download_http(media["url"], destination, args.force)
                    origin = media["url"]
                if not _valid_mp4(destination):
                    raise RuntimeError(f"downloaded media has no MP4 ftyp signature: {destination}")
                digest = _sha256(destination)
                manifest["media"][media["id"]] = {
                    "status": status,
                    "source_url": None if media.get("local_only") else media["url"],
                    "provenance": origin,
                    "file": str(destination.relative_to(ROOT)),
                    "bytes": destination.stat().st_size,
                    "sha256": digest,
                }
                print(f"VIDEO {media['id']}: {status} ({destination.stat().st_size} bytes, sha256={digest})")
            except Exception as exc:
                manifest["media"][media["id"]] = {"error": f"{type(exc).__name__}: {exc}"}
                failures.append(f"media {media['id']}: {exc}")
        manifest["failures"] = failures
        _atomic_json(ROOT / "models" / "download_manifest.json", manifest)
    else:
        manifest = download_models(config, args.force, include_media=not args.models_only)

    failures = manifest.get("failures", [])
    print(f"Download summary: {len(manifest['models'])} model entries, {len(manifest['media'])} media entries, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

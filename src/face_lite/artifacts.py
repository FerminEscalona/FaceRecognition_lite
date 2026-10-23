"""Pinned upstream assets; inference never downloads implicitly."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
SOURCE = "https://github.com/opencv/opencv_zoo"
ASSETS = {
    "yunet": {
        "file": "face_detection_yunet_2023mar.onnx",
        "folder": "face_detection_yunet",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "bytes": 232589,
        "license": "MIT",
    },
    "fp32": {
        "file": "face_recognition_sface_2021dec.onnx",
        "folder": "face_recognition_sface",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "bytes": 38696353,
        "license": "Apache-2.0 (as distributed by OpenCV Zoo)",
    },
    "int8": {
        "file": "face_recognition_sface_2021dec_int8.onnx",
        "folder": "face_recognition_sface",
        "sha256": "2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a",
        "bytes": 9896933,
        "license": "Apache-2.0 (as distributed by OpenCV Zoo)",
    },
}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def verify_asset(directory: Path, key: str) -> Path:
    asset = ASSETS[key]
    path = directory / asset["file"]
    if not path.is_file():
        raise ValueError(f"Missing model {path}; run face-lite download-models explicitly")
    if path.stat().st_size != asset["bytes"] or digest_file(path) != asset["sha256"]:
        raise ValueError(f"Model integrity check failed: {path.name}")
    return path


def download_file(url: str, destination: Path, expected_hash: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".download-")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=120) as source:
            while chunk := source.read(1024 * 1024):
                out.write(chunk)
        if expected_hash and digest_file(Path(temporary)) != expected_hash:
            raise ValueError(f"SHA-256 mismatch downloading {destination.name}")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def download_models(directory: Path, variants: list[str]) -> dict:
    for key in ["yunet", *variants]:
        asset = ASSETS[key]
        destination = directory / asset["file"]
        if destination.exists():
            verify_asset(directory, key)
        else:
            url = (f"https://media.githubusercontent.com/media/opencv/opencv_zoo/"
                   f"{REVISION}/models/{asset['folder']}/{asset['file']}")
            download_file(url, destination, asset["sha256"])
            verify_asset(directory, key)
        license_path = directory / f"{asset['folder']}.LICENSE"
        if not license_path.exists():
            download_file(f"https://raw.githubusercontent.com/opencv/opencv_zoo/{REVISION}/"
                          f"models/{asset['folder']}/LICENSE", license_path)
    manifest = {"source": SOURCE, "revision": REVISION,
                "assets": {k: ASSETS[k] for k in ["yunet", *variants]}}
    (directory / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


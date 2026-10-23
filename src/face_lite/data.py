"""Explicit dataset preparation and leakage checks. No training/test resplitting."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .artifacts import digest_file, download_file, fingerprint

ORL_URL = "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip"
ORL_SOURCE = "https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html"
ORL_SHA256 = "509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec"


def prepare_orl(directory: Path) -> Path:
    """Fixed split; ORL is a functional experiment, never cross-camera evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "att_faces.zip"
    if not archive.exists():
        download_file(ORL_URL, archive, ORL_SHA256)
    if digest_file(archive) != ORL_SHA256:
        raise ValueError("ORL archive integrity check failed")
    image_root = directory / "images"
    image_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        # Only the expected 400 image entries are extracted, never archive paths.
        for subject in range(1, 41):
            subject_dir = image_root / f"s{subject}"
            subject_dir.mkdir(exist_ok=True)
            for index in range(1, 11):
                name = f"s{subject}/{index}.pgm"
                content = zipped.read(name)
                if len(content) > 100_000:
                    raise ValueError("Unexpected ORL archive entry size")
                (image_root / name).write_bytes(content)
    enrollment, validation, test = [], [], []
    for subject in range(1, 6):
        person = f"s{subject}"
        enrollment.append({"person_id": person, "session": f"{person}-registration",
                           "images": [f"images/{person}/{i}.pgm" for i in (1, 2, 3)]})
        for split, indices, label in [(validation, (4, 5, 6), "validation"), (test, (7, 8, 9, 10), "test")]:
            for i in indices:
                split.append({"image": f"images/{person}/{i}.pgm", "person_id": person,
                              "subject_id": person, "session": f"{person}-{label}"})
    for split, subjects, label in [(validation, range(6, 11), "validation"), (test, range(11, 21), "test")]:
        for subject in subjects:
            for index in range(1, 11):
                split.append({"image": f"images/s{subject}/{index}.pgm", "person_id": None,
                              "subject_id": f"s{subject}", "session": f"s{subject}-{label}"})
    manifest = {"schema": 1, "scope": "public-functional-only", "source": ORL_SOURCE,
                "attribution": "AT&T Laboratories Cambridge",
                "archive_sha256": digest_file(archive),
                "limitations": "92x112 grayscale; no camera/session guarantees; session labels here are split labels only",
                "enrollment": enrollment, "validation": validation, "test": test}
    destination = directory / "manifest.json"
    write_json(destination, manifest)
    return destination


def write_json(destination: str | Path, value: object) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def load_manifest(path: Path) -> tuple[dict, str]:
    manifest = read_json(path)
    if manifest.get("schema") != 1 or manifest.get("scope") not in ("public-functional-only", "phone-robot"):
        raise ValueError("Unsupported dataset schema/scope")
    registered = [row["person_id"] for row in manifest["enrollment"]]
    if not registered or len(set(registered)) != len(registered):
        raise ValueError("Enrollment identities must be unique and nonempty")
    if any(not isinstance(p, str) or not p for p in registered):
        raise ValueError("Invalid enrollment person_id")
    unknown: dict[str, set] = {"validation": set(), "test": set()}
    seen: set[str] = set()
    session_splits: dict[tuple[str, str], str] = {}
    content = []

    def validate_image(relative: str, subject: str, session: str, split: str):
        image = (path.parent / relative).resolve()
        if not image.is_relative_to(path.parent.resolve()) or not image.is_file():
            raise ValueError("Dataset paths must exist inside manifest directory")
        if not isinstance(session, str) or not session:
            raise ValueError("Every sample requires a session label")
        key = (subject, session)
        if key in session_splits and session_splits[key] != split:
            raise ValueError("Session leakage between dataset partitions")
        session_splits[key] = split
        digest = digest_file(image)
        if digest in seen:
            raise ValueError("Duplicate image content across samples or partitions")
        seen.add(digest)
        content.append({"split": split, "subject": subject, "session": session, "sha256": digest})

    for row in manifest["enrollment"]:
        if len(row["images"]) != 3:
            raise ValueError("Each enrollment needs exactly three images")
        for image in row["images"]:
            validate_image(image, row["person_id"], row["session"], "enrollment")
    for split in ("validation", "test"):
        if not manifest[split]:
            raise ValueError("Both validation and test samples are required")
        for row in manifest[split]:
            person, subject = row["person_id"], row["subject_id"]
            if not isinstance(subject, str) or not subject:
                raise ValueError("subject_id is required even for unknown subjects")
            if person is not None and (person not in registered or subject != person):
                raise ValueError("Known sample identity is not enrolled or has conflicting subject_id")
            if person is None:
                if subject in registered:
                    raise ValueError("An unknown identity cannot also be enrolled")
                unknown[split].add(subject)
            validate_image(row["image"], subject, row["session"], split)
        if not unknown[split] or not any(row["person_id"] is not None for row in manifest[split]):
            raise ValueError("Each partition needs known and unknown queries")
    if unknown["validation"] & unknown["test"]:
        raise ValueError("Unknown identities must be disjoint between validation and test")
    return manifest, fingerprint({"manifest": manifest, "content": content})

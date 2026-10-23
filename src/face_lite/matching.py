"""Strict template contract and open-set decision rule, independent of OpenCV."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .artifacts import fingerprint


def vector(value: Any, dimension: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (dimension,) or raw.dtype.kind not in "fiu":
        raise ValueError("Embedding must be a numeric one-dimensional vector of declared size")
    a = raw.astype(np.float32)
    if not np.all(np.isfinite(a)) or not np.isclose(np.linalg.norm(a), 1, atol=1e-4):
        raise ValueError("Embedding must be finite and L2-normalized")
    return a


@dataclass(frozen=True)
class Calibration:
    model_version: str
    threshold: float
    margin: float
    validation_digest: str
    enrollment_min_similarity: float
    scoring: str = "mean-top2-of3-cosine-v1"
    scope: str = "preliminary"

    def __post_init__(self):
        if not self.model_version or not self.validation_digest:
            raise ValueError("Calibration needs model_version and validation provenance")
        if not np.isfinite(self.threshold) or not -1 <= self.threshold <= 1.000001:
            raise ValueError("Invalid threshold")
        if not np.isfinite(self.margin) or not 0 <= self.margin <= 2.000001:
            raise ValueError("Invalid margin")
        if not np.isfinite(self.enrollment_min_similarity) or not -1 <= self.enrollment_min_similarity <= 1:
            raise ValueError("Invalid enrollment consistency threshold")
        if self.scoring != "mean-top2-of3-cosine-v1":
            raise ValueError("Unsupported scoring rule")

    @property
    def calibration_id(self) -> str:
        return fingerprint(asdict(self))[:16]

    def to_dict(self) -> dict:
        return asdict(self)


class Gallery:
    """Immutable validated copy; validate once when replacing a local snapshot."""

    def __init__(self, rows: list[dict], model_version: str, dimension: int = 128):
        if not isinstance(rows, list):
            raise ValueError("Gallery must be a JSON array of face_templates rows")
        grouped: dict[str, list[np.ndarray]] = {}
        template_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"template_id", "person_id", "embedding", "model_version"}:
                raise ValueError("Invalid face_templates fields")
            if row["model_version"] != model_version:
                raise ValueError("Incompatible model_version")
            for field in ("person_id", "template_id"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"{field} must be a nonempty string")
            if row["template_id"] in template_ids:
                raise ValueError("Duplicate template_id")
            template_ids.add(row["template_id"])
            grouped.setdefault(row["person_id"], []).append(vector(row["embedding"], dimension))
        if any(len(items) != 3 for items in grouped.values()):
            raise ValueError("Every person must have exactly three templates")
        for items in grouped.values():
            if len({item.tobytes() for item in items}) != 3:
                raise ValueError("Duplicate embeddings cannot substitute three distinct captures")
        self.person_ids = tuple(sorted(grouped))
        self.model_version = model_version
        self.dimension = dimension
        self._matrix = (np.stack([np.stack(grouped[p]) for p in self.person_ids])
                        if grouped else np.empty((0, 3, dimension), dtype=np.float32))
        self._matrix.setflags(write=False)

    def scores(self, embedding: Any) -> list[tuple[str, float]]:
        query = vector(embedding, self.dimension)
        if not self.person_ids:
            return []
        similarities = np.clip(self._matrix @ query, -1, 1)
        scores = np.sort(similarities, axis=1)[:, -2:].mean(axis=1)
        return sorted(zip(self.person_ids, map(float, scores)), key=lambda pair: (-pair[1], pair[0]))

    def check_consistency(self, minimum: float) -> None:
        for items in self._matrix:
            pairwise = items @ items.T
            if float(pairwise[np.triu_indices(3, 1)].min()) < minimum:
                raise ValueError("Enrollment templates fail consistency policy")


def decide(scores: list[tuple[str, float]], policy: Calibration) -> dict:
    base = {"person_id": None, "similarity": None, "gap": None}
    if not scores:
        return {**base, "status": "unknown"}
    person, similarity = scores[0]
    # None means there is no second identity, not a fictitious competing template.
    gap = similarity - scores[1][1] if len(scores) > 1 else None
    base.update(similarity=similarity, gap=gap)
    if similarity < policy.threshold:
        return {**base, "status": "unknown"}
    if gap is not None and (gap <= 1e-7 or gap < policy.margin):
        return {**base, "status": "ambiguous"}
    return {**base, "status": "identified", "person_id": person}


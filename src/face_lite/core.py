"""One shared, locked YuNet/SFace pipeline for phone enrollment and robot queries."""
from __future__ import annotations

import hashlib
import io
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .artifacts import ASSETS, REVISION, fingerprint, verify_asset
from .matching import Calibration, Gallery, decide

POSITIONS = ("front", "left", "right")


@dataclass(frozen=True)
class QualityConfig:
    min_face_pixels: int = 50
    min_laplacian_variance: float = 20.0
    min_brightness: float = 30.0
    max_brightness: float = 225.0

    def __post_init__(self):
        if (not isinstance(self.min_face_pixels, int) or self.min_face_pixels < 1
                or not np.isfinite(self.min_laplacian_variance) or self.min_laplacian_variance < 0):
            raise ValueError("Invalid quality limits")
        if not 0 <= self.min_brightness < self.max_brightness <= 255:
            raise ValueError("Invalid brightness limits")


@dataclass(frozen=True)
class PipelineConfig:
    variant: str = "fp32"
    threads: int = 2
    detector_score: float = 0.8
    detector_max_side: int = 640
    enrollment_min_similarity: float = 0.3
    quality: QualityConfig = field(default_factory=QualityConfig)

    def __post_init__(self):
        if self.variant not in ("fp32", "int8") or not isinstance(self.threads, int) or not 1 <= self.threads <= 64:
            raise ValueError("Invalid variant or thread count")
        if (not 0 < self.detector_score <= 1 or not isinstance(self.detector_max_side, int)
                or not 64 <= self.detector_max_side <= 2048):
            raise ValueError("Invalid detector configuration")
        if not -1 <= self.enrollment_min_similarity <= 1:
            raise ValueError("Invalid enrollment consistency limit")

    @classmethod
    def from_dict(cls, value: dict) -> PipelineConfig:
        value = dict(value)
        if "quality" in value:
            value["quality"] = QualityConfig(**value["quality"])
        return cls(**value)


class FaceEngine:
    """Own one instance per worker; operations serialize access to native model state.

    Arrays are uint8 BGR HWC. Files/encoded bytes are EXIF-oriented via Pillow.
    OpenCV thread count is process-global: use the same setting for all workers
    in a process. Do not change it during active inference.
    """

    dimension = 128

    def __init__(self, models: str | Path = "artifacts/models", config: PipelineConfig | None = None):
        start = time.perf_counter()
        self.config = config or PipelineConfig()
        detector_path = verify_asset(Path(models), "yunet")
        extractor_path = verify_asset(Path(models), self.config.variant)
        cv2.setNumThreads(self.config.threads)
        self._lock = threading.RLock()
        self._detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320),
                                                  self.config.detector_score, 0.3, 5000,
                                                  cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        self._extractor = cv2.FaceRecognizerSF.create(str(extractor_path), "",
                                                     cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        self.manifest = {
            "schema": 1, "dimension": self.dimension, "dtype": "float32", "normalization": "L2",
            "revision": REVISION, "detector": ASSETS["yunet"], "extractor": ASSETS[self.config.variant],
            "opencv": cv2.__version__, "preprocessing": "exif-rgb-to-bgr-yunet-landmarks-alignCrop-sface-v1",
            "config": asdict(self.config), "quality_policy_status": "provisional-until-domain-validation",
        }
        # Threads do not change the semantic version of the embedding contract.
        identity = {**self.manifest, "config": {k: v for k, v in asdict(self.config).items() if k != "threads"}}
        self.model_version = f"sface-{self.config.variant}-" + fingerprint(identity)[:20]
        self.manifest["model_version"] = self.model_version
        self.cold_start_ms = (time.perf_counter() - start) * 1000

    @staticmethod
    def _load(image: str | Path | bytes | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
                raise ValueError("Expected uint8 BGR HWC array")
            if image.shape[0] * image.shape[1] > 20_000_000:
                raise ValueError("Image exceeds 20 megapixels")
            if min(image.shape[:2]) < 32:
                raise ValueError("Image dimensions must be at least 32 pixels")
            return np.ascontiguousarray(image).copy()
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.stat().st_size > 20_000_000:
                raise ValueError("Encoded image exceeds 20 MB")
            data = path.read_bytes()
        elif isinstance(image, bytes):
            data = image
        else:
            raise ValueError("Expected file path, encoded bytes or BGR array")
        if len(data) > 20_000_000:
            raise ValueError("Encoded image exceeds 20 MB")
        with Image.open(io.BytesIO(data)) as pil:
            if pil.width * pil.height > 20_000_000 or getattr(pil, "n_frames", 1) != 1:
                raise ValueError("Expected one image, at most 20 megapixels")
            if min(pil.size) < 32:
                raise ValueError("Image dimensions must be at least 32 pixels")
            rgb = np.asarray(ImageOps.exif_transpose(pil).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def extract(self, image: str | Path | bytes | np.ndarray) -> dict:
        started = time.perf_counter()
        result: dict[str, Any] = {"status": "invalid_image", "embedding": None,
                                  "model_version": self.model_version, "quality": {}, "timings_ms": {}}

        def finish(status: str, reason: str | None = None) -> dict:
            result.update(status=status, reason=reason, elapsed_ms=(time.perf_counter() - started) * 1000)
            return result

        try:
            bgr = self._load(image)
        except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
            return finish("invalid_image", "decode_or_size")
        result["image_digest"] = hashlib.sha256(bgr.tobytes() + str(bgr.shape).encode()).hexdigest()
        result["timings_ms"]["decode"] = (time.perf_counter() - started) * 1000
        with self._lock:
            stage = time.perf_counter()
            h, w = bgr.shape[:2]
            scale = min(1.0, self.config.detector_max_side / max(h, w))
            preview = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))))
            sx, sy = preview.shape[1] / w, preview.shape[0] / h
            self._detector.setInputSize((preview.shape[1], preview.shape[0]))
            _, faces = self._detector.detect(preview)
            result["timings_ms"]["detect"] = (time.perf_counter() - stage) * 1000
            if faces is None or len(faces) == 0:
                return finish("no_face")
            if len(faces) != 1:
                return finish("multiple_faces")
            face = faces[0].copy()
            face[[0, 2, 4, 6, 8, 10, 12]] /= sx
            face[[1, 3, 5, 7, 9, 11, 13]] /= sy
            stage = time.perf_counter()
            if min(face[2:4]) < self.config.quality.min_face_pixels:
                return finish("insufficient_quality", "face_too_small")
            aligned = self._extractor.alignCrop(bgr, face)
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())
            result["quality"] = {"laplacian_variance": blur, "brightness": brightness,
                                 "face_pixels": float(min(face[2:4])), "detector_score": float(face[-1])}
            result["timings_ms"]["align_quality"] = (time.perf_counter() - stage) * 1000
            q = self.config.quality
            if blur < q.min_laplacian_variance:
                return finish("insufficient_quality", "blur")
            if not q.min_brightness <= brightness <= q.max_brightness:
                return finish("insufficient_quality", "lighting")
            stage = time.perf_counter()
            embedding = self._extractor.feature(aligned).reshape(-1).astype(np.float32).copy()
            norm = np.linalg.norm(embedding)
            if embedding.shape != (self.dimension,) or not np.isfinite(embedding).all() or norm < 1e-12:
                raise RuntimeError("Extractor output violates the model manifest")
            result["embedding"] = (embedding / norm).tolist()
            result["timings_ms"]["embedding"] = (time.perf_counter() - stage) * 1000
        return finish("ok")

    def template(self, image, person_id: str, template_id: str) -> dict:
        if not all(isinstance(s, str) and s.strip() for s in (person_id, template_id)):
            raise ValueError("Identifiers must be nonempty strings supplied by the integrator")
        result = self.extract(image)
        row = None
        if result["status"] == "ok":
            row = {"template_id": template_id, "person_id": person_id,
                   "embedding": result["embedding"], "model_version": self.model_version}
        return {"status": result["status"], "reason": result.get("reason"), "template": row,
                "elapsed_ms": result["elapsed_ms"]}

    def enroll(self, images: list, person_id: str, template_ids: list[str]) -> dict:
        if len(images) != 3 or len(template_ids) != 3 or len(set(template_ids)) != 3:
            raise ValueError("Supply three ordered images and three distinct template identifiers")
        if not isinstance(person_id, str) or not person_id.strip() or any(not isinstance(t, str) or not t.strip() for t in template_ids):
            raise ValueError("Identifiers must be nonempty strings")
        started = time.perf_counter()
        captures, extractions, seen = [], [], set()
        for position, image in zip(POSITIONS, images):
            item = self.extract(image)
            digest = item.get("image_digest")
            if item["status"] == "ok" and digest in seen:
                item.update(status="duplicate_capture", reason="repeat_this_position")
            seen.add(digest)
            extractions.append(item)
            captures.append({"position": position, "status": item["status"], "reason": item.get("reason")})
        rows = []
        status = "recapture_required"
        consistency = None
        if all(item["status"] == "ok" for item in extractions):
            matrix = np.array([item["embedding"] for item in extractions], dtype=np.float32)
            consistency = float((matrix @ matrix.T)[np.triu_indices(3, 1)].min())
            if consistency < self.config.enrollment_min_similarity:
                status = "inconsistent_enrollment"
                for capture in captures:
                    capture.update(status="recapture_required", reason="identity_consistency")
            else:
                rows = [{"template_id": tid, "person_id": person_id, "embedding": item["embedding"],
                         "model_version": self.model_version} for tid, item in zip(template_ids, extractions)]
                try:
                    Gallery(rows, self.model_version).check_consistency(self.config.enrollment_min_similarity)
                    status = "enrolled"
                except ValueError:
                    rows = []
                    status = "inconsistent_enrollment"
                    for capture in captures:
                        capture.update(status="recapture_required", reason="duplicate_or_inconsistent_embedding")
        return {"status": status, "captures": captures, "templates": rows, "min_pair_similarity": consistency,
                "model_version": self.model_version, "elapsed_ms": (time.perf_counter() - started) * 1000}

    def identify(self, image, gallery: Gallery, policy: Calibration) -> dict:
        started = time.perf_counter()
        base = {"person_id": None, "similarity": None, "model_version": self.model_version,
                "calibration_id": policy.calibration_id}
        if (gallery.model_version != self.model_version or policy.model_version != self.model_version
                or policy.enrollment_min_similarity != self.config.enrollment_min_similarity):
            return {**base, "status": "incompatible_gallery", "elapsed_ms": (time.perf_counter() - started) * 1000}
        try:
            gallery.check_consistency(policy.enrollment_min_similarity)
        except ValueError:
            return {**base, "status": "incompatible_gallery", "elapsed_ms": (time.perf_counter() - started) * 1000}
        item = self.extract(image)
        if item["status"] != "ok":
            return {**base, "status": item["status"], "reason": item.get("reason"),
                    "timings_ms": item["timings_ms"], "elapsed_ms": (time.perf_counter() - started) * 1000}
        stage = time.perf_counter()
        decision = decide(gallery.scores(item["embedding"]), policy)
        item["timings_ms"]["match"] = (time.perf_counter() - stage) * 1000
        return {**base, **decision, "timings_ms": item["timings_ms"],
                "elapsed_ms": (time.perf_counter() - started) * 1000}

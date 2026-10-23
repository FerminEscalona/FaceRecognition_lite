import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from face_lite import Calibration, FaceEngine, Gallery, PipelineConfig


def test_exif_bytes_and_bgr_array_share_orientation_and_color():
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    rgb[:, :40] = [255, 0, 0]
    image = Image.fromarray(rgb)
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", exif=exif)
    loaded = FaceEngine._load(buffer.getvalue())
    expected = np.rot90(rgb, k=3)[:, :, ::-1]
    np.testing.assert_array_equal(loaded, expected)
    np.testing.assert_array_equal(FaceEngine._load(loaded), expected)


def fake_engine(monkeypatch, results):
    engine = object.__new__(FaceEngine)
    engine.config = PipelineConfig()
    engine.model_version = "fake"
    items = iter(results)
    monkeypatch.setattr(engine, "extract", lambda image: next(items))
    return engine


def extraction(index, status="ok"):
    vector = np.zeros(128)
    vector[0], vector[1] = 1, .01 * index
    vector /= np.linalg.norm(vector)
    return {"status": status, "embedding": vector.tolist(), "image_digest": str(index),
            "model_version": "fake", "reason": None, "elapsed_ms": .1}


def test_atomic_registration_and_recapture_position(monkeypatch):
    engine = fake_engine(monkeypatch, [extraction(1), extraction(2, "no_face"), extraction(3)])
    result = engine.enroll(["a", "b", "c"], "person", ["a", "b", "c"])
    assert result["templates"] == []
    assert result["captures"][1]["position"] == "left"
    assert result["captures"][1]["status"] == "no_face"


def test_duplicate_photo_cannot_complete_registration(monkeypatch):
    engine = fake_engine(monkeypatch, [extraction(1), extraction(1), extraction(2)])
    result = engine.enroll(["a", "a", "b"], "person", ["a", "b", "c"])
    assert result["templates"] == []
    assert result["captures"][1]["status"] == "duplicate_capture"


def test_three_templates_use_exact_supabase_contract(monkeypatch):
    engine = fake_engine(monkeypatch, [extraction(1), extraction(2), extraction(3)])
    result = engine.enroll(["a", "b", "c"], "person", ["a", "b", "c"])
    assert result["status"] == "enrolled"
    assert len(result["templates"]) == 3
    assert set(result["templates"][0]) == {"person_id", "template_id", "embedding", "model_version"}
    assert Gallery(result["templates"], "fake").person_ids == ("person",)


@pytest.fixture(scope="module")
def real_engine():
    if not Path("artifacts/models/face_recognition_sface_2021dec.onnx").exists():
        pytest.skip("Run face-lite download-models to exercise real ONNX inference")
    return FaceEngine()


@pytest.mark.integration
def test_real_corrupt_and_no_face_images(real_engine):
    assert real_engine.extract(b"not an image")["status"] == "invalid_image"
    assert real_engine.extract(np.zeros((240, 320, 3), dtype=np.uint8))["status"] == "no_face"
    assert real_engine.extract(np.zeros((1, 1, 3), dtype=np.uint8))["status"] == "invalid_image"


@pytest.mark.integration
def test_real_model_consistency_and_version_gate(real_engine):
    image = Path("data/orl/images/s1/1.pgm")
    if not image.exists():
        pytest.skip("Run face-lite prepare-orl")
    first = real_engine.template(image, "p", "t")
    second = real_engine.extract(image)
    assert first["status"] == second["status"] == "ok"
    np.testing.assert_allclose(first["template"]["embedding"], second["embedding"], atol=1e-7)
    incompatible = Calibration("different-model", .9, .1, "validation", .3)
    assert real_engine.identify(image, Gallery([], real_engine.model_version), incompatible)["status"] == "incompatible_gallery"


@pytest.mark.integration
def test_real_multiple_faces_are_not_silently_selected(real_engine):
    import cv2
    first = Path("data/orl/images/s1/1.pgm")
    second = Path("data/orl/images/s2/1.pgm")
    if not first.exists() or not second.exists():
        pytest.skip("Run face-lite prepare-orl")
    collage = np.concatenate([FaceEngine._load(first), FaceEngine._load(second)], axis=1)
    collage = cv2.resize(collage, None, fx=2, fy=2)
    assert real_engine.extract(collage)["status"] == "multiple_faces"


@pytest.mark.integration
def test_quality_rejection_has_no_embedding(real_engine):
    from face_lite import QualityConfig
    image = Path("data/orl/images/s1/1.pgm")
    if not image.exists():
        pytest.skip("Run face-lite prepare-orl")
    engine = FaceEngine(config=PipelineConfig(quality=QualityConfig(min_laplacian_variance=1e12)))
    result = engine.extract(image)
    assert result["status"] == "insufficient_quality"
    assert result["embedding"] is None


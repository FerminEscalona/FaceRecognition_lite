from pathlib import Path
from types import SimpleNamespace

import numpy as np

from face_lite.data import read_json, write_json
from face_lite.evaluation import evaluate


def setup_experiment(tmp_path):
    for i in range(7):
        (tmp_path / f"{i}.png").write_bytes(f"distinct-{i}".encode())
    manifest = {"schema": 1, "scope": "phone-robot",
                "enrollment": [{"person_id": "p", "session": "phone", "images": ["0.png", "1.png", "2.png"]}],
                "validation": [{"image": "3.png", "person_id": "p", "subject_id": "p", "session": "val"},
                               {"image": "4.png", "person_id": None, "subject_id": "u", "session": "val"}],
                "test": [{"image": "5.png", "person_id": "p", "subject_id": "p", "session": "test"},
                         {"image": "6.png", "person_id": None, "subject_id": "v", "session": "test"}]}
    write_json(tmp_path / "manifest.json", manifest)
    called = []
    vectors = []
    for i in range(3):
        a = np.zeros(128)
        a[0], a[1] = 1, .01*(i+1)
        vectors.append((a/np.linalg.norm(a)).tolist())
    rows = [{"template_id": f"t{i}", "person_id": "p", "model_version": "fake", "embedding": a}
            for i, a in enumerate(vectors)]

    def extract(path):
        called.append(Path(path).name)
        a = np.zeros(128)
        a[0 if Path(path).stem in ("3", "5") else 1] = 1
        return {"status": "ok", "embedding": a.tolist(), "timings_ms": {}}

    engine = SimpleNamespace(model_version="fake", manifest={}, cold_start_ms=1,
                             config=SimpleNamespace(enrollment_min_similarity=.3), extract=extract,
                             enroll=lambda *args: {"templates": rows, "status": "enrolled"})
    return engine, called


def test_calibrate_does_not_run_test_inference(tmp_path):
    engine, called = setup_experiment(tmp_path)
    report = evaluate(engine, tmp_path / "manifest.json", tmp_path / "cal", mode="calibrate")
    assert set(called) == {"3.png", "4.png"}
    assert "test" not in report
    assert report["acceptance"] is False


def test_frozen_evaluation_never_searches_again(tmp_path, monkeypatch):
    engine, called = setup_experiment(tmp_path)
    evaluate(engine, tmp_path / "manifest.json", tmp_path / "cal", mode="calibrate")
    frozen = read_json(tmp_path / "cal/calibration.json")

    def forbidden(*args):
        raise AssertionError("Test must not search thresholds")

    monkeypatch.setattr("face_lite.evaluation.calibrate", forbidden)
    # A tiny runtime rounding difference must not change dataset provenance.
    original = engine.extract

    def slightly_different_runtime(path):
        result = original(path)
        if Path(path).name == "3.png":
            value = np.array(result["embedding"])
            value[1] = 1e-4
            result["embedding"] = (value/np.linalg.norm(value)).tolist()
        return result

    engine.extract = slightly_different_runtime
    report = evaluate(engine, tmp_path / "manifest.json", tmp_path / "test", mode="evaluate",
                      calibration_path=tmp_path / "cal/calibration.json")
    assert report["calibration"] == frozen
    assert report["test"]["correct_known"] == 1
    assert report["acceptance"] is False  # Insufficient coverage and no real robot.


def test_changed_validation_data_rejects_frozen_calibration(tmp_path):
    engine, called = setup_experiment(tmp_path)
    evaluate(engine, tmp_path / "manifest.json", tmp_path / "cal", mode="calibrate")
    (tmp_path / "3.png").write_bytes(b"new-validation-image")
    called.clear()
    report = evaluate(engine, tmp_path / "manifest.json", tmp_path / "out", mode="evaluate",
                      calibration_path=tmp_path / "cal/calibration.json")
    assert report["status"] == "calibration-failed"
    assert not {"5.png", "6.png"} & set(called)


def test_failed_enrollment_is_not_dropped_from_known_denominator(tmp_path):
    engine, called = setup_experiment(tmp_path)
    # Add a registered identity whose enrollment cannot be processed.
    manifest = read_json(tmp_path / "manifest.json")
    for i in range(7, 12):
        (tmp_path / f"{i}.png").write_bytes(f"distinct-{i}".encode())
    manifest["enrollment"].append({"person_id": "q", "session": "phone", "images": ["7.png", "8.png", "9.png"]})
    manifest["validation"].append({"person_id": "q", "subject_id": "q", "session": "val", "image": "10.png"})
    manifest["test"].append({"person_id": "q", "subject_id": "q", "session": "test", "image": "11.png"})
    write_json(tmp_path / "manifest.json", manifest)
    original = engine.enroll
    engine.enroll = lambda images, person, ids: original() if person == "p" else {"status": "recapture_required", "templates": []}
    result = evaluate(engine, tmp_path / "manifest.json", tmp_path / "out")
    assert result["requested_people"] == 2
    assert result["enrolled_people"] == 1
    assert result["test"]["known_queries"] == 2
    assert result["test"]["known_identification_rate"] == .5

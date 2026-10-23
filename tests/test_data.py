import pytest

from face_lite.data import load_manifest, write_json


@pytest.fixture
def dataset(tmp_path):
    for i in range(7):
        (tmp_path / f"{i}.pgm").write_bytes(f"unique-image-content-{i}".encode())
    manifest = {"schema": 1, "scope": "phone-robot",
                "enrollment": [{"person_id": "p", "session": "phone", "images": ["0.pgm", "1.pgm", "2.pgm"]}],
                "validation": [{"person_id": "p", "subject_id": "p", "session": "robot-val", "image": "3.pgm"},
                               {"person_id": None, "subject_id": "q", "session": "val", "image": "4.pgm"}],
                "test": [{"person_id": "p", "subject_id": "p", "session": "robot-test", "image": "5.pgm"},
                         {"person_id": None, "subject_id": "r", "session": "test", "image": "6.pgm"}]}
    return tmp_path, manifest


def test_persisted_split_is_reproducible(dataset):
    root, manifest = dataset
    write_json(root / "manifest.json", manifest)
    assert load_manifest(root / "manifest.json")[1] == load_manifest(root / "manifest.json")[1]


@pytest.mark.parametrize("mutation", [
    lambda m: m["test"][0].update(image="3.pgm"),
    lambda m: m["test"][0].update(session="robot-val"),
    lambda m: m["test"][1].update(subject_id="q"),
    lambda m: m["test"][1].update(subject_id="p"),
    lambda m: m["test"][0].update(person_id="not-enrolled"),
    lambda m: m["test"][0].update(image="../outside.pgm"),
])
def test_leakage_and_label_errors_are_rejected(dataset, mutation):
    root, manifest = dataset
    mutation(manifest)
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError):
        load_manifest(root / "manifest.json")


def test_same_image_bytes_with_different_names_rejected(dataset):
    root, manifest = dataset
    (root / "5.pgm").write_bytes((root / "3.pgm").read_bytes())
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(root / "manifest.json")


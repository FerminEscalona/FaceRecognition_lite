from face_lite.artifacts import ASSETS, verify_asset
import pytest


def test_same_size_corruption_is_detected_before_inference(tmp_path):
    path = tmp_path / ASSETS["yunet"]["file"]
    path.write_bytes(b"\0" * ASSETS["yunet"]["bytes"])
    with pytest.raises(ValueError, match="integrity"):
        verify_asset(tmp_path, "yunet")

from concurrent.futures import Future
from threading import Event
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from face_lite import camera
from face_lite.supabase_gallery import IntegrationError


class Source:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def fetch(self, *args):
        self.calls += 1
        if self.fail:
            raise IntegrationError("network_error", "No se pudo consultar Supabase.")
        return f"new-gallery-{self.calls}", {"download_ms": 10}


class Engine:
    model_version = "v"
    dimension = 128
    config = NS(enrollment_min_similarity=.3)

    def __init__(self):
        self.inputs = []

    def identify(self, frame, gallery, policy):
        self.inputs.append((frame.copy(), gallery))
        assert np.all(frame < 200), "UI overlay must never enter recognition"
        return {"status": "identified", "person_id": "private-person", "similarity": .9, "elapsed_ms": 5}


class Capture:
    def __init__(self, opened=True, fail=False):
        self.opened, self.fail = opened, fail
        self.released = False
        self.frames = 0

    def isOpened(self):
        return self.opened

    def read(self):
        self.frames += 1
        return (False, None) if self.fail else (True, np.full((64, 64, 3), self.frames, dtype=np.uint8))

    def release(self):
        self.released = True


class UI:
    WINDOW_NORMAL = FONT_HERSHEY_SIMPLEX = LINE_AA = WND_PROP_VISIBLE = 0

    def __init__(self, keys=(27,), closed=False):
        self.keys = iter(keys)
        self.closed = closed
        self.destroyed = False

    def namedWindow(self, *args):
        pass

    def putText(self, frame, *args):
        frame[0, 0] = 255

    def imshow(self, *args):
        pass

    def waitKey(self, *args):
        return next(self.keys)

    def getWindowProperty(self, *args):
        return 0 if self.closed else 1

    def destroyAllWindows(self):
        self.destroyed = True


class ImmediatePool:
    def __init__(self, **kwargs):
        pass

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future

    def shutdown(self, **kwargs):
        pass


def policy():
    return NS(model_version="v", enrollment_min_similarity=.3)


def test_only_requested_photos_refresh_and_unmodified_next_frame(monkeypatch):
    monkeypatch.setattr(camera, "ThreadPoolExecutor", ImmediatePool)
    source, engine, cap = Source(), Engine(), Capture()
    ui, emitted = UI([32, -1, 32, -1, 27]), []
    camera.run_camera(engine, source, policy(), capture_factory=lambda _: cap, ui=ui, emit=emitted.append)
    assert source.calls == len(engine.inputs) == 2
    assert [g for _, g in engine.inputs] == ["new-gallery-1", "new-gallery-2"]
    assert [int(f[1, 1, 0]) for f, _ in engine.inputs] == [2, 4]
    assert cap.released and ui.destroyed
    assert all("person_id" not in event for event in emitted)
    assert all(event["calibration_status"] == "provisional" for event in emitted)


def test_refresh_failure_does_not_use_previous_gallery(monkeypatch):
    monkeypatch.setattr(camera, "ThreadPoolExecutor", ImmediatePool)
    source, engine, cap = Source(fail=True), Engine(), Capture()
    ui, emitted = UI([32, -1, 27]), []
    camera.run_camera(engine, source, policy(), capture_factory=lambda _: cap, ui=ui, emit=emitted.append)
    assert source.calls == 1 and not engine.inputs
    assert emitted[0]["code"] == "network_error"
    assert cap.released and ui.destroyed


def test_preview_alone_does_not_query_or_infer():
    source, engine, cap = Source(), Engine(), Capture()
    camera.run_camera(engine, source, policy(), capture_factory=lambda _: cap, ui=UI([-1, 27]))
    assert not source.calls and not engine.inputs and cap.released


@pytest.mark.parametrize("opened,fail,code", [(False, False, "camera_unavailable"), (True, True, "camera_disconnected")])
def test_camera_failures_release_device(opened, fail, code):
    cap, ui = Capture(opened, fail), UI()
    with pytest.raises(IntegrationError) as caught:
        camera.run_camera(Engine(), Source(), policy(), capture_factory=lambda _: cap, ui=ui)
    assert caught.value.code == code and cap.released and ui.destroyed


def test_window_close_releases_camera():
    cap, ui = Capture(), UI([-1], closed=True)
    camera.run_camera(Engine(), Source(), policy(), capture_factory=lambda _: cap, ui=ui)
    assert cap.released and ui.destroyed


def test_ui_failure_releases_camera():
    cap, ui = Capture(), UI()
    def fail(*args):
        raise cv2.error("GUI unavailable")
    ui.namedWindow = fail
    with pytest.raises(IntegrationError) as caught:
        camera.run_camera(Engine(), Source(), policy(), capture_factory=lambda _: cap, ui=ui)
    assert caught.value.code == "camera_ui_error" and cap.released


def test_incompatible_calibration_before_opening_camera():
    other = NS(model_version="other")
    def forbidden(_):
        pytest.fail("Camera must not open")
    with pytest.raises(IntegrationError) as caught:
        camera.run_camera(Engine(), Source(), other, capture_factory=forbidden)
    assert caught.value.code == "incompatible_calibration"


def test_busy_download_drops_repeated_requests_and_finishes():
    entered, release = Event(), Event()
    class BlockingSource(Source):
        def fetch(self, *args):
            entered.set()
            assert release.wait(2)
            return super().fetch(*args)
    source = BlockingSource()
    request = camera.CaptureRequest(source, Engine())
    try:
        assert request.request() and entered.wait(2)
        assert not request.request() and request.poll() is None
        release.set()
        request.future.result(timeout=2)
        assert request.poll()[0] == "new-gallery-1"
        assert source.calls == 1 and request.future is None
    finally:
        release.set()
        request.close()

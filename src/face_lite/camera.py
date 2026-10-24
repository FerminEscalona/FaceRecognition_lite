"""Mac camera demo: continuous preview, one inference per requested photo."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from .supabase_gallery import IntegrationError


class CaptureRequest:
    """At most one download, no request queue and no retained gallery fallback."""

    def __init__(self, source, engine):
        self.source = source
        self.engine = engine
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gallery-download")
        self.future = None
        self.started = None

    def request(self) -> bool:
        if self.future is not None:
            return False
        self.started = time.perf_counter()
        self.future = self.executor.submit(self.source.fetch, self.engine.model_version,
                                           self.engine.dimension, self.engine.config.enrollment_min_similarity)
        return True

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        return (*future.result(), self.started)

    def close(self):
        # No new requests; the HTTP timeout bounds any already running download.
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.future = None


def _open_camera(index: int):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    return cv2.VideoCapture(index, backend)


def _emit(report: dict):
    # The UI may display the identity; terminal diagnostics omit personal identifiers.
    print(json.dumps(report, ensure_ascii=False), flush=True)


def run_camera(engine, source, policy, *, camera_index: int = 0,
               capture_factory=None, ui=cv2, emit=_emit):
    if camera_index < 0:
        raise IntegrationError("invalid_camera", "El índice de cámara debe ser no negativo.")
    if (policy.model_version != engine.model_version
            or policy.enrollment_min_similarity != engine.config.enrollment_min_similarity):
        raise IntegrationError("incompatible_calibration", "La calibración no corresponde al modelo y procesamiento locales.")
    capture = None
    requests = CaptureRequest(source, engine)
    window = "Face Lite - DEMO PROVISIONAL"
    lines = ["SPACE: refresh gallery + photo | ESC: exit", "Calibration provisional; accuracy not validated"]
    try:
        capture = (capture_factory or _open_camera)(camera_index)
        if not capture.isOpened():
            raise IntegrationError("camera_unavailable", "No se pudo abrir la cámara. Revisa permisos de macOS, índice y aplicaciones que la usan.")
        ui.namedWindow(window, ui.WINDOW_NORMAL)
        while True:
            snapshot = None
            try:
                snapshot = requests.poll()
            except IntegrationError as error:
                lines = [f"ERROR: {error.code} - press SPACE to retry", "No recognition performed"]
                emit(error.report())
            # Preview drains the capture stream while the network worker runs.
            # After successful refresh, read the next frame, never a stored preview.
            capture_started = time.perf_counter()
            ok, frame = capture.read()
            capture_ms = (time.perf_counter() - capture_started) * 1000
            if not ok or frame is None or not frame.size:
                raise IntegrationError("camera_disconnected", "No se pudo obtener una imagen de la cámara.")
            if snapshot is not None:
                gallery, report, started = snapshot
                result = engine.identify(frame, gallery, policy)
                total_ms = (time.perf_counter() - started) * 1000
                similarity = result.get("similarity")
                lines = [f"Result: {result['status']}", f"person_id: {result.get('person_id') or '-'}",
                         f"Cosine: {similarity:.4f}" if similarity is not None else "Cosine: -",
                         f"Network+validation: {report['download_ms']:.0f} ms | Capture: {capture_ms:.0f} ms",
                         f"Identification: {result['elapsed_ms']:.0f} ms | Total: {total_ms:.0f} ms"]
                emit({"status": result["status"], "calibration_status": "provisional",
                      "download_ms": report["download_ms"], "capture_ms": capture_ms,
                      "identification_ms": result["elapsed_ms"], "total_ms": total_ms})
                del snapshot, gallery, report, result
            # Render only on a copy. The original BGR frame is the model input.
            preview = frame.copy()
            for i, text in enumerate(["DEMO PROVISIONAL | SPACE: photo | ESC: exit", *lines]):
                ui.putText(preview, text, (12, 28 + i * 28), ui.FONT_HERSHEY_SIMPLEX,
                           .55, (0, 255, 255), 1, ui.LINE_AA)
            ui.imshow(window, preview)
            key = ui.waitKey(10) & 0xFF
            if key == 27:
                break
            try:
                if ui.getWindowProperty(window, ui.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                # Some GUI backends destroy the window before this property read.
                break
            if key == 32 and requests.request():
                lines = ["Refreshing Supabase...", "Photo will be captured after successful validation"]
    except cv2.error:
        raise IntegrationError("camera_ui_error", "OpenCV no pudo usar la cámara o mostrar la ventana; verifica instalación con GUI y permisos.") from None
    finally:
        # Release the physical device before waiting on a pending HTTP request.
        if capture is not None:
            capture.release()
        try:
            ui.destroyAllWindows()
        except cv2.error:
            pass
        requests.close()

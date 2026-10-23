# Fuentes, licencias y atribución

## Modelos

Origen: [OpenCV Zoo](https://github.com/opencv/opencv_zoo), revisión `48b10fa505aeb51fbd127fc60a5c660040e823b7`. Los hashes y tamaños exactos se fijan en `src/face_lite/artifacts.py` y se verifican al descargar y al iniciar la inferencia.

| Artefacto | Licencia publicada por OpenCV Zoo | Origen |
| --- | --- | --- |
| YuNet 2023mar | MIT | [Detector y licencia](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet) |
| SFace 2021dec FP32 e INT8 | Apache-2.0 en su directorio | [Extractor y licencia](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface) |

SFace es trabajo de Yaoyao Zhong y colaboradores, con conversión ONNX atribuida en OpenCV Zoo a Chengrui Wang. YuNet procede de los autores de libfacedetection. La fuente de SFace documenta que los archivos contienen instancias MobileFaceNet entrenadas con la función de pérdida SFace. Las licencias originales se descargan junto a los modelos; deben acompañar a una eventual redistribución de los pesos y no se sustituyen por la licencia de este proyecto.

Esta documentación registra la licencia que publica el distribuidor; no afirma auditoría del origen o derechos de todas las imágenes de preentrenamiento ni ofrece permisos adicionales para otros usos. La descarga explícita obtiene los pesos upstream sin incorporarlos al historial Git.

## Datos públicos

AT&T Laboratories Cambridge, **The Database of Faces (ORL)**, [archivo oficial alojado por Cambridge](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html). La página solicita reconocer a AT&T Laboratories Cambridge al usar las imágenes. No se atribuye una licencia abierta genérica ni se redistribuyen las fotos en este proyecto.

El ZIP obtenido de la fuente oficial tiene SHA-256 `509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec`; el preparador verifica ese contenido antes de extraerlo. Se usa para pruebas funcionales locales con atribución. Cualquier uso posterior debe revisar las condiciones aplicables; la presencia pública del archivo no resuelve por sí sola otros derechos.

## Software

Las dependencias de ejecución y herramientas están fijadas en `pyproject.toml` y `requirements-dev.lock`. Cada distribución conserva su licencia propia: OpenCV, NumPy, Pillow, psutil, pytest y las herramientas de construcción. Los pesos, datos, galerías privadas y calibraciones experimentales no se incorporan al wheel.

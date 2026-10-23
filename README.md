# FaceRecognition Lite

Componente Python de reconocimiento facial para una demostración de interacción con el robot X2 Ultra. Una persona se registra desde el teléfono con **tres fotos**; la aplicación usa este componente para generar sus embeddings. El robot genera otro embedding desde su cámara y compara contra las plantillas registradas.

**Estado: prototipo local.** Está implementado con modelos preentrenados y probado en Mac M4 Pro con AT&T/ORL. La precisión objetivo y la aceptación teléfono–robot siguen pendientes. Ver [resultados locales](reports/resultado-local.md) y [plan de implementación](plans/01-modelo-reconocimiento-facial.md).

## Entendimiento del negocio y alcance

El robot debe identificar al menos a cinco personas en interiores, una por imagen, y obtener `person_id` o desconocido en hasta dos segundos, incluyendo la captura física. La identidad permite a la integración externa consultar nombre, categoría y destino para personalizar una interacción.

Se prioriza evitar confundir personas. Si la evidencia es insuficiente o ambigua, el resultado no contiene identidad. «Liviano» significa consumir recursos de forma controlada mientras el robot hace otras tareas; ejecutar en ESP32 no es requisito.

Este repositorio entrega detección, alineación, embeddings, registro de tres plantillas, comparación, calibración, evaluación y medición. La web, backend HTTP, Supabase, sincronización, captura física, control del robot y decisiones de acceso pertenecen a la integración externa. El backend aún está por definir; la primera entrega es Python, no un modelo para navegador.

```text
Teléfono: frontal + giro izquierdo + giro derecho
    → backend con nuestro componente → tres embeddings → Supabase
Robot: imagen de su cámara
    → mismo componente → embedding → galería local → person_id / desconocido
    → integración externa consulta persons y responde
```

Agregar personas genera plantillas; no reentrena la red. Las fotos se procesan antes de guardar los embeddings. El robot recibe los embeddings de Supabase, no necesita descargar las fotos. Las tres capturas deben ser válidas, diferentes y consistentes; el registro devuelve posiciones para recapturar y no publica filas parciales.

## Tablas acordadas con la aplicación

| Tabla | Campos |
| --- | --- |
| `persons` | `person_id`, `first_name`, `last_name`, `category`, `destination` |
| `face_templates` | `template_id`, `person_id`, `embedding`, `model_version` |

Son tres filas de `face_templates` por persona. El embedding es una lista JSON de 128 números finitos normalizados L2. Los identificadores los suministra la aplicación. `category` y `destination` no son entradas del reconocimiento.

La integración descarga una lista de filas y la convierte en `Gallery`. Se rechazan galerías incompletas o versiones mezcladas. Cambiar modelo o procesamiento requiere regenerar plantillas y recalibrar; los embeddings no permiten recuperar las fotos. Los detalles están en el [contrato de integración](docs/contrato-integracion.md).

## Instalación reproducible

Requiere Python 3.11 y CPU. En el Mac de desarrollo existe `/opt/homebrew/bin/python3.11`. En otros equipos usar su instalación de Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-build-isolation --no-deps -e .
face-lite download-models
face-lite doctor --output outputs/doctor.json
```

La descarga es explícita y requiere internet. La inferencia posterior funciona sin internet ni Supabase. Los tres artefactos se comprueban contra los SHA-256 del repositorio oficial, fijados en `face_lite.artifacts`. Sus licencias se descargan junto a los pesos. Los modelos se mantienen en `artifacts/models/`, fuera de Git.

La referencia usa OpenCV 4.12.0, YuNet 2023mar y SFace 2021dec ONNX. La variante `int8` tiene versión propia. El manifiesto exporta revisión, hashes, dimensión, procesamiento y configuración. No se afirma que un archivo INT8 tenga aceleración hardware o menor uso de CPU sin medirlo.

## Experimento público reproducible

AT&T/ORL sirve como comprobación funcional; sus imágenes pequeñas en escala de grises no representan las cámaras del proyecto. Ver [protocolo de datos](docs/protocolo-datos.md).

```bash
face-lite prepare-orl
face-lite experiment --manifest data/orl/manifest.json --output outputs/orl-fp32
face-lite experiment --manifest data/orl/manifest.json --variant int8 --output outputs/orl-int8
```

Cada experimento produce `gallery.json`, `model-manifest.json`, `calibration.json` y `report.json` en su directorio. El reporte incluye errores de registro y calidad; no se ocultan rechazos para subir la precisión. Las galerías contienen datos biométricos y permanecen en `outputs/`, ignorado por Git.

Para reproducir descarga, preparación, ambas variantes y benchmarks secuenciales con un resumen agregado, ejecutar `python scripts/reproduce_local.py`. El resumen queda en `outputs/local-summary.json`, sin fotos ni embeddings.

También se pueden separar calibración y prueba:

```bash
face-lite calibrate --manifest data/orl/manifest.json --output outputs/calibration-only
face-lite evaluate --manifest data/orl/manifest.json --calibration outputs/calibration-only/calibration.json --output outputs/frozen-test
```

El segundo comando no busca nuevos parámetros y comprueba que la calibración corresponde al modelo y al corpus de validación. `experiment` hace las dos etapas en orden, con la política congelada antes de procesar prueba. Una configuración que rechaza a todos no cumple el objetivo de calidad.

## Registro e identificación con fotos propias

```bash
face-lite enroll --images front.jpg left.jpg right.jpg --person-id p01 --template-ids t01 t02 t03 --output outputs/enrollment.json
```

Si el resultado es `enrolled`, la aplicación inserta las filas de `templates` en Supabase. El archivo completo es una respuesta de registro; para `--gallery` se usa exclusivamente la lista de filas, agregando los registros completos de las personas deseadas.

La calibración de ORL es solo experimental. Para fotos propias crear un manifiesto con `scope: phone-robot`, sesiones independientes de registro, validación y prueba, y ejecutar el mismo protocolo.

```bash
face-lite identify --image robot-query.jpg --gallery outputs/gallery.json --calibration outputs/own/calibration.json --output outputs/identification.json
```

`identified` devuelve identidad; `unknown`, `ambiguous`, `no_face`, `multiple_faces`, `insufficient_quality`, `invalid_image` e `incompatible_gallery` no. La similitud no es una probabilidad. Una galería con estructura inválida falla al cargarla y debe corregirse antes de invocarla.

Para integración reutilizar `FaceEngine`, `Gallery` y `Calibration`; consultar los ejemplos del contrato. Las rutas o bytes aplican EXIF. Los arrays deben ser BGR `uint8` ya orientados. El modelo se carga una vez por worker y se reutiliza; no construirlo por foto.

## Comparación y consumo

Ejecutar los benchmarks secuencialmente en procesos distintos, sin otras mediciones simultáneas:

```bash
face-lite benchmark --manifest data/orl/manifest.json --gallery outputs/orl-fp32/gallery.json --calibration outputs/orl-fp32/calibration.json --repeats 200 --output outputs/orl-fp32/benchmark.json
face-lite benchmark --manifest data/orl/manifest.json --variant int8 --gallery outputs/orl-int8/gallery.json --calibration outputs/orl-int8/calibration.json --repeats 200 --output outputs/orl-int8/benchmark.json
face-lite compare --baseline outputs/orl-fp32 --candidate outputs/orl-int8 --output outputs/comparison.json
```

Se miden arranque, latencia p50/p95/máxima, etapas, consultas >2 s, CPU y memoria RSS del proceso. El benchmark incluye lectura de archivo e inferencia; no captura física ni red. Mantener igual galería, configuración y orden en ambas variantes. La comparación no promueve automáticamente un modelo.

El parámetro `--config configs/pipeline.json` permite cambiar límites provisionales de calidad y consistencia, confianza del detector e hilos. Los cambios semánticos alteran `model_version`. Ajustar límites solo con validación y documentar el nuevo experimento; no modificar el umbral mirando prueba.

## Precisión y aceptación

Objetivo de ingeniería: ≥95 % de identificación correcta en conocidos y cero asignaciones incorrectas observadas, con cinco personas y muestras independientes. La aceptación real exige al menos veinte consultas por conocido y cien consultas de al menos cinco desconocidos, además de validación separada.

El comparador promedia las dos mejores similitudes de las tres plantillas por persona. La aceptación requiere umbral y separación entre identidades, calibrados conjuntamente sobre validación. Los empates se rechazan.

El reporte muestra denominadores completos y métricas solo sobre entradas procesables. Los intervalos de incertidumbre suponen consultas independientes. Cero errores en un ensayo no garantiza riesgo cero. Los datos ORL y los resultados de Mac no validan la distancia, iluminación ni rendimiento del X2 Ultra.

La referencia preentrenada permite empezar sin entrenar. Transfer learning será una fase posterior si los errores y datos adicionales lo justifican; las tres fotos por usuario son plantillas de registro, no un conjunto de entrenamiento suficiente. Si una versión no cumple el objetivo se reporta como tal.

## Pruebas y distribución

```bash
python -m pytest -q
python -m build --no-isolation
```

Las pruebas de integración requieren los artefactos descargados y ORL; si faltan, se marcan como omitidas. Las pruebas unitarias cubren contratos, desconocidos, ambigüedad, registro atómico, datos inválidos y fugas entre particiones. La distribución contiene código; pesos y calibración se entregan por separado con su manifiesto. No se incluye HTTP ni acceso a la base de datos.

## Datos y límites pendientes

Fotos y embeddings son biométricos: usar capturas autorizadas, acceso restringido y eliminación conforme a la política del proyecto. Mantenerlos fuera del repositorio y logs. Las fuentes y licencias están documentadas en [THIRD_PARTY.md](THIRD_PARTY.md).

Pendiente: acceso al X2 Ultra, capturas de teléfono y robot en sesiones distintas, distancia operativa, evaluación de calidad real y presupuesto de recursos mientras funcionan voz/navegación. La integración debe gestionar sincronización y revocaciones. El prototipo no verifica presencia real ni es una solución de autenticación o autorización de acceso.

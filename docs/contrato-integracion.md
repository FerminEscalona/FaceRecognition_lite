# Contrato del componente Python

La aplicación web y el robot instalan el mismo paquete y descargan los mismos artefactos. El backend web está por definir; no se ofrece HTTP ni código de navegador en esta versión.

## Registro

```python
from face_lite import FaceEngine

engine = FaceEngine("artifacts/models")  # Una instancia persistente por worker.
result = engine.enroll(
    ["front.jpg", "left.jpg", "right.jpg"],
    person_id="identificador-creado-por-la-app",
    template_ids=["id-frontal", "id-izquierda", "id-derecha"],
)
if result["status"] == "enrolled":
    rows_for_supabase = result["templates"]
else:
    positions_to_repeat = result["captures"]
```

Las imágenes pueden ser rutas, bytes codificados o arrays NumPy `uint8` BGR HWC. Las rutas y los bytes corrigen EXIF y se convierten a BGR; los arrays ya deben estar orientados. JPEG, PNG y PGM son los formatos ejercitados. El límite de entrada es 20 MB y 20 megapíxeles; el backend debe convertir fotos mayores o formatos no soportados sin alterar el contrato.

Cada fila tiene exactamente `template_id`, `person_id`, `embedding` (128 números) y `model_version`. Los identificadores se tratan como cadenas no vacías; pueden ser UUID serializados. La app inserta las tres filas de forma atómica. Un resultado parcial devuelve cero filas y la posición que requiere repetición. El paquete no controla transacciones de Supabase ni modifica `persons`.

`engine.template(image, person_id, template_id)` permite producir una plantilla individual. Esta operación no confirma la consistencia del registro completo; antes de publicar deben validarse las tres capturas con `enroll`. No completar registros duplicando imágenes o embeddings.

## Identificación

```python
import json
from face_lite import Calibration, Gallery

# Filas ya descargadas por la integración externa, sin convertir nombres en embeddings.
rows = json.load(open("gallery.json"))
gallery = Gallery(rows, engine.model_version, engine.dimension)
policy = Calibration(**json.load(open("calibration.json")))
result = engine.identify("robot-query.jpg", gallery, policy)
```

Solo `status == "identified"` lleva un `person_id` utilizable para consultar `persons`. Los demás estados devuelven `person_id: null`. `similarity` es una puntuación coseno, no probabilidad. `gap` compara identidades distintas, no plantillas de una misma persona. Con una sola identidad no hay segundo candidato; siguen aplicando el umbral y el resto de las comprobaciones.

Estados: `identified`, `unknown`, `ambiguous`, `no_face`, `multiple_faces`, `insufficient_quality`, `invalid_image`, `incompatible_gallery`. Los registros también pueden devolver `recapture_required`, `duplicate_capture` o `inconsistent_enrollment`. No se selecciona silenciosamente la cara más grande.

`Gallery` valida la instantánea al cargarla y lanza `ValueError` para filas inválidas. La integración debe mantener la última instantánea válida y gestionar la actualización de forma atómica. `identify` rechaza calibraciones o galerías de otra versión. La consistencia y dimensión se comprueban antes de comparar.

## Versiones y concurrencia

`model_version` deriva de los hashes de modelos, preprocesamiento, versión de OpenCV y configuración semántica; los hilos no la cambian. La versión FP32 no se mezcla con INT8. Los manifiestos de diagnóstico y cada experimento incluyen los hashes y parámetros usados. `calibration_id` identifica umbral, margen y procedencia de validación.

Mantener la misma configuración en registro y robot. Tras una actualización, regenerar las tres plantillas por persona y distribuir la nueva calibración antes de activarla; los embeddings anteriores no permiten reconstruir las fotos.

La instancia serializa el acceso al estado nativo del modelo. Reutilizar una instancia persistente por worker. OpenCV configura los hilos globalmente en el proceso: todas las instancias en un mismo proceso deben compartir esa configuración y no cambiarla durante inferencia. El integrador controla la frecuencia y descarta fotogramas atrasados, sin acumular una cola ilimitada.

El motor no captura imágenes ni consulta bases de datos. La demo opcional `face-lite camera` integra captura en el Mac y el adaptador `SupabaseGallerySource`; no incluye captura mediante SDK del robot ni confirmación temporal. Los dos segundos se verifican externamente incluyendo captura. Las mediciones originales del modelo incluyen decodificación de archivos e inferencia, sin tiempos de red o consulta de datos personales.

## Adaptador de Supabase para la demo local

```python
from face_lite.supabase_gallery import SupabaseSettings, SupabaseGallerySource

with SupabaseGallerySource(SupabaseSettings.load()) as source:
    gallery, diagnostics = source.fetch(
        engine.model_version, engine.dimension, engine.config.enrollment_min_similarity
    )
    # Capturar una imagen reciente después de esta descarga y usar engine.identify.
```

El adaptador solo autentica y lee `face_templates`; no crea personas ni publica plantillas. Conserva la sesión en memoria y la renueva antes de caducar. Acepta embeddings como listas numéricas o texto JSON de vector; no altera valores ni etiquetas de modelo. Una lectura fallida genera `IntegrationError` con código, mensaje seguro y diagnóstico agregado, sin credenciales, identificadores personales ni vectores.

Los comandos `supabase-check` y `camera` cargan `.env` desde la ruta explícita (predeterminado: directorio actual) y respetan variables del entorno. Los permisos del usuario autenticado determinan qué filas son visibles. No se asume acceso a toda la tabla. La demo rechaza una galería vacía, inválida o incompatible y cancela ese intento sin usar datos de otro momento.

La ventana muestra `person_id` cuando corresponde; sus diagnósticos de terminal omiten el identificador. No descarga `persons` ni implementa la interacción de negocio. Los comandos existentes con archivos JSON conservan su comportamiento.

# Plan del componente de reconocimiento facial

Estado: prototipo local implementado. La precisión objetivo aún no se alcanza en el ensayo público; la aceptación con capturas propias y el robot permanece pendiente. Consultar los resultados en `reports/resultado-local.md`.

## 1. Objetivo y alcance

Desarrollar un componente Python reutilizable que genere plantillas faciales desde las imágenes del teléfono e identifique personas desde imágenes del robot X2 Ultra, utilizando el mismo modelo y preprocesamiento.

Escenario inicial:

- Al menos cinco personas registradas.
- Tres imágenes por persona: frontal, giro suave a la izquierda y giro suave a la derecha.
- Uso en interiores, con una persona frente a la cámara.
- Identificación contra la galería completa, con resultado `person_id` o desconocido.
- Prioridad de evitar identidades incorrectas, aunque sea necesario repetir una captura.
- Hasta dos segundos desde la captura del robot hasta el resultado de identificación.
- Demostración de interacción: saludo e información de categoría o destino.

Incluido: detección, alineación, extracción de embeddings, comparación, calibración, evaluación, empaquetado y medición del consumo.

Excluido: aplicación web, servidor HTTP, conexión a Supabase, migraciones, sincronización, control del robot, seguimiento continuo, reconocimiento de grupos, autorización de acceso y detección de ataques con fotos o videos. El ESP32 no es un requisito de ejecución.

## 2. Arquitectura y contratos

```text
Registro:
Tres fotos → validación → detección y alineación
          → extractor facial → tres plantillas

Reconocimiento:
Imagen del robot → validación → detección y alineación
                 → mismo extractor → embedding
                 → comparación con plantillas → identidad o desconocido
```

El equipo web invocará el componente desde un backend cuya tecnología todavía no está definida. La entrega inicial será un paquete Python y una interfaz de comandos; la integración del backend corresponde al otro equipo.

### Modelo inicial y ejecución

- Python 3.11 en un entorno aislado compatible con el Mac M4 Pro.
- OpenCV 4.x y NumPy, YuNet para detección y SFace para embeddings.
- Artefactos ONNX `face_detection_yunet_2023mar.onnx` y `face_recognition_sface_2021dec.onnx`.
- Dependencias fijadas, origen, revisión y SHA-256 de los artefactos registrados.
- CPU inicial; modelos cargados una vez por instancia persistente de trabajo. Dos hilos de inferencia configurables. Se reutiliza la instancia entre solicitudes.
- Registro y consulta comparten orientación, orden de canales, alineación mediante puntos faciales y normalización L2.
- Modelos y datos descargados fuera del historial Git.

La selección se apoya en el [flujo oficial de OpenCV](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html). Es una referencia técnica inicial, no una garantía de precisión en el robot.

### Contrato de datos

| Tabla | Campos |
| --- | --- |
| `persons` | `person_id`, `first_name`, `last_name`, `category`, `destination` |
| `face_templates` | `template_id`, `person_id`, `embedding`, `model_version` |

Un registro completo genera tres filas de `face_templates`, con el mismo `person_id`. Los identificadores son suministrados por el integrador; el componente no crea personas.

- Embedding: vector finito, normalizado L2, serializable como lista JSON, con dimensión verificada contra el manifiesto. La referencia SFace utiliza 128 elementos.
- `model_version`: identificador inmutable de pesos, detector y procesamiento. El manifiesto documenta además las versiones de dependencias y políticas de calidad.
- Manifiesto y configuración de calibración distribuidos con el paquete, sin agregar tablas.
- Nombre, categoría y destino no intervienen en la comparación.
- Galería recibida como lista JSON equivalente a las filas de Supabase. La conexión a la base es externa.

Operaciones públicas: generar plantilla, validar registro de tres imágenes e identificar contra una galería.

La respuesta incluye estado, `person_id` nullable, similitud cuando exista, versión y tiempo. Se distinguen identificado, desconocido, ambiguo, sin rostro, múltiples rostros, calidad insuficiente, imagen inválida y galería incompatible. Un error no produce identidad.

### Política de comparación

- Similitud coseno contra las tres plantillas de cada persona.
- Puntuación por persona: promedio de las dos mejores similitudes.
- Aceptar solo si supera el umbral calibrado y la separación mínima respecto a la segunda identidad.
- Calibración conjunta de umbral y separación; rechazo de empates.
- Similitud no equivale a porcentaje de certeza.
- Rechazo de versiones incompatibles, vectores inválidos y registros incompletos. Galería vacía sin identidad.
- Comprobar consistencia entre las tres plantillas; estas no prueban por sí mismas que correspondan a una sola persona.

## 3. Etapas y entregables

### A. Base reproducible y datos

Preparar entorno, descarga verificable, manifiesto y diagnóstico.

Utilizar inicialmente AT&T/ORL para pruebas funcionales, con atribución y condiciones de su [fuente oficial](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html). Imágenes pequeñas en escala de grises no validan teléfono–robot. La fuente solicita atribución; no se inventará una licencia comercial o un permiso de redistribución de las fotos.

Persistir particiones de registro, validación y prueba. Separar desconocidos entre validación y prueba. Contabilizar fallos de detección. No distribuir imágenes ni embeddings como parte del paquete.

Entregable: protocolo y experimento reproducible, etiquetado como preliminar.

### B. Registro e inferencia

Implementar procesamiento común, tres plantillas y comparación local. Validar integridad de imagen, cara única, tamaño, nitidez e iluminación. Los límites son configurables, provisionales y deben evaluarse con datos de validación.

Una captura rechazada devuelve su posición y motivo. No se reemplazan capturas por duplicados. Se publican las tres filas solo al completar un registro consistente.

Entregable: paquete y comandos locales sin dependencia de red durante inferencia.

### C. Calibración y evaluación

Evaluar identificación de conocidos y rechazo de desconocidos con la regla final de puntuación. Elegir umbral y separación en validación, priorizando cero asignaciones incorrectas observadas y después la mayor identificación correcta. Congelar antes de prueba.

Si no se cumplen los objetivos, reportarlo sin ajustar sobre prueba. Entregar configuración versionada, denominadores, errores por persona y limitaciones.

### D. Optimización

Medir tamaño, memoria máxima y latencia por etapa. Comparar la referencia con la [variante INT8 oficial de SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).

La variante optimizada solo es elegible si mantiene calidad y mejora recursos o latencia. Tendrá versión, galería regenerada y calibración propias. No se mezclan embeddings entre variantes.

Transfer learning no es obligatorio para esta primera entrega: las tres capturas son registro, no un conjunto suficiente para ajustar pesos. Si no se alcanza la precisión, documentar los errores y abrir una fase de ajuste con datos independientes. No declarar terminado un modelo que no cumpla la aceptación.

### E. Entrega y validación en robot

Entregar paquete, modelos descargables, manifiesto, calibración, ejemplos y reporte. Actualizar README con dos tablas, tres plantillas, cinco personas, dos segundos y backend pendiente.

Al disponer del robot, comprobar compatibilidad de plantillas y ejecutar con sus otras tareas activas. Separar resultados Mac y X2 Ultra. Estado de prototipo local hasta completar esa validación.

## 4. Pruebas y aceptación

### Componente

- JSON de plantillas sin pérdida relevante de similitud.
- Rechazo de dimensión, versión y valores inválidos.
- Imágenes corruptas, sin cara, con varias caras y de calidad insuficiente.
- Galería vacía, desconocidos y ambigüedad.
- Tres plantillas vinculadas correctamente y registro atómico.
- Consistencia entre registro y consulta.
- Ejecución repetida sin crecimiento sostenido de memoria.
- Uso sin internet o Supabase cuando los artefactos y la galería estén disponibles.

### Calidad real

- Al menos cinco personas.
- Al menos 95 % de identificación correcta en consultas conocidas.
- Cero asignaciones incorrectas observadas en la aceptación.
- Contabilizar fallos de procesamiento y rechazos en el denominador completo, además de métricas del comparador.

Muestra mínima: veinte consultas independientes por cada una de cinco personas conocidas y cien consultas distribuidas entre al menos cinco desconocidos. Datos adicionales separados para calibración; fotogramas contiguos no constituyen ensayos independientes.

Registro desde teléfono y consultas del robot en sesiones distintas. Reportar incertidumbre: cero errores observados no equivale a riesgo cero. El conjunto ORL no satisface la validación real ni sus garantías de sesión/cámara.

### Rendimiento

Medir captura y procesamiento completo; separar consultas de negocio. Reportar mediana, p95, máximo y proporción sobre dos segundos. Ninguna consulta del ensayo de aceptación debe exceder dos segundos.

Medir arranque en frío separado de operación normal con modelos cargados. Comparar recursos y respuesta del robot con y sin reconocimiento. El presupuesto de CPU/GPU y los límites de impacto se fijarán al acceder al equipo; el bajo impacto permanece pendiente hasta entonces.

## 5. Dependencias y límites

- Desarrollo local posible sin Supabase, capturas propias o robot.
- Aceptación final dependiente del robot y capturas reales.
- Una persona por imagen, interiores; distancia por documentar con evidencia.
- Contrato inicial Python en backend. Navegador requeriría otro despliegue.
- Cambio de extractor o procesamiento: nueva versión, recalibración y regeneración.
- Datos biométricos fuera de Git y logs. Solo métricas agregadas en reportes publicados.
- Se fija cómo demostrar la precisión; no se garantiza antes de medirla.

# Verificación de Supabase y cámara del MacBook

Fecha: 2026-10-23. Entorno: Mac de desarrollo, Python 3.11, OpenCV 4.12.0.88, referencia FP32. Demostración con calibración pública provisional, no aceptación de precisión teléfono–robot.

## Conexión y contrato

Se autenticó el usuario configurado y se consultó `face_templates` con la clave publicable, bajo sus permisos. No se modificaron tablas, políticas ni registros.

- **3 filas visibles, correspondientes a 1 persona.** Esto no demuestra acceso a todas las personas existentes en la base.
- Tres vectores de **128 dimensiones**, finitos, normalizados, distintos y consistentes.
- Versión de las tres plantillas: `sface-fp32-f6ae35223478f49a5818`, compatible exactamente con el componente local.
- Comprobación final: 1.407 ms para autenticación, consulta y validación; aproximadamente 881 ms de autenticación, 525 ms de consulta y 0,74 ms de validación. Es una observación, no un benchmark.

No se incluyen credenciales, identificadores personales, tokens ni embeddings en este reporte. Los diagnósticos locales agregados están en `outputs/supabase-check.json`, fuera de Git.

## Captura real

La primera apertura solicitó permiso de cámara a macOS y no pudo inicializarla. En el segundo intento se abrió la cámara, se mostró la vista previa, se descargó la galería, se capturó un fotograma y se ejecutó una sola identificación. La ventana y el dispositivo se liberaron al terminar.

El ensayo breve disparó una única solicitud automáticamente para comprobar el recorrido; en el comando de uso normal, **Espacio** solicita cada foto y **Escape** termina. No se guardó la foto, el embedding de consulta ni la galería.

| Medición de una captura | Tiempo observado |
| --- | ---: |
| Autenticación, consulta y validación | 1.064 ms |
| Lectura del fotograma después de la actualización | 18,1 ms |
| Identificación local completa | 30,5 ms |
| Solicitud a resultado, incluida coordinación de interfaz | 1.119 ms |

El componente devolvió `identified`. **No se verificó externamente que la identidad devuelta fuera correcta.** Un ensayo con una sola persona visible y una calibración pública no mide precisión, rechazo de desconocidos ni comportamiento con cinco personas. Estas cifras tampoco prueban latencia máxima ni rendimiento del X2 Ultra.

Los diagnósticos de este ensayo están en `outputs/camera-smoke.json`, fuera de Git, sin imágenes, vectores ni identificadores personales.

## Pruebas reproducibles y límites

Las pruebas automatizadas usan clientes y cámara simulados; no leen `.env`, no consultan Supabase ni abren el dispositivo. Cubren errores sin secretos, paginación, renovación, validación estricta, actualización por foto, ausencia de inferencia tras un error, fotogramas sin textos y liberación de recursos. Se conserva la regresión del motor original.

Para repetir manualmente:

```bash
source .venv/bin/activate
face-lite supabase-check
face-lite camera --camera 0 --calibration outputs/orl-fp32/calibration.json
```

Pendiente: al menos cinco personas visibles con tres plantillas cada una, calibración específica con sesiones independientes y prueba de conocidos/desconocidos. La demo solo consume `face_templates` y muestra `person_id`; la consulta de información de negocio en `persons` pertenece a la integración de la aplicación.

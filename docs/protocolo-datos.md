# Protocolo de datos y evaluación

## Preparación pública

`face-lite prepare-orl` descarga AT&T/ORL desde Cambridge, comprueba el SHA-256 y extrae solo los 400 PGM esperados. Se reconoce a AT&T Laboratories Cambridge, según su [fuente oficial](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html). Se conservan imágenes, archivo original, hash y manifiesto dentro de `data/`, ignorado por Git. La disponibilidad pública no se interpreta como permiso comercial o de redistribución irrestricto.

Partición fija:

| Uso | Personas | Imágenes |
| --- | --- | --- |
| Registro | s1–s5 | 1, 2, 3 |
| Conocidos de validación | s1–s5 | 4, 5, 6 |
| Conocidos de prueba | s1–s5 | 7, 8, 9, 10 |
| Desconocidos de validación | s6–s10 | 1–10 |
| Desconocidos de prueba | s11–s20 | 1–10 |

No se usan las imágenes restantes. Las posiciones front/left/right de ORL son etiquetas de los tres archivos de referencia, no ángulos comprobados. Las sesiones del manifiesto público son etiquetas de partición, no sesiones reales verificadas. No se presenta este conjunto como entrenamiento ni validación teléfono–robot. El posible solapamiento con datos originales de preentrenamiento del extractor no puede descartarse.

## Datos propios

Crear un directorio privado con un manifiesto JSON y rutas relativas. Usar identificadores seudónimos. Ejemplo estructural (debe ampliarse con todos los participantes y muestras):

```json
{
  "schema": 1,
  "scope": "phone-robot",
  "source": "Capturas autorizadas del proyecto",
  "enrollment": [
    {"person_id": "p01", "session": "phone-day1", "images": ["p01/front.jpg", "p01/left.jpg", "p01/right.jpg"]}
  ],
  "validation": [
    {"image": "p01/validation.jpg", "person_id": "p01", "subject_id": "p01", "session": "robot-day2"},
    {"image": "u01/validation.jpg", "person_id": null, "subject_id": "u01", "session": "robot-day2"}
  ],
  "test": [
    {"image": "p01/test.jpg", "person_id": "p01", "subject_id": "p01", "session": "robot-day3"},
    {"image": "u02/test.jpg", "person_id": null, "subject_id": "u02", "session": "robot-day3"}
  ]
}
```

Para aceptación: cinco conocidos, tres imágenes de registro por persona, mínimo veinte consultas de prueba por conocido y cien de al menos cinco desconocidos. Obtener una validación adicional con conocidos y desconocidos distintos de los desconocidos de prueba. No usar las tres fotos de registro para entrenar o evaluar.

Guardar condiciones de cámara, iluminación, distancia y captura en la documentación privada del ensayo. Cada consulta debe ser una observación independiente; no generar el conteo de muestras cortando un video en fotogramas adyacentes.

El cargador impide duplicación exacta de archivos, rutas fuera del directorio, mezcla de sesiones por persona entre particiones y desconocidos compartidos entre validación y prueba. No puede detectar automáticamente todas las imágenes casi idénticas ni verificar que las etiquetas sean verdaderas; revisar manualmente las particiones. Las tres capturas deberían representar a la misma persona; la comprobación por embeddings no es una garantía de identidad ni de presencia real.

## Calibración y prueba congelada

Calcular similitudes con la galería de tres plantillas y la puntuación media de las dos mejores. Buscar conjuntamente umbral y margen en las fronteras observadas de validación. Elegir cero asignaciones incorrectas; maximizar identificaciones correctas y, en empate, favorecer umbral y margen mayores. Siempre existe una configuración que rechaza todo; no se considera éxito si no reconoce al menos 95 % de los conocidos.

La consistencia mínima de registro (0,3), límites de calidad y confianza del detector son políticas provisionales. Para ajustarlas usar validación independiente, versionar todo cambio y reservar una prueba nueva si la anterior ya influyó en las decisiones. El programa no adapta esos valores mirando los resultados de prueba.

`calibrate` solo procesa registro y validación. `evaluate` exige una calibración congelada compatible con el corpus. `experiment` hace ambas etapas en orden y nunca busca parámetros sobre prueba. Un cambio de galería o número de personas exige reevaluar la tasa de confusión; la calibración no garantiza escalabilidad a más identidades.

Reportar identificación correcta de conocidos sobre todas sus consultas, identidades incorrectas, rechazos, estado de procesamiento y falsos positivos sobre desconocidos. Un desconocido sin rostro válido queda sin identidad, pero no se usa como evidencia de capacidad del comparador. Se incluyen resultados por persona e intervalos Wilson de 95 %; su interpretación requiere independencia. El comparador se reporta también solo sobre entradas procesables, sin borrar errores del total.

## Medición

Medir en procesos distintos FP32 e INT8, con igual conjunto, orden, galería, hilos y repeticiones. Rehacer plantillas y calibración para cada variante. El benchmark carga archivos y ejecuta la identificación completa con modelos calientes; excluye la captura física y red. Informar arranque, p50, p95, máximo, consultas >2 s, CPU del proceso, RSS muestreada y máximo de vida del proceso. Los bytes de modelo no prueban uso de INT8 acelerado ni menor consumo de CPU.

En el robot medir además desde el inicio de captura, bajo sus tareas habituales. Acordar límites de impacto con el equipo del robot. Los reportes locales mantienen la aceptación global en falso hasta contar con esa evidencia externa.

# Resultado del prototipo local

Fecha: 23 de octubre de 2026. Estado: **prototipo local, sin aceptación final**.

Se implementó el paquete Python, CLI, validación de registro, galería, política de comparación, calibración separada, evaluación, medición y distribución. Los modelos se descargaron de OpenCV Zoo con revisión y SHA-256 fijados. Se usaron pesos preentrenados; no se ajustaron pesos con transfer learning ni se entrenó una red desde cero.

El experimento puede repetirse con `python scripts/reproduce_local.py` en el entorno instalado. El detalle agregado está en [local-summary.json](local-summary.json). No contiene fotos ni embeddings.

## Calidad observada

AT&T/ORL: cinco personas con tres plantillas por persona; quince consultas conocidas y cincuenta desconocidas para validación; veinte conocidas y cien desconocidas para prueba. Las identidades desconocidas de validación y prueba son distintas. El umbral y el margen se congelaron antes de inferencia de prueba.

| Medición de prueba | FP32 | INT8 |
| --- | --- | --- |
| Personas registradas | 5 | 5 |
| Conocidos correctamente identificados | 17/20 (85 %) | 18/20 (90 %) |
| Conocidos rechazados | 3/20 | 2/20 |
| Conocidos asignados a otra persona | 0/20 | 0/20 |
| Desconocidos asignados a alguien registrado | 0/100 | 0/100 |
| Objetivo de ≥95 % cumplido | No | No |

Las consultas de prueba pudieron procesarse sin fallos de detección o calidad. Los rechazos de conocidos procedieron del umbral o de ambigüedad. Los reportes detallados privados del experimento permiten revisar errores por identidad. No se modificó la política para corregir los resultados de prueba.

Intervalos Wilson aproximados del 95 % para identificación correcta: FP32 64,0–94,8 %, INT8 69,9–97,2 %. Para falsas identificaciones de desconocidos, el intervalo de este ensayo alcanza aproximadamente 3,7 % aun habiendo observado cero casos. Su interpretación supone independencia; ORL no permite garantizar independencia de sesiones. No son garantías poblacionales ni validación del robot.

## Rendimiento local

Entorno: macOS ARM64, Python 3.11.14, OpenCV 4.12.0, NumPy 2.2.6, CPU y dos hilos configurados en OpenCV. El equipo del proyecto fue descrito como MacBook Pro M4 Pro; el identificador de chip no se consultó mediante una API permitida del entorno. No se utilizó GPU.

Benchmarks secuenciales, procesos independientes, cinco calentamientos y doscientas consultas por variante:

| Medición | FP32 | INT8 |
| --- | --- | --- |
| Bytes de modelos, incluyendo detector | 38.929 MB | 10.130 MB |
| Latencia p95 | 10,02 ms | 8,88 ms |
| Latencia máxima | 11,28 ms | 12,32 ms |
| Consultas mayores de 2 s | 0/200 | 0/200 |
| Pico RSS muestreado del proceso | 224,81 MiB | 102,09 MiB |
| Deriva mediana RSS entre mitades | 1,06 MiB | 1,75 MiB |

Los tiempos incluyen decodificación de PGM pequeño, detección, alineación, extracción y comparación. Excluyen cámara física, transferencia, red, consulta de información de negocio y actividades simultáneas del robot. La latencia de fotos grandes del teléfono y capturas reales puede ser diferente.

El uso medio de CPU del proceso fue aproximadamente 236 % y 250 % de un núcleo durante procesamiento continuo. Configurar dos hilos de OpenCV no limita todo el proceso a dos núcleos; bibliotecas nativas y tareas auxiliares también consumen CPU. INT8 no redujo automáticamente esa medida. La frecuencia de invocación y la convivencia con otras tareas deben evaluarse en el robot.

La memoria no mostró crecimiento sostenido según la heurística local documentada (diferencia mediana entre mitades ≤10 MiB), pero doscientas consultas no sustituyen una prueba prolongada ni demuestran ausencia absoluta de fugas.

## Decisión y pendientes

Se conserva FP32 como referencia y se entrega INT8 como candidato experimental, con plantillas, versión y calibración propias. No se promueve automáticamente: ninguna variante cumple todavía la calidad exigida.

Pendiente para aceptación:

- Datos propios de teléfono y robot en sesiones distintas; el conjunto público no verifica el cruce de cámaras.
- Al menos cien consultas conocidas (veinte por cada una de cinco personas) y cien desconocidas, además de calibración independiente.
- Revisión de los rechazos, captura y procesamiento con datos de validación nuevos. Si se necesita ajuste de pesos, preparar otra fase con datos de entrenamiento independientes.
- Confirmar hardware, entorno y cámara del X2 Ultra; medir los dos segundos desde captura física y recursos con voz/navegación activas.
- Acordar presupuesto de recursos y límites de impacto sobre esas tareas con el equipo de integración.

Las pruebas automatizadas verifican contratos y comportamiento del componente, no sustituyen estas validaciones de calidad y hardware. Los comandos que generaron los resultados se ejecutaron localmente sin conexión a Supabase ni implementación de la web.

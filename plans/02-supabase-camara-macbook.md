# Integración de Supabase y cámara del MacBook

Plan aprobado e implementado. Conexión real y una captura de cámara verificadas; ver [reporte](../reports/integracion-supabase-camara.md). La precisión con usuarios reales y el rendimiento del robot siguen pendientes.

## 1. Comportamiento acordado

- Vista previa continua de cámara; Espacio solicita una foto y una sola inferencia; Escape cierra.
- Consultar y validar embeddings de Supabase antes de cada captura; cancelar el intento si falla la actualización.
- Devolver identidad, desconocido o causa de rechazo. Señalar la calibración pública existente como provisional.

## 2. Credenciales y conexión

- Guardar las cuatro variables suministradas en `.env`, con permisos `0600`, normalizando únicamente escapes de formato.
- Excluir `.env` y `.env.*` de Git, excepto `.env.example`, que contendrá variables vacías.
- No escribir credenciales, tokens ni embeddings en logs, reportes o archivos versionados.
- Crear un adaptador independiente con el cliente Python de Supabase y carga de `.env`, versiones fijadas.
- Autenticar con correo y contraseña; sesión en memoria y renovación. Consultas sujetas a RLS; no cambiar políticas, tablas ni registros.

## 3. Lectura y validación

- Añadir `face-lite supabase-check`: autenticar y consultar los cuatro campos de `face_templates`, paginados y ordenados.
- Aceptar vectores como lista numérica o texto de vector, sin renormalizar ni alterar valores.
- Validar dimensión 128, finitud, normalización, identificadores, tres plantillas distintas por persona, consistencia y versión exacta.
- Informar cantidades, versiones y causas estructurales, sin vectores. Una consulta vacía significa «sin filas visibles», no prueba ausencia de registros.
- Crear una galería nueva antes de cada foto. No usar datos antiguos tras un error ni cambiar etiquetas de versiones para forzar compatibilidad.
- Verificar en ejecución la compatibilidad real de los embeddings existentes; no asumirla.

## 4. Captura y reconocimiento

- Añadir `face-lite camera`, configurable para cámara, modelos, procesamiento y calibración.
- Cargar modelos una sola vez; abrir OpenCV; actualizar galería antes de capturar un fotograma reciente sin textos ni transformaciones.
- Impedir concurrencia y colas de capturas. Mostrar estado, identificador, similitud y tiempos de red, captura, inferencia y total; similitud no es certeza.
- Mantener sesión, imágenes y galería solo en memoria. Gestionar permisos, desconexión, cámara ocupada y cierre de ventana; liberar recursos siempre.
- Usar `opencv-python==4.12.0.88` en macOS y la variante headless en otros sistemas; nunca ambas en un entorno.
- El permiso de cámara lo solicita macOS al ejecutar la prueba.

## 5. Verificación y entrega

- Probar configuración y errores sin secretos, autenticación y renovación, vacío y red, paginación y formatos, incompatibilidades y registros incompletos.
- Probar actualización por captura, bloqueo de inferencia tras errores, una sola captura, ausencia de concurrencia y liberación de cámara.
- Ejecutar regresión del componente y comprobar dependencias.
- Consultar Supabase realmente; informar filas/personas visibles y cumplimiento del contrato sin modificar registros.
- Abrir cámara con permiso y comprobar captura/identificación cuando la galería sea compatible; medir red y procesamiento por separado.
- Actualizar README con comandos, controles y solución de problemas. Documentar bloqueos reales sin afirmar validación de precisión ni aceptación del robot.

Referencias: [Supabase Auth](https://supabase.com/docs/reference/python/auth-signinwithpassword), [consultas](https://supabase.com/docs/reference/python/select), [OpenCV Python](https://pypi.org/project/opencv-python/).

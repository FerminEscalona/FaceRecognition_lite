"""Read-only Supabase adapter. Credentials, sessions and vectors stay in memory."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values
from postgrest.exceptions import APIError
from supabase import ClientOptions, create_client
from supabase_auth.errors import AuthError, AuthRetryableError

from .matching import Gallery, vector


class IntegrationError(RuntimeError):
    """Only fixed, safe messages and aggregate diagnostics may reach the UI."""

    def __init__(self, code: str, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics or {}

    def report(self) -> dict:
        return {"status": "error", "code": self.code, "message": str(self), **self.diagnostics}


@dataclass(frozen=True)
class SupabaseSettings:
    url: str = field(repr=False)
    publishable_key: str = field(repr=False)
    email: str = field(repr=False)
    password: str = field(repr=False)

    @classmethod
    def load(cls, env_file: Path = Path(".env")) -> "SupabaseSettings":
        try:
            if env_file.exists() and os.name == "posix" and env_file.stat().st_mode & 0o077:
                raise IntegrationError("env_permissions", "El archivo .env requiere permisos 0600 (chmod 600 .env).")
            # No shell evaluation, interpolation or implicit search in parent directories.
            values = dotenv_values(env_file, interpolate=False) if env_file.exists() else {}
        except OSError:
            raise IntegrationError("env_unreadable", "No se pudo leer el archivo de configuración.") from None
        names = ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "SUPABASE_AUTH_EMAIL", "SUPABASE_AUTH_PASSWORD")
        values = {key: os.environ.get(key, values.get(key)) for key in names}
        if any(not isinstance(values[k], str) or not values[k] for k in names):
            raise IntegrationError("missing_configuration", "Faltan variables requeridas de Supabase; revisa .env.example.")
        url = values[names[0]].rstrip("/")
        try:
            parsed = urlsplit(url)
            valid = (parsed.scheme == "https" and parsed.hostname and not parsed.username
                     and not parsed.password and not parsed.query and not parsed.fragment and not parsed.path)
        except ValueError:
            valid = False
        if not valid:
            raise IntegrationError("invalid_configuration", "SUPABASE_URL debe ser una URL HTTPS del proyecto.")
        if not values[names[1]].startswith("sb_publishable_"):
            raise IntegrationError("invalid_configuration", "Esta demo requiere una clave publicable de Supabase.")
        return cls(url, values[names[1]], values[names[2]], values[names[3]])


def validate_rows(rows: list, model_version: str, dimension: int, minimum: float) -> tuple[Gallery, dict]:
    """Adapt transport only; never repair vectors or silently drop incompatible rows."""
    if not isinstance(rows, list):
        raise IntegrationError("invalid_response", "Supabase no devolvió una lista de plantillas.")
    versions: dict[str, int] = {}
    dimensions: dict[str, int] = {}
    people: set[str] = set()
    invalid_vectors = 0
    converted = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"template_id", "person_id", "embedding", "model_version"}:
            raise IntegrationError("invalid_fields", "Las plantillas no contienen los cuatro campos requeridos.")
        version = row["model_version"]
        safe_version = version if isinstance(version, str) and re.fullmatch(r"[a-zA-Z0-9._:-]{1,100}", version) else "<invalid>"
        versions[safe_version] = versions.get(safe_version, 0) + 1
        if isinstance(row["person_id"], str) and row["person_id"].strip():
            people.add(row["person_id"])
        embedding = row["embedding"]
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except (ValueError, TypeError):
                embedding = None
        size = str(len(embedding)) if isinstance(embedding, list) else "invalid"
        dimensions[size] = dimensions.get(size, 0) + 1
        try:
            vector(embedding, dimension)
        except (TypeError, ValueError, OverflowError):
            invalid_vectors += 1
        converted.append({**row, "embedding": embedding})
    diagnostics = {"visible_rows": len(rows), "visible_people": len(people), "versions": versions,
                   "dimensions": dimensions, "invalid_vectors": invalid_vectors,
                   "expected_model_version": model_version, "expected_dimension": dimension}
    if not rows:
        raise IntegrationError("no_visible_rows", "Sin filas visibles: revisa registros y permisos RLS del usuario.", diagnostics)
    if set(versions) != {model_version}:
        raise IntegrationError("incompatible_model", "Los embeddings no corresponden a la versión exacta del modelo local.", diagnostics)
    if invalid_vectors:
        raise IntegrationError("invalid_embeddings", "Hay embeddings con dimensión, valores o normalización incompatibles.", diagnostics)
    try:
        gallery = Gallery(converted, model_version, dimension)
        gallery.check_consistency(minimum)
    except (ValueError, TypeError, OverflowError):
        raise IntegrationError("invalid_gallery", "Galería inválida: revisa identificadores, tres plantillas distintas por persona y consistencia.", diagnostics) from None
    return gallery, {"status": "compatible", **diagnostics}


class SupabaseGallerySource:
    """Serialized requests; no persistent token storage and no database writes."""

    def __init__(self, settings: SupabaseSettings, *, client=None, page_size: int = 500):
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be in [1, 1000]")
        self._settings = settings
        self._http = None
        if client is None:
            # Bounded timeouts cover Auth as well as PostgREST. No background auth timer.
            self._http = httpx.Client(timeout=10.0)
            try:
                client = create_client(settings.url, settings.publishable_key, options=ClientOptions(
                    persist_session=False, auto_refresh_token=False, postgrest_client_timeout=10,
                    httpx_client=self._http))
            except Exception:
                self._http.close()
                raise IntegrationError("client_initialization", "No se pudo inicializar el cliente de Supabase.") from None
        self._client = client
        self._session = None
        self._page_size = page_size
        self._lock = threading.Lock()

    def _authenticate(self) -> None:
        try:
            if self._session is None:
                response = self._client.auth.sign_in_with_password({
                    "email": self._settings.email, "password": self._settings.password})
            elif self._session.expires_at is None or self._session.expires_at <= time.time() + 60:
                response = self._client.auth.refresh_session(self._session.refresh_token)
            else:
                return
            self._session = response.session
            if self._session is None:
                raise IntegrationError("authentication_failed", "Supabase no devolvió una sesión autenticada.")
        except AuthRetryableError:
            raise IntegrationError("network_error", "No se pudo completar la conexión HTTPS de autenticación.") from None
        except AuthError:
            self._session = None
            raise IntegrationError("authentication_failed", "No se pudo autenticar o renovar la sesión; revisa usuario, contraseña y estado de la cuenta.") from None

    def _read(self) -> list[dict]:
        rows: list[dict] = []
        expected_count = None
        while True:
            response = (self._client.table("face_templates")
                        .select("template_id,person_id,embedding,model_version", count="exact")
                        .order("template_id").range(len(rows), len(rows) + self._page_size - 1).execute())
            if not isinstance(response.data, list):
                raise IntegrationError("invalid_response", "Respuesta de plantillas inválida.")
            if expected_count is None:
                expected_count = response.count
            elif response.count != expected_count:
                raise IntegrationError("gallery_changed", "La galería cambió durante la descarga; repite la consulta.")
            if not response.data:
                if expected_count is not None and len(rows) != expected_count:
                    raise IntegrationError("incomplete_download", "La descarga no coincide con el número de filas visibles.")
                return rows
            rows.extend(response.data)
            if len(rows) > 100_000:
                raise IntegrationError("gallery_too_large", "La galería excede el límite de esta demo local.")
            if expected_count is not None and len(rows) >= expected_count:
                if len(rows) != expected_count:
                    raise IntegrationError("gallery_changed", "La galería cambió durante la descarga; repite la consulta.")
                return rows

    def fetch(self, model_version: str, dimension: int = 128, minimum: float = .3) -> tuple[Gallery, dict]:
        started = time.perf_counter()
        with self._lock:
            try:
                self._authenticate()
                authenticated = time.perf_counter()
                try:
                    rows = self._read()
                except APIError as error:
                    # A server-side JWT expiry/revocation may occur before local expiry.
                    if str(error.code) not in {"PGRST301", "PGRST303"}:
                        raise
                    self._session = None
                    self._authenticate()
                    rows = self._read()  # Restart pagination; never combine sessions.
                downloaded = time.perf_counter()
                gallery, report = validate_rows(rows, model_version, dimension, minimum)
                report["download_ms"] = (time.perf_counter() - started) * 1000
                report["auth_ms"] = (authenticated - started) * 1000
                report["query_ms"] = (downloaded - authenticated) * 1000
                report["validation_ms"] = (time.perf_counter() - downloaded) * 1000
                return gallery, report
            except IntegrationError:
                raise
            except (httpx.HTTPError, OSError):
                raise IntegrationError("network_error", "No se pudo completar la conexión HTTPS con Supabase.") from None
            except APIError:
                raise IntegrationError("query_failed", "Consulta rechazada: revisa tabla, campos y permisos SELECT/RLS.") from None
            except Exception:
                # Third-party exceptions can contain URLs, credentials or response bodies.
                raise IntegrationError("supabase_error", "Supabase devolvió un error inesperado; no se realizó reconocimiento.") from None

    def close(self) -> None:
        # Lock also lets a pending download finish before disposing its HTTP session.
        with self._lock:
            self._session = None
            if self._http is not None:
                self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

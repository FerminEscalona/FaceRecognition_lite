import json
import os
import time
from types import SimpleNamespace as NS

import httpx
import numpy as np
import pytest
from postgrest.exceptions import APIError
from supabase_auth.errors import AuthApiError

from face_lite.supabase_gallery import IntegrationError, SupabaseGallerySource, SupabaseSettings, validate_rows


def rows():
    result = []
    for i in range(3):
        embedding = np.zeros(128)
        embedding[:2] = [1, (i + 1) * .01]
        embedding /= np.linalg.norm(embedding)
        result.append(dict(template_id=f"t{i}", person_id="private-person", model_version="v", embedding=embedding.tolist()))
    return result


def settings():
    return SupabaseSettings("https://example.supabase.co", "sb_publishable_private", "private@example.com", "private-password")


class FakeClient:
    def __init__(self, data=None):
        self.rows = data if data is not None else rows()
        self.auth = self
        self.signins = self.refreshes = 0
        self.queries = []
        self.session = NS(expires_at=time.time() + 3600, refresh_token="private-refresh-token")
        self.error = None
        self.counts = []

    def sign_in_with_password(self, values):
        self.signins += 1
        assert values["password"] == settings().password
        return NS(session=self.session)

    def refresh_session(self, token):
        assert token == self.session.refresh_token
        self.refreshes += 1
        self.session.expires_at = time.time() + 3600
        return NS(session=self.session)

    def table(self, name):
        assert name == "face_templates"
        return self

    def select(self, columns, count):
        assert columns == "template_id,person_id,embedding,model_version"
        assert count == "exact"
        return self

    def order(self, key):
        assert key == "template_id"
        return self

    def range(self, start, end):
        self.interval = (start, end)
        self.queries.append(self.interval)
        return self

    def execute(self):
        if self.error:
            error, self.error = self.error, None
            raise error
        start, end = self.interval
        return NS(data=self.rows[start:end + 1], count=self.counts.pop(0) if self.counts else len(self.rows))


def test_settings_private_file_and_literal_values(tmp_path, monkeypatch):
    for key in ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "SUPABASE_AUTH_EMAIL", "SUPABASE_AUTH_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text('SUPABASE_URL=https://example.supabase.co/\nSUPABASE_PUBLISHABLE_KEY=sb_publishable_x\nSUPABASE_AUTH_EMAIL=a@b.local\nSUPABASE_AUTH_PASSWORD="${LITERAL}\\password"\n')
    env.chmod(0o600)
    config = SupabaseSettings.load(env)
    assert config.url == "https://example.supabase.co"
    assert config.password == '${LITERAL}\\password'
    assert config.password not in repr(config) and config.email not in repr(config)
    monkeypatch.setenv("SUPABASE_AUTH_PASSWORD", "environment-password")
    assert SupabaseSettings.load(env).password == "environment-password"
    if os.name == "posix":
        env.chmod(0o644)
        with pytest.raises(IntegrationError, match="0600"):
            SupabaseSettings.load(env)
    with pytest.raises(IntegrationError) as error:
        SupabaseSettings.load(tmp_path / "absent")
    assert error.value.code == "missing_configuration"


def test_pagination_and_refresh_each_fetch():
    client = FakeClient()
    with SupabaseGallerySource(settings(), client=client, page_size=2) as source:
        g, report = source.fetch("v")
        assert len(g.person_ids) == 1
        assert report["visible_rows"] == 3 and report["dimensions"] == {"128": 3}
        assert client.queries == [(0, 1), (2, 3)]
        source.fetch("v")
        assert len(client.queries) == 4 and client.signins == 1
        client.session.expires_at = time.time() - 1
        source.fetch("v")
        assert client.refreshes == 1


def test_server_expired_session_restarts_download_once():
    client = FakeClient()
    client.error = APIError({"code": "PGRST301", "message": "private-token"})
    source = SupabaseGallerySource(settings(), client=client)
    source.fetch("v")
    assert client.signins == 2 and client.queries == [(0, 499), (0, 499)]


@pytest.mark.parametrize("error, expected", [
    (httpx.ConnectError("private-password"), "network_error"),
    (APIError({"code": "42501", "message": "private-password"}), "query_failed"),
    (RuntimeError("private-password"), "supabase_error"),
])
def test_query_errors_do_not_leak_credentials(error, expected):
    client = FakeClient()
    client.error = error
    source = SupabaseGallerySource(settings(), client=client)
    with pytest.raises(IntegrationError) as caught:
        source.fetch("v")
    report = caught.value.report()
    assert report["code"] == expected
    assert "private-password" not in json.dumps(report)


def test_auth_failure_safe():
    client = FakeClient()
    def fail(_):
        raise AuthApiError("private-password", 400, "invalid_credentials")
    client.sign_in_with_password = fail
    with pytest.raises(IntegrationError) as caught:
        SupabaseGallerySource(settings(), client=client).fetch("v")
    assert caught.value.code == "authentication_failed"
    assert "private-password" not in json.dumps(caught.value.report())
    assert not client.queries


def test_changed_count_cancels_partial_gallery():
    client = FakeClient()
    client.counts = [3, 4]
    with pytest.raises(IntegrationError) as caught:
        SupabaseGallerySource(settings(), client=client, page_size=2).fetch("v")
    assert caught.value.code == "gallery_changed"


def test_text_vector_transport_does_not_change_values():
    source = rows()
    raw = source[0]["embedding"][:]
    for row in source:
        row["embedding"] = json.dumps(row["embedding"])
    gallery, report = validate_rows(source, "v", 128, .3)
    np.testing.assert_array_equal(gallery._matrix[0, 0], np.array(raw, dtype=np.float32))
    assert report["invalid_vectors"] == 0
    assert "private-person" not in json.dumps(report)


@pytest.mark.parametrize("mutation,code", [
    (lambda r: r.clear(), "no_visible_rows"),
    (lambda r: r[0].update(model_version="other"), "incompatible_model"),
    (lambda r: r[0].update(embedding=[1, 0]), "invalid_embeddings"),
    (lambda r: r[0].update(embedding=[2] + [0] * 127), "invalid_embeddings"),
    (lambda r: r[0].update(embedding=[float("nan")] + [0] * 127), "invalid_embeddings"),
    (lambda r: r.pop(), "invalid_gallery"),
    (lambda r: r[1].update(embedding=r[0]["embedding"]), "invalid_gallery"),
])
def test_invalid_gallery_rejected_without_repairs(mutation, code):
    source = rows()
    mutation(source)
    with pytest.raises(IntegrationError) as caught:
        validate_rows(source, "v", 128, .3)
    assert caught.value.code == code

from __future__ import annotations

import importlib
import urllib.parse as up

import pytest
from fastapi.testclient import TestClient

PUB = {"host": "public.example.com"}
_META = {"authorization_endpoint": "https://idp/auth", "token_endpoint": "https://idp/token"}


def _reload_app(tmp_path, monkeypatch, *, sso: bool):
    monkeypatch.setenv("GATEWAY_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("GATEWAY_DB_URL", f"sqlite:///{tmp_path/'gw.db'}")
    monkeypatch.setenv("GATEWAY_UPLOADS_BASE_DIR", str(tmp_path / "up"))
    ry = tmp_path / "services.yaml"
    ry.write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setenv("GATEWAY_REGISTRY_PATH", str(ry))
    if sso:
        monkeypatch.setenv("GATEWAY_AUTH__OIDC_ISSUER", "https://idp/realms/bioq")
        monkeypatch.setenv("GATEWAY_AUTH__OIDC_CLIENT_ID", "cid")
        monkeypatch.setenv("GATEWAY_AUTH__OIDC_CLIENT_SECRET", "sec")
        monkeypatch.setenv("GATEWAY_AUTH__JWT_JWKS_URL", "https://idp/certs")
    import server.app as appmod
    importlib.reload(appmod)
    appmod.app.state.db.create_all()
    return appmod


@pytest.fixture
def sso_app(tmp_path, monkeypatch):
    return _reload_app(tmp_path, monkeypatch, sso=True)


@pytest.fixture
def nosso_app(tmp_path, monkeypatch):
    return _reload_app(tmp_path, monkeypatch, sso=False)


def _login_get_state(client, monkeypatch):
    from server.admin import sso
    monkeypatch.setattr(sso, "discover", lambda issuer, **kw: _META)
    r = client.get("/admin/auth/login", headers=PUB, follow_redirects=False)
    assert r.status_code == 303
    q = up.parse_qs(up.urlparse(r.headers["location"]).query)
    return q["state"][0]


def test_sso_login_404_when_disabled(nosso_app):
    c = TestClient(nosso_app.app)
    r = c.get("/admin/auth/login", headers=PUB, follow_redirects=False)
    assert r.status_code == 404


def test_login_page_shows_sso_button(sso_app):
    c = TestClient(sso_app.app)
    assert "/admin/auth/login" in c.get("/admin/login", headers=PUB).text


def test_sso_login_redirects_to_idp(sso_app, monkeypatch):
    c = TestClient(sso_app.app)
    from server.admin import sso
    monkeypatch.setattr(sso, "discover", lambda issuer, **kw: _META)
    r = c.get("/admin/auth/login", headers=PUB, follow_redirects=False)
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith("https://idp/auth?")
    assert "client_id=cid" in loc and "state=" in loc


def test_sso_callback_admin_logs_in(sso_app, monkeypatch):
    from server.admin import routes, sso
    c = TestClient(sso_app.app)
    state = _login_get_state(c, monkeypatch)
    monkeypatch.setattr(sso, "exchange_code", lambda s, code, ru: {"access_token": "AT"})
    monkeypatch.setattr(routes, "verify_jwt", lambda tok, **kw: {
        "sub": "u1", "groups": ["bioq-admins"], "preferred_username": "u1"})
    r = c.get(f"/admin/auth/callback?code=x&state={state}", headers=PUB,
              follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/admin"
    assert sso_app.app.state.db.get_user("u1").role == "admin"
    assert c.get("/admin", headers=PUB).status_code == 200


def test_sso_callback_non_admin_403(sso_app, monkeypatch):
    from server.admin import routes, sso
    c = TestClient(sso_app.app)
    state = _login_get_state(c, monkeypatch)
    monkeypatch.setattr(sso, "exchange_code", lambda s, code, ru: {"access_token": "AT"})
    monkeypatch.setattr(routes, "verify_jwt", lambda tok, **kw: {
        "sub": "u2", "groups": ["other-group"]})
    r = c.get(f"/admin/auth/callback?code=x&state={state}", headers=PUB,
              follow_redirects=False)
    assert r.status_code == 403
    assert sso_app.app.state.db.get_user("u2").role == "user"  # provisioned, not admin


def test_sso_callback_bad_state(sso_app, monkeypatch):
    c = TestClient(sso_app.app)
    _login_get_state(c, monkeypatch)
    r = c.get("/admin/auth/callback?code=x&state=WRONG", headers=PUB,
              follow_redirects=False)
    assert r.status_code == 400

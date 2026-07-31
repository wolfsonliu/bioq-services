from __future__ import annotations

import importlib
import re

import pytest
from fastapi.testclient import TestClient

VPC = {"host": "fc-x.cn-hangzhou-vpc.fcapp.run"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("GATEWAY_DB_URL", f"sqlite:///{tmp_path/'gw.db'}")
    monkeypatch.setenv("GATEWAY_UPLOADS_BASE_DIR", str(tmp_path / "uploads"))
    registry_yaml = tmp_path / "services.yaml"
    registry_yaml.write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setenv("GATEWAY_REGISTRY_PATH", str(registry_yaml))
    import server.app as appmod
    importlib.reload(appmod)
    appmod.app.state.db.create_all()
    c = TestClient(appmod.app)
    c._registry_yaml = registry_yaml  # for reload test
    return c


def _csrf(client) -> str:
    """Prime the session (VPC bypass) and pull the CSRF token from a page."""
    html = client.get("/admin/accounts", headers=VPC).text
    m = re.search(r'name="csrf" value="([^"]+)"', html)
    assert m, "no csrf token in page"
    return m.group(1)


def test_create_account_requires_csrf(client):
    r = client.post("/admin/accounts", data={"account_id": "x"}, headers=VPC)
    assert r.status_code == 403


def test_create_account(client):
    import server.app as appmod
    tok = _csrf(client)
    r = client.post("/admin/accounts",
                    data={"account_id": "alice", "display_name": "Alice",
                          "role": "admin", "csrf": tok},
                    headers=VPC, follow_redirects=False)
    assert r.status_code == 303
    u = appmod.app.state.db.get_user("alice")
    assert u is not None and u.role == "admin"


def test_create_account_duplicate(client):
    import server.app as appmod
    appmod.app.state.db.create_user("dup")
    tok = _csrf(client)
    r = client.post("/admin/accounts",
                    data={"account_id": "dup", "role": "user", "csrf": tok},
                    headers=VPC)
    assert r.status_code == 400
    assert "exists" in r.text or "已存在" in r.text


def test_create_account_blank_rejected(client):
    tok = _csrf(client)
    r = client.post("/admin/accounts",
                    data={"account_id": "  ", "role": "user", "csrf": tok},
                    headers=VPC)
    assert r.status_code == 400


def test_create_key_shows_secret_once(client):
    import server.app as appmod
    appmod.app.state.db.create_user("alice")
    tok = _csrf(client)
    r = client.post("/admin/accounts/alice/keys", data={"csrf": tok},
                    headers=VPC, follow_redirects=True)
    assert r.status_code == 200
    m = re.search(r"secret: <code>([^<]+)</code>", r.text)
    assert m, "plaintext secret not shown after creation"
    secret = m.group(1)
    # the created key authenticates as alice
    from server.auth.api_key import hash_secret
    assert appmod.app.state.db.find_api_key(hash_secret(secret)).account_id == "alice"
    # revisiting the page does NOT show the secret again (flash popped)
    r2 = client.get("/admin/accounts/alice", headers=VPC)
    assert secret not in r2.text


def test_create_key_unknown_account_404(client):
    tok = _csrf(client)
    r = client.post("/admin/accounts/ghost/keys", data={"csrf": tok}, headers=VPC)
    assert r.status_code == 404


def test_revoke_key(client):
    import server.app as appmod
    db = appmod.app.state.db
    db.create_user("alice")
    db.create_api_key("alice", secret="sv", key_id="gk_rev")
    tok = _csrf(client)
    r = client.post("/admin/keys/gk_rev/revoke",
                    data={"csrf": tok, "account_id": "alice"},
                    headers=VPC, follow_redirects=False)
    assert r.status_code == 303
    from server.auth.api_key import hash_secret
    assert db.find_api_key(hash_secret("sv")) is None


def test_create_key_requires_csrf(client):
    import server.app as appmod
    appmod.app.state.db.create_user("alice")
    r = client.post("/admin/accounts/alice/keys", data={}, headers=VPC)
    assert r.status_code == 403

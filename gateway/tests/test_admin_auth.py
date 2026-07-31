from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

PUB = {"host": "public.example.com"}
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
    return TestClient(appmod.app)


def test_login_required_redirects(client):
    r = client.get("/admin", headers=PUB, follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/admin/login"


def test_login_page_public(client):
    r = client.get("/admin/login", headers=PUB)
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_login_rejects_non_admin(client):
    import server.app as appmod
    appmod.app.state.db.create_user("u")  # role defaults to "user"
    appmod.app.state.db.create_api_key("u", secret="k", key_id="ku")
    r = client.post("/admin/login", data={"api_key": "k"}, headers=PUB)
    assert r.status_code == 401


def test_login_admin_then_access(client):
    import server.app as appmod
    appmod.app.state.db.create_user("root", role="admin")
    appmod.app.state.db.create_api_key("root", secret="ak", key_id="kr")
    r = client.post("/admin/login", data={"api_key": "ak"}, headers=PUB,
                    follow_redirects=False)
    assert r.status_code == 303
    r2 = client.get("/admin", headers=PUB)
    assert r2.status_code == 200


def test_logout_clears_session(client):
    import server.app as appmod
    appmod.app.state.db.create_user("root", role="admin")
    appmod.app.state.db.create_api_key("root", secret="ak", key_id="kr")
    client.post("/admin/login", data={"api_key": "ak"}, headers=PUB,
                follow_redirects=False)
    client.get("/admin/logout", headers=PUB, follow_redirects=False)
    r = client.get("/admin", headers=PUB, follow_redirects=False)
    assert r.status_code == 307


def test_vpc_bypass_no_login(client):
    r = client.get("/admin", headers=VPC)
    assert r.status_code == 200

from __future__ import annotations

import importlib

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
    return TestClient(appmod.app)


def test_dashboard_renders(client):
    import server.app as appmod
    appmod.app.state.db.create_user("a")
    r = client.get("/admin", headers=VPC)
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "root@gateway" in r.text          # terminal chrome
    assert "总览" in r.text                    # default zh


def test_lang_toggle_english(client):
    r = client.get("/admin/setlang?code=en&next=/admin", headers=VPC,
                   follow_redirects=False)
    assert r.status_code == 303
    assert "lang=en" in r.headers.get("set-cookie", "")
    r2 = client.get("/admin", headers=VPC)
    assert "Overview" in r2.text              # now en


def test_dashboard_status_distribution(client):
    import server.app as appmod
    db = appmod.app.state.db
    db.create_user("a")
    db.create_job(job_id="j1", account_id="a", svc="s", endpoint="e",
                  input_params={}, output_prefix=None)
    db.update_job("a", "j1", status="running")
    r = client.get("/admin", headers=VPC)
    assert "running" in r.text


def test_accounts_list(client):
    import server.app as appmod
    db = appmod.app.state.db
    db.create_user("alice")
    db.create_user("bob", role="admin")
    db.create_api_key("alice", secret="s1", key_id="k1")
    r = client.get("/admin/accounts", headers=VPC)
    assert r.status_code == 200
    assert "alice" in r.text and "bob" in r.text


def test_account_detail_hides_secret(client):
    import server.app as appmod
    db = appmod.app.state.db
    db.create_user("alice")
    db.create_api_key("alice", secret="super-secret-value", key_id="k1")
    r = client.get("/admin/accounts/alice", headers=VPC)
    assert r.status_code == 200
    assert "k1" in r.text
    assert "super-secret-value" not in r.text   # secret never rendered


def test_account_detail_404(client):
    r = client.get("/admin/accounts/nobody", headers=VPC)
    assert r.status_code == 404

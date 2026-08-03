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


def _seed_job(appmod, account="alice", job_id="j1", status="running"):
    db = appmod.app.state.db
    db.create_user(account)
    db.create_job(job_id=job_id, account_id=account, svc="s", endpoint="e",
                  input_params={}, output_prefix=None)
    db.update_job(account, job_id, status=status)


def test_cancel_job(client):
    import server.app as appmod
    _seed_job(appmod, job_id="jc", status="running")
    tok = _csrf(client)
    r = client.post("/admin/jobs/alice/jc/cancel", data={"csrf": tok},
                    headers=VPC, follow_redirects=False)
    assert r.status_code == 303
    assert appmod.app.state.db.get_job("alice", "jc").status == "cancelled"


def test_cancel_unknown_job_404(client):
    tok = _csrf(client)
    r = client.post("/admin/jobs/alice/nope/cancel", data={"csrf": tok}, headers=VPC)
    assert r.status_code == 404


def test_cancel_requires_csrf(client):
    import server.app as appmod
    _seed_job(appmod, job_id="jx", status="running")
    r = client.post("/admin/jobs/alice/jx/cancel", data={}, headers=VPC)
    assert r.status_code == 403


def test_services_reload_picks_up_new_service(client):
    # registry starts empty (fixture writes 'services: {}')
    assert "boltz-server" not in client.get("/admin/services", headers=VPC).text
    client._registry_yaml.write_text(
        "services:\n  boltz-server:\n    url: https://boltz.local\n", encoding="utf-8")
    tok = _csrf(client)
    r = client.post("/admin/services/reload", data={"csrf": tok},
                    headers=VPC, follow_redirects=False)
    assert r.status_code == 303
    assert "boltz-server" in client.get("/admin/services", headers=VPC).text


def test_services_reload_requires_csrf(client):
    r = client.post("/admin/services/reload", data={}, headers=VPC)
    assert r.status_code == 403

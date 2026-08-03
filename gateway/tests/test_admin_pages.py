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
    r = client.get("/admin/accounts", headers=VPC)
    assert r.status_code == 200
    assert "alice" in r.text and "bob" in r.text


def test_account_detail(client):
    import server.app as appmod
    appmod.app.state.db.create_user("alice", display_name="Alice")
    r = client.get("/admin/accounts/alice", headers=VPC)
    assert r.status_code == 200
    assert "alice" in r.text


def test_account_detail_404(client):
    r = client.get("/admin/accounts/nobody", headers=VPC)
    assert r.status_code == 404


def _seed_job(appmod, account="alice", job_id="j1", svc="rfdiffusion-server",
              status="running"):
    db = appmod.app.state.db
    db.create_user(account)
    db.create_job(job_id=job_id, account_id=account, svc=svc, endpoint="generate",
                  input_params={"contigs": "100-100"}, output_prefix=None)
    db.update_job(account, job_id, status=status)


def test_jobs_board_and_filter(client):
    import server.app as appmod
    _seed_job(appmod, job_id="j1", status="running")
    _seed_job(appmod, job_id="j2", status="completed")
    r = client.get("/admin/jobs", headers=VPC)
    assert r.status_code == 200
    assert "j1" in r.text and "j2" in r.text
    r2 = client.get("/admin/jobs?status=running", headers=VPC)
    assert "j1" in r2.text and "j2" not in r2.text


def test_job_detail_live_status(client):
    import server.app as appmod
    from bioq_service.service_registry import ServiceRecord
    _seed_job(appmod, job_id="jd")
    appmod.app.state.registry._services = {
        "rfdiffusion-server": ServiceRecord(url="https://svc.local")}

    class _Disp:
        def status(self, rec, task_id):
            return {"status": "completed"}

    appmod.app.state.dispatch = _Disp()
    r = client.get("/admin/jobs/alice/jd", headers=VPC)
    assert r.status_code == 200
    assert "jd" in r.text
    assert "contigs" in r.text            # input_params rendered
    assert "completed" in r.text          # live status


def test_job_detail_degrades_on_status_error(client):
    import server.app as appmod
    from bioq_service.service_registry import ServiceRecord
    _seed_job(appmod, job_id="je")
    appmod.app.state.registry._services = {
        "rfdiffusion-server": ServiceRecord(url="https://svc.local")}

    class _Disp:
        def status(self, rec, task_id):
            raise RuntimeError("downstream boom")

    appmod.app.state.dispatch = _Disp()
    r = client.get("/admin/jobs/alice/je", headers=VPC)
    assert r.status_code == 200            # page still renders


def test_job_detail_404(client):
    r = client.get("/admin/jobs/alice/nope", headers=VPC)
    assert r.status_code == 404


def test_services_list(client):
    import server.app as appmod
    from bioq_service.service_registry import ServiceRecord
    appmod.app.state.registry._services = {
        "rfdiffusion-server": ServiceRecord(url="https://rfd.local", function="fc_rfdiffusion"),
        "boltz-server": ServiceRecord(url="https://boltz.local"),
    }
    r = client.get("/admin/services", headers=VPC)
    assert r.status_code == 200
    assert "rfdiffusion-server" in r.text and "boltz-server" in r.text


def test_services_describe_degrades(client):
    import server.app as appmod
    from bioq_service.service_registry import ServiceRecord
    appmod.app.state.registry._services = {
        "rfdiffusion-server": ServiceRecord(url="https://rfd.local")}

    class _Disp:
        def describe_base_url(self, rec):
            return rec.url

    class _Discover:
        def describe(self, svc, base):
            raise RuntimeError("cold start timeout")

    appmod.app.state.dispatch = _Disp()
    appmod.app.state.discover = _Discover()
    r = client.get("/admin/services?describe=rfdiffusion-server", headers=VPC)
    assert r.status_code == 200            # page still renders
    assert "cold start timeout" in r.text

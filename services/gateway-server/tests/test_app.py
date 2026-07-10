from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("GATEWAY_DB_URL", f"sqlite:///{tmp_path/'gw.db'}")
    monkeypatch.setenv("GATEWAY_UPLOADS_BASE_DIR", str(tmp_path / "uploads"))
    registry_md = tmp_path / "registry.md"
    registry_md.write_text("", encoding="utf-8")
    monkeypatch.setenv("GATEWAY_REGISTRY_PATH", str(registry_md))
    # Re-import app fresh with env applied.
    import importlib
    import server.app as appmod
    importlib.reload(appmod)
    return TestClient(appmod.app)


def test_healthz_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def _seed_key(appmod, principal="alice", secret="s3cr3t", key_id="gk_1"):
    appmod.app.state.db.create_user(principal)
    appmod.app.state.db.create_api_key(principal, secret=secret, key_id=key_id)


def test_v1_requires_auth(client):
    r = client.get("/v1/services", headers={"host": "public.example.com"})
    assert r.status_code == 401


def test_run_and_status_happy(client):
    import server.app as appmod
    _seed_key(appmod)
    appmod.app.state.registry._urls = {"openbpmd-server": "https://svc.local"}

    class _Disp:
        def submit(self, base, ep, job_id, data):
            self.last = (base, ep, job_id, data)

        def status(self, base, job_id):
            return {"status": "completed"}

    appmod.app.state.dispatch = _Disp()

    hdr = {"x-api-key": "s3cr3t", "host": "public.example.com"}
    r = client.post("/v1/run/openbpmd-server/score", json={"nreps": 1}, headers=hdr)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    r2 = client.get(f"/v1/jobs/{job_id}", headers=hdr)
    assert r2.status_code == 200
    assert r2.json()["status"] == "completed"
    assert r2.json()["principal"] == "alice"


def test_run_unknown_service_404(client):
    import server.app as appmod
    _seed_key(appmod, key_id="gk_2", secret="k2")
    hdr = {"x-api-key": "k2", "host": "public.example.com"}
    r = client.post("/v1/run/nope/score", json={}, headers=hdr)
    assert r.status_code == 404


def test_tenant_isolation(client):
    import server.app as appmod
    _seed_key(appmod, principal="alice", secret="alice-sec", key_id="gk_alice")
    _seed_key(appmod, principal="bob", secret="bob-sec", key_id="gk_bob")
    appmod.app.state.registry._urls = {"openbpmd-server": "https://svc.local"}

    class _Disp:
        def submit(self, base, ep, job_id, data):
            pass

        def status(self, base, job_id):
            return {"status": "running"}

    appmod.app.state.dispatch = _Disp()

    alice = {"x-api-key": "alice-sec", "host": "public.example.com"}
    bob = {"x-api-key": "bob-sec", "host": "public.example.com"}

    r = client.post("/v1/run/openbpmd-server/score", json={}, headers=alice)
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    # bob must NOT see or act on alice's job
    assert client.get(f"/v1/jobs/{job_id}", headers=bob).status_code == 404
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=bob).status_code == 404

    # alice can cancel her own job; response status is consistent
    c = client.post(f"/v1/jobs/{job_id}/cancel", headers=alice)
    assert c.status_code == 200
    assert c.json()["status"] == "cancelled"


def test_presign_route(client):
    import server.app as appmod
    _seed_key(appmod, principal="alice", secret="p-sec", key_id="gk_p")
    from server.models import PresignResponse

    class _FakePresigner:
        def presign_put(self, principal, filename, sha256):
            return PresignResponse(uri=f"oss://b/users/{principal}/inputs/{sha256}/{filename}",
                                   exists=False, url="https://oss.put/signed")

    appmod.app.state.presigner = _FakePresigner()
    hdr = {"x-api-key": "p-sec", "host": "public.example.com"}
    r = client.post("/v1/uploads/presign",
                    json={"filename": "x.rst7", "sha256": "abc"}, headers=hdr)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["url"] == "https://oss.put/signed"
    assert body["uri"] == "oss://b/users/alice/inputs/abc/x.rst7"
    assert body["exists"] is False

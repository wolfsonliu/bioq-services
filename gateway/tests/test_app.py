from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("GATEWAY_DB_URL", f"sqlite:///{tmp_path/'gw.db'}")
    monkeypatch.setenv("GATEWAY_UPLOADS_BASE_DIR", str(tmp_path / "uploads"))
    registry_yaml = tmp_path / "services.yaml"
    registry_yaml.write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setenv("GATEWAY_REGISTRY_PATH", str(registry_yaml))
    # Re-import app fresh with env applied.
    import importlib
    import server.app as appmod
    importlib.reload(appmod)
    # App startup no longer create_all()s (schema is Alembic-managed in prod);
    # bootstrap the throwaway sqlite schema for the test.
    appmod.app.state.db.create_all()
    return TestClient(appmod.app)


def test_healthz_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def _seed_key(appmod, account_id="alice", secret="s3cr3t", key_id="gk_1"):
    appmod.app.state.db.create_user(account_id)
    appmod.app.state.db.create_api_key(account_id, secret=secret, key_id=key_id)


def test_v1_requires_auth(client):
    r = client.get("/v1/services", headers={"host": "public.example.com"})
    assert r.status_code == 401


def test_run_and_status_happy(client):
    import server.app as appmod
    _seed_key(appmod)
    from bioq_service.service_registry import ServiceRecord
    appmod.app.state.registry._services = {
        "openbpmd-server": ServiceRecord(url="https://svc.local")
    }

    class _Disp:
        def submit(self, base, ep, job_id, data, oss_prefix=None):
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
    assert r2.json()["account_id"] == "alice"


def test_run_unknown_service_404(client):
    import server.app as appmod
    _seed_key(appmod, key_id="gk_2", secret="k2")
    hdr = {"x-api-key": "k2", "host": "public.example.com"}
    r = client.post("/v1/run/nope/score", json={}, headers=hdr)
    assert r.status_code == 404


def test_tenant_isolation(client):
    import server.app as appmod
    _seed_key(appmod, account_id="alice", secret="alice-sec", key_id="gk_alice")
    _seed_key(appmod, account_id="bob", secret="bob-sec", key_id="gk_bob")
    from bioq_service.service_registry import ServiceRecord
    appmod.app.state.registry._services = {
        "openbpmd-server": ServiceRecord(url="https://svc.local")
    }

    class _Disp:
        def submit(self, base, ep, job_id, data, oss_prefix=None):
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
    _seed_key(appmod, account_id="alice", secret="p-sec", key_id="gk_p")
    from server.models import PresignResponse

    class _FakePresigner:
        def presign_put(self, account_id, job_id, filename, sha256=None):
            return PresignResponse(
                uri=f"oss://b/users/{account_id}/{job_id}/input/{filename}",
                exists=False, url="https://oss.put/signed")

    appmod.app.state.storage = _FakePresigner()
    hdr = {"x-api-key": "p-sec", "host": "public.example.com"}
    r = client.post("/v1/uploads/presign",
                    json={"job_id": "job1", "filename": "x.rst7"}, headers=hdr)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["uri"] == "oss://b/users/alice/job1/input/x.rst7"
    assert body["exists"] is False


def test_download_redirects_to_oss_when_present(client):
    import server.app as appmod
    _seed_key(appmod, account_id="alice", secret="d-sec", key_id="gk_d")
    from bioq_service.service_registry import ServiceRecord
    appmod.app.state.registry._services = {"openbpmd-server": ServiceRecord(url="https://svc.local")}

    class _Disp:
        def submit(self, base, ep, job_id, data, oss_prefix=None):
            pass

        def status(self, base, job_id):
            return {"status": "completed"}

    appmod.app.state.dispatch = _Disp()

    class _Presigner:
        def presign_get_if_exists(self, account_id, job_id, filename):
            return "https://oss.get/signed-results"

    appmod.app.state.storage = _Presigner()

    hdr = {"x-api-key": "d-sec", "host": "public.example.com"}
    r = client.post("/v1/run/openbpmd-server/score",
                    json={}, headers={**hdr, "x-bioagent-job-id": "cjob1"})
    assert r.status_code == 202
    assert r.json()["job_id"] == "cjob1"

    r2 = client.get("/v1/jobs/cjob1/download", headers=hdr, follow_redirects=False)
    assert r2.status_code == 302
    assert r2.headers["location"] == "https://oss.get/signed-results"


def test_download_falls_back_to_proxy_when_not_on_oss(client):
    import server.app as appmod
    _seed_key(appmod, account_id="alice", secret="f-sec", key_id="gk_f")
    from bioq_service.service_registry import ServiceRecord
    appmod.app.state.registry._services = {"openbpmd-server": ServiceRecord(url="https://svc.local")}

    class _Disp:
        def submit(self, base, ep, job_id, data, oss_prefix=None):
            pass

        def status(self, base, job_id):
            return {"status": "completed"}

        def download(self, base, job_id, dest):
            from pathlib import Path
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"PROXYZIP")
            return dest

    appmod.app.state.dispatch = _Disp()

    class _Presigner:
        def presign_get_if_exists(self, account_id, job_id, filename):
            return None  # not on OSS => fall back to proxying the downstream

    appmod.app.state.storage = _Presigner()

    hdr = {"x-api-key": "f-sec", "host": "public.example.com"}
    r = client.post("/v1/run/openbpmd-server/score",
                    json={}, headers={**hdr, "x-bioagent-job-id": "fjob1"})
    assert r.status_code == 202
    r2 = client.get("/v1/jobs/fjob1/download", headers=hdr)
    assert r2.status_code == 200
    assert r2.content == b"PROXYZIP"


def test_run_rewrites_oss_inputs_to_mount_for_mounted_service(client):
    import server.app as appmod
    _seed_key(appmod, account_id="alice", secret="m-sec", key_id="gk_m")
    from bioq_service.service_registry import ServiceRecord
    appmod.app.state.registry._services = {
        "openbpmd-server": ServiceRecord(url="https://svc.local", oss_mount=True),
    }

    captured = {}

    class _Disp:
        def submit(self, base, ep, job_id, data, oss_prefix=None):
            captured["data"] = data

        def status(self, base, job_id):
            return {"status": "running"}

    appmod.app.state.dispatch = _Disp()

    hdr = {"x-api-key": "m-sec", "host": "public.example.com"}
    r = client.post(
        "/v1/run/openbpmd-server/score",
        headers={**hdr, "x-bioagent-job-id": "mjob1"},
        json={"structure_uri": "oss://bioagent-inputs/users/alice/mjob1/input/x.rst7",
              "nreps": 1},
    )
    assert r.status_code == 202, r.text
    # default GATEWAY_OSS_BUCKET is "bioagent-inputs"
    assert captured["data"]["structure_uri"] == "/mnt/oss/users/alice/mjob1/input/x.rst7"
    assert captured["data"]["nreps"] == 1


def test_run_keeps_oss_uris_for_unmounted_service(client):
    import server.app as appmod
    _seed_key(appmod, account_id="alice", secret="u-sec", key_id="gk_u")
    from bioq_service.service_registry import ServiceRecord
    appmod.app.state.registry._services = {
        "openbpmd-server": ServiceRecord(url="https://svc.local"),  # oss_mount defaults False
    }

    captured = {}

    class _Disp:
        def submit(self, base, ep, job_id, data, oss_prefix=None):
            captured["data"] = data

        def status(self, base, job_id):
            return {"status": "running"}

    appmod.app.state.dispatch = _Disp()

    hdr = {"x-api-key": "u-sec", "host": "public.example.com"}
    r = client.post(
        "/v1/run/openbpmd-server/score",
        headers={**hdr, "x-bioagent-job-id": "ujob1"},
        json={"structure_uri": "oss://bioagent-inputs/users/alice/ujob1/input/x.rst7"},
    )
    assert r.status_code == 202
    assert captured["data"]["structure_uri"] == "oss://bioagent-inputs/users/alice/ujob1/input/x.rst7"


def test_file_backend_put_get_roundtrip(client, tmp_path):
    import server.app as appmod
    from server.storage import FileStorage
    _seed_key(appmod, account_id="alice", secret="fa", key_id="gk_fa")
    shared = tmp_path / "shared"
    appmod.app.state.storage = FileStorage(shared)

    hdr = {"x-api-key": "fa", "host": "public.example.com"}
    key = "users/alice/j1/input/x.pdb"
    r = client.put(f"/v1/files/{key}", content=b"HELLO", headers=hdr)
    assert r.status_code == 200, r.text
    assert (shared / key).read_bytes() == b"HELLO"

    r2 = client.get(f"/v1/files/{key}", headers=hdr)
    assert r2.status_code == 200
    assert r2.content == b"HELLO"


def test_file_backend_tenant_guard(client, tmp_path):
    import server.app as appmod
    from server.storage import FileStorage
    _seed_key(appmod, account_id="alice", secret="fa", key_id="gk_fa")
    appmod.app.state.storage = FileStorage(tmp_path / "shared")

    hdr = {"x-api-key": "fa", "host": "public.example.com"}
    # alice may not write under bob's prefix
    r = client.put("/v1/files/users/bob/j1/input/x.pdb", content=b"x", headers=hdr)
    assert r.status_code == 403
    r2 = client.get("/v1/files/users/bob/j1/results.zip", headers=hdr)
    assert r2.status_code == 403


def test_file_routes_404_on_non_file_backend(client):
    import server.app as appmod
    _seed_key(appmod, account_id="alice", secret="fa", key_id="gk_fa")

    class _NotFile:
        def presign_put(self, *a, **k):
            raise AssertionError

        def presign_get_if_exists(self, *a, **k):
            return None

    appmod.app.state.storage = _NotFile()
    hdr = {"x-api-key": "fa", "host": "public.example.com"}
    r = client.put("/v1/files/users/alice/j1/input/x.pdb", content=b"x", headers=hdr)
    assert r.status_code == 404


def test_file_backend_download_redirects_to_gateway_file_url(client, tmp_path):
    import server.app as appmod
    from bioq_service.service_registry import ServiceRecord
    from server.storage import FileStorage
    _seed_key(appmod, account_id="alice", secret="fa", key_id="gk_fa")
    appmod.app.state.registry._services = {"openbpmd-server": ServiceRecord(url="https://svc.local")}

    class _Disp:
        def submit(self, rec, ep, job_id, data, oss_prefix=None):
            pass

        def status(self, rec, job_id):
            return {"status": "completed"}

    appmod.app.state.dispatch = _Disp()
    shared = tmp_path / "shared"
    appmod.app.state.storage = FileStorage(shared)
    # simulate the worker having mirrored results to the shared volume
    out = shared / "users/alice/dljob/results.zip"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"RESULTZIP")

    hdr = {"x-api-key": "fa", "host": "public.example.com"}
    r = client.post("/v1/run/openbpmd-server/score", json={},
                    headers={**hdr, "x-bioagent-job-id": "dljob"})
    assert r.status_code == 202
    r2 = client.get("/v1/jobs/dljob/download", headers=hdr, follow_redirects=False)
    assert r2.status_code == 302
    assert r2.headers["location"] == "/v1/files/users/alice/dljob/results.zip"

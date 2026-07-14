"""Offline tests for diamond-server (no real DIAMOND binary needed)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DATA_DIR = Path(__file__).resolve().parent / "data"
QUERY = DATA_DIR / "query.faa"
SUBJECT = DATA_DIR / "subject.faa"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Import `server.app` fresh against patched env vars."""
    monkeypatch.setenv("DIAMOND_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("DIAMOND_DB_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("DIAMOND_MSA_DB", "ref.dmnd")
    monkeypatch.setenv("DIAMOND_BINARY", "/bin/true")  # never really executed here
    (tmp_path / "db").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Health / manifest -----

def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "diamond"
    assert "version" in health


def test_healthz_detail(client):
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "diamond"
    assert "db_loaded" in detail
    assert "db_dir" in detail


def test_manifest_lists_four_endpoints(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    for p in ("/api/blastp", "/api/blastx", "/api/cluster", "/api/msa"):
        assert p in paths


def test_manifest_extras(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "tool_outputs" in extras
    for k in ("blastp", "blastx", "cluster", "msa"):
        assert k in extras["tool_outputs"]
    assert "db_handling" in extras


def test_task_endpoints_registered(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    for p in ("/api/tasks/blastp", "/api/tasks/blastx", "/api/tasks/cluster", "/api/tasks/msa"):
        assert p in paths


# ----- Submit (argv built, /bin/true "runs") -----

def _files(*specs):
    return [(name, (path.name, path.open("rb"), "text/plain")) for name, path in specs]


def test_blastp_submit_with_subject(client):
    r = client.post(
        "/api/blastp",
        files=_files(("query", QUERY), ("subject", SUBJECT)),
        data={"name": "hits"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] in ("pending", "running", "completed")
    assert "job_id" in body


def test_blastx_submit_with_subject(client):
    r = client.post(
        "/api/blastx",
        files=_files(("query", QUERY), ("subject", SUBJECT)),
        data={"name": "hits"},
    )
    assert r.status_code == 200, r.text


def test_cluster_submit(client):
    r = client.post(
        "/api/cluster",
        files=_files(("sequences", SUBJECT)),
        data={"algorithm": "cluster", "name": "lib"},
    )
    assert r.status_code == 200, r.text


def test_msa_submit_default_db(client):
    r = client.post(
        "/api/msa",
        files=_files(("query", QUERY)),
        data={"name": "query"},
    )
    assert r.status_code == 200, r.text


# ----- Validation (422) -----

def test_blastp_missing_reference_422(client):
    r = client.post("/api/blastp", files=_files(("query", QUERY)), data={"name": "x"})
    assert r.status_code == 422


def test_blastp_both_references_422(client):
    r = client.post(
        "/api/blastp",
        files=_files(("query", QUERY), ("subject", SUBJECT)),
        data={"name": "x", "db_uri": "file:///tmp/ref.dmnd"},
    )
    assert r.status_code == 422


def test_blastp_bad_outfmt_422(client):
    r = client.post(
        "/api/blastp",
        files=_files(("query", QUERY), ("subject", SUBJECT)),
        data={"name": "x", "outfmt": "999"},
    )
    assert r.status_code == 422


def test_cluster_bad_algorithm_422(client):
    r = client.post(
        "/api/cluster",
        files=_files(("sequences", SUBJECT)),
        data={"algorithm": "nope"},
    )
    assert r.status_code == 422


def test_msa_no_db_422(client, monkeypatch, tmp_path):
    monkeypatch.delenv("DIAMOND_MSA_DB", raising=False)
    sys.modules.pop("server.app", None)
    import importlib as _il
    app2 = _il.import_module("server.app")
    with TestClient(app2.app) as c2:
        r = c2.post("/api/msa", files=_files(("query", QUERY)), data={"name": "q"})
        assert r.status_code == 422

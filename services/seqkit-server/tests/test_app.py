"""Offline tests for seqkit-server (no real seqkit binary needed).

SEQKIT_BIN is pointed at /bin/true so the argv is built and the runner "runs"
without actually invoking seqkit. Validation (422) paths never reach the
subprocess.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DATA_DIR = Path(__file__).resolve().parent / "data"
FASTA = DATA_DIR / "input.fasta"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Import `server.app` fresh against patched env vars."""
    monkeypatch.setenv("SEQKIT_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("SEQKIT_BIN", "/bin/true")  # never really executes seqkit

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


def _files():
    return {"input_fasta": (FASTA.name, FASTA.open("rb"), "text/plain")}


# ----- Health / manifest -----

def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "seqkit"
    assert "version" in health


def test_healthz_detail(client):
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "seqkit"
    assert "ready" in detail
    assert "checks" in detail
    assert detail["checks"]["bin_exists"] is True


def test_manifest_lists_endpoints(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/stats" in paths
    assert "/api/revcomp" in paths


def test_task_endpoints_registered(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/tasks/stats" in paths
    assert "/api/tasks/revcomp" in paths


def test_manifest_extras(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "tool_outputs" in extras
    assert "stats" in extras["tool_outputs"]
    assert "revcomp" in extras["tool_outputs"]
    assert "input_uri_schemes" in extras


# ----- Submit (argv built, /bin/true "runs") -----

def test_stats_submit_default(client):
    r = client.post("/api/stats", files=_files())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] in ("pending", "running", "completed")
    assert "job_id" in body


def test_stats_submit_core_only(client):
    r = client.post("/api/stats", files=_files(), data={"all_stats": "false"})
    assert r.status_code == 200, r.text


def test_revcomp_submit_default(client):
    r = client.post("/api/revcomp", files=_files())
    assert r.status_code == 200, r.text
    assert "job_id" in r.json()


def test_revcomp_submit_dna(client):
    r = client.post("/api/revcomp", files=_files(), data={"seq_type": "dna"})
    assert r.status_code == 200, r.text


# ----- Validation (422) -----

def test_bad_seq_type_422(client):
    r = client.post("/api/revcomp", files=_files(), data={"seq_type": "nope"})
    assert r.status_code == 422


def test_stats_missing_input_422(client):
    r = client.post("/api/stats")
    assert r.status_code == 422


def test_revcomp_missing_input_422(client):
    r = client.post("/api/revcomp")
    assert r.status_code == 422

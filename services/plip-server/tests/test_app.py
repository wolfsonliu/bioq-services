"""Offline tests for plip-server (no real PLIP binary needed).

PLIP_PYTHON is pointed at /bin/true so the argv is built and the runner "runs"
without actually invoking PLIP. Validation (422) paths never reach the subprocess.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DATA_DIR = Path(__file__).resolve().parent / "data"
PDB = DATA_DIR / "1vsn.pdb"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Import `server.app` fresh against patched env vars."""
    monkeypatch.setenv("PLIP_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("PLIP_UPSTREAM_DIR", str(tmp_path / "upstream"))
    monkeypatch.setenv("PLIP_PYTHON", "/bin/true")  # never really executes PLIP
    (tmp_path / "upstream" / "plip").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


def _files():
    return {"input_pdb": (PDB.name, PDB.open("rb"), "chemical/x-pdb")}


# ----- Health / manifest -----

def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "plip"
    assert "version" in health


def test_healthz_detail(client):
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "plip"
    assert "ready" in detail
    assert "checks" in detail
    assert "pymol_available" in detail


def test_manifest_lists_profile(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/profile" in paths


def test_task_endpoint_registered(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/tasks/profile" in paths


def test_manifest_extras(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "tool_outputs" in extras
    assert "profile" in extras["tool_outputs"]
    assert "modes" in extras
    assert "interaction_types" in extras


# ----- Submit (argv built, /bin/true "runs") -----

def test_profile_submit_default(client):
    r = client.post("/api/profile", files=_files(), data={"name": "vsn"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] in ("pending", "running", "completed")
    assert "job_id" in body


def test_profile_submit_dnareceptor(client):
    r = client.post(
        "/api/profile", files=_files(),
        data={"name": "vsn", "mode": "dnareceptor", "report_formats": json.dumps(["xml"])},
    )
    assert r.status_code == 200, r.text


def test_profile_submit_peptide(client):
    r = client.post(
        "/api/profile", files=_files(),
        data={"name": "vsn", "mode": "peptide", "peptide_chains": json.dumps(["A"])},
    )
    assert r.status_code == 200, r.text


# ----- Validation (422) -----

def test_bad_mode_422(client):
    r = client.post("/api/profile", files=_files(), data={"mode": "nope"})
    assert r.status_code == 422


def test_bad_report_format_422(client):
    r = client.post("/api/profile", files=_files(), data={"report_formats": json.dumps(["pdf"])})
    assert r.status_code == 422


def test_peptide_missing_chains_422(client):
    r = client.post("/api/profile", files=_files(), data={"mode": "peptide"})
    assert r.status_code == 422


def test_intra_missing_chain_422(client):
    r = client.post("/api/profile", files=_files(), data={"mode": "intra"})
    assert r.status_code == 422


def test_bad_name_422(client):
    r = client.post("/api/profile", files=_files(), data={"name": "bad/name"})
    assert r.status_code == 422

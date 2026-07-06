"""Offline HTTP tests — mock SubprocessRunner, verify endpoint wiring."""
from __future__ import annotations

import importlib
import io
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("QLIGFEP_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("QLIGFEP_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("QLIGFEP_UPSTREAM_DIR", str(tmp_path / "upstream" / "qligfep"))
    (tmp_path / "root").mkdir()
    (tmp_path / "upstream" / "qligfep").mkdir(parents=True)
    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


# ----- Health / manifest -----

def test_health_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "qligfep"
    assert "version" in body


def test_healthz_detail_reports_missing_binaries(client):
    r = client.get("/healthz/detail")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "qligfep"
    assert body["binaries_loaded"] is False  # tmp_path has no Q6
    assert "qdyn" in body["binaries_missing"]
    assert body["task_endpoints_enabled"] is False


def test_manifest_lists_all_endpoints(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "qligfep"
    paths = {e["path"] for e in body["endpoints"]}
    for p in (
        "/api/ligprep", "/api/protprep", "/api/cog",
        "/api/setup-ligfep", "/api/setup-resfep", "/api/setup-lie",
        "/api/run-fep", "/api/analyze-fep", "/api/analyze-lie",
    ):
        assert p in paths


def test_manifest_no_task_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert not any(p.startswith("/api/tasks/") for p in paths)


def test_manifest_extras_has_tool_outputs(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["no_model_weights"] is True
    assert "qdyn_cuda" in extras["q6_binaries"]["gpu"]


# ----- Endpoint smoke: verify submit returns job_id (subprocess never runs) -----

def test_ligprep_submits(client):
    r = client.post(
        "/api/ligprep",
        data={"ligand_name": "17"},
        files={"ligand": ("17.mol2", b"@<TRIPOS>MOLECULE\n17\n", "chemical/x-mol2")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["ligand_name"] == "17"


def test_protprep_submits(client):
    r = client.post(
        "/api/protprep",
        data={"sphere_radius": "22", "sphere_center": "0:0:0", "forcefield": "OPLSAAM"},
        files={"protein_pdb": ("p.pdb", b"ATOM 1", "chemical/x-pdb")},
    )
    assert r.status_code == 200, r.text
    assert "job_id" in r.json()


def test_cog_submits(client):
    r = client.post(
        "/api/cog",
        data={"mode": "all"},
        files={"pdb": ("p.pdb", b"ATOM 1", "chemical/x-pdb")},
    )
    assert r.status_code == 200, r.text


def test_setup_ligfep_submits(client):
    lig_zip = _zip_bytes({"17.lib": b"x", "17.prm": b"x", "17.pdb": b"x",
                          "18.lib": b"x", "18.prm": b"x", "18.pdb": b"x"})
    prot_zip = _zip_bytes({"protein.pdb": b"ATOM", "water.pdb": b"HETATM"})
    r = client.post(
        "/api/setup-ligfep",
        data={"lig1_name": "17", "lig2_name": "18"},
        files={"ligprep_zip": ("lp.zip", lig_zip, "application/zip"),
               "protprep_zip": ("pp.zip", prot_zip, "application/zip")},
    )
    assert r.status_code == 200, r.text


def test_run_fep_submits(client):
    setup_zip = _zip_bytes({"FEP1/md_0500_0500.inp": b"md"})
    r = client.post(
        "/api/run-fep",
        data={"window_idx": "0", "leg": "protein", "device": "cpu"},
        files={"setup_zip": ("s.zip", setup_zip, "application/zip")},
    )
    assert r.status_code == 200, r.text


def test_analyze_fep_submits(client):
    r = client.post(
        "/api/analyze-fep",
        data={"temperature": "298.15", "start": "0.5"},
        files={"fep_run_zip": ("r.zip", _zip_bytes({"x.txt": b""}), "application/zip")},
    )
    assert r.status_code == 200, r.text


# ----- Settings -----

def test_settings_defaults():
    from server.settings import QligfepSettings

    class _Off(QligfepSettings):
        model_config = SettingsConfigDict(env_prefix="QLIGFEP_TEST_", env_file=None, extra="ignore")

    s = _Off()
    assert s.jobs_base_dir == Path("/data/qligfep_jobs")
    assert s.q_bin_dir == Path("/opt/Q6/bin")
    assert s.task_endpoints_enabled is False
    assert s.upstream_dir == Path("/opt/qligfep-server/upstream/qligfep")

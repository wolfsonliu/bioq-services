"""Offline FastAPI app tests for rfdiffusion2-server.

Uses /bin/true as the subprocess so the task endpoints return immediately
without needing a real RFdiffusion2 install.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RFDIFFUSION2_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("RFDIFFUSION2_ROOT", str(tmp_path / "rfdiffusion2"))
    monkeypatch.setenv("RFDIFFUSION2_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv(
        "RFDIFFUSION2_INFERENCE_SCRIPT",
        str(tmp_path / "rfdiffusion2" / "rf_diffusion" / "run_inference.py"),
    )
    monkeypatch.setenv(
        "RFDIFFUSION2_PYTHONPATH",
        str(tmp_path / "rfdiffusion2"),
    )
    monkeypatch.setenv("RFDIFFUSION2_PYTHON", "/bin/true")
    (tmp_path / "rfdiffusion2" / "rf_diffusion").mkdir(parents=True, exist_ok=True)
    (tmp_path / "rfdiffusion2" / "rf_diffusion" / "run_inference.py").write_text("")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Task endpoint smoke (synchronous; /bin/true so it returns immediately) -----


def test_active_site_task_endpoint_accepts_upload(client):
    """POST /api/tasks/generate/active_site accepts upload and blocks."""
    pdb_bytes = b"REMARK fake\nATOM\nEND\n"
    resp = client.post(
        "/api/tasks/generate/active_site",
        data={
            "contigs": "46,A106-106,59,A166-166,2,A169-169,23,A193-193,46",
            "ligand": "NAD,OXM",
            "contig_atoms": (
                '{"A106": "NE,CD,CZ", "A166": "OD1,CG", '
                '"A169": "NH2,CZ", "A193": "NE2,CD2,CE1"}'
            ),
            "contig_as_guidepost": "true",
            "num_designs": "1",
        },
        files={"input_pdb": ("motif.pdb", pdb_bytes, "chemical/x-pdb")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_small_molecule_binder_task_endpoint_honors_job_id_header(client):
    """POST /api/tasks/generate/small_molecule_binder respects X-Bioagent-Job-Id."""
    pdb_bytes = b"REMARK\nEND\n"
    resp = client.post(
        "/api/tasks/generate/small_molecule_binder",
        data={
            "contigs": "150",
            "length": "150-150",
            "ligand": "PH2",
            "rasa_active": "true",
            "rasa_target": "0.0",
            "num_designs": "1",
        },
        files={"input_pdb": ("ligand.pdb", pdb_bytes, "chemical/x-pdb")},
        headers={"X-Bioagent-Job-Id": "rfd2-task-001"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["job_id"] == "rfd2-task-001"
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_generate_custom_task_endpoint_returns_terminal_status(client):
    """POST /api/tasks/generate (custom) works without an input PDB."""
    resp = client.post(
        "/api/tasks/generate",
        data={
            "contigs": "150",
            "config_name": "aa",
            "num_designs": "1",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None

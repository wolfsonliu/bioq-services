"""Offline FastAPI app tests for rfdiffusion-server.

Uses /bin/true as the subprocess so the task endpoints return immediately
without needing a real RFdiffusion install.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RFDIFFUSION_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("RFDIFFUSION_ROOT", str(tmp_path / "rfdiffusion"))
    monkeypatch.setenv("RFDIFFUSION_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv(
        "RFDIFFUSION_INFERENCE_SCRIPT",
        str(tmp_path / "rfdiffusion" / "scripts" / "run_inference.py"),
    )
    monkeypatch.setenv("RFDIFFUSION_PYTHON", "/bin/true")
    (tmp_path / "rfdiffusion" / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "rfdiffusion" / "scripts" / "run_inference.py").write_text("")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Task endpoint smoke (synchronous; /bin/true so it returns immediately) -----


def test_unconditional_task_endpoint_returns_terminal_status(client):
    """POST /api/tasks/generate/unconditional blocks until subprocess exits."""
    resp = client.post(
        "/api/tasks/generate/unconditional",
        data={"min_length": "100", "max_length": "100", "num_designs": "1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_unconditional_task_endpoint_honors_job_id_header(client):
    resp = client.post(
        "/api/tasks/generate/unconditional",
        data={"min_length": "100", "max_length": "100", "num_designs": "1"},
        headers={"X-Bioagent-Job-Id": "rfdiff-task-001"},
    )
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "rfdiff-task-001"


def test_symmetry_task_endpoint_returns_terminal_status(client):
    """POST /api/tasks/generate/symmetry blocks until subprocess exits."""
    resp = client.post(
        "/api/tasks/generate/symmetry",
        data={"symmetry": "c6", "total_length": "480", "num_designs": "1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_binder_task_endpoint_with_upload_returns_terminal_status(client, tmp_path):
    """POST /api/tasks/generate/binder accepts an UploadFile + blocks until done."""
    pdb = tmp_path / "target.pdb"
    pdb.write_text("ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00\n")

    with open(pdb, "rb") as fh:
        resp = client.post(
            "/api/tasks/generate/binder",
            data={
                "contigs": "A1-150/0 70-100",
                "num_designs": "1",
            },
            files={"input_pdb": ("target.pdb", fh, "chemical/x-pdb")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None

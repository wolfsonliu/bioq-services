"""Offline HTTP tests — subprocess never runs (submit returns before exec)."""
from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REINVENT_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("REINVENT_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("REINVENT_PRIOR_BASE", str(tmp_path / "priors"))
    (tmp_path / "root").mkdir()
    sys.modules.pop("server.app", None)
    app_mod = importlib.import_module("server.app")
    return TestClient(app_mod.app)


def test_health(client):
    body = client.get("/healthz").json()
    assert body["service"] == "reinvent"


def test_healthz_detail_reports_missing_priors(client):
    body = client.get("/healthz/detail").json()
    assert body["service"] == "reinvent"
    assert body["priors_loaded"] is False
    assert ".reinvent" in body["priors_missing"]
    assert body["task_endpoints_enabled"] is True
    # cuda_available derives from the GPU list; no nvidia-smi in the test env.
    assert body["cuda_available"] is False
    assert body["gpus"] == []


def test_probe_gpus_parses_nvidia_smi(client, monkeypatch):
    app_mod = sys.modules["server.app"]

    class _R:
        returncode = 0
        stdout = "Tesla T4\nTesla T4\n"

    monkeypatch.setattr(app_mod, "_GPU_CACHE", None)
    monkeypatch.setattr(app_mod.subprocess, "run", lambda *a, **k: _R())
    assert app_mod._probe_gpus() == ["Tesla T4", "Tesla T4"]


def test_probe_gpus_empty_when_nvidia_smi_absent(client, monkeypatch):
    app_mod = sys.modules["server.app"]

    def _boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(app_mod, "_GPU_CACHE", None)
    monkeypatch.setattr(app_mod.subprocess, "run", _boom)
    assert app_mod._probe_gpus() == []


def test_manifest_lists_endpoints_and_tasks(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    for p in ("/api/sampling", "/api/scoring", "/api/enumeration",
              "/api/transfer-learning", "/api/staged-learning"):
        assert p in paths
    assert any(p.startswith("/api/tasks/") for p in paths)


def test_sampling_submits(client):
    r = client.post("/api/sampling", data={"generator": "reinvent", "num_smiles": "10"})
    assert r.status_code == 200, r.text
    assert "job_id" in r.json()


def test_scoring_submits_with_json_field(client):
    r = client.post(
        "/api/scoring",
        data={"scoring": '{"type":"geometric_mean","component":[]}'},
        files={"smiles_file": ("c.smi", b"CCO\n", "text/plain")},
    )
    assert r.status_code == 200, r.text


def test_staged_learning_submits_with_json_stages(client):
    r = client.post(
        "/api/staged-learning",
        data={"generator": "reinvent",
              "stages": '[{"chkpt_name":"s1.chkpt","scoring":{"type":"geometric_mean","component":[]}}]'},
    )
    assert r.status_code == 200, r.text


def test_sampling_task_endpoint_registered(client):
    # Task endpoints run synchronously; subprocess exec fails fast (no reinvent
    # bin in test env) -> 200 with status FAILED, but the route must exist (not 404).
    r = client.post("/api/tasks/sampling", data={"generator": "reinvent", "num_smiles": "5"})
    assert r.status_code != 404, r.text

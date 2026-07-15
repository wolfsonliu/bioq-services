"""Offline tests for lightdock-server (no real lightdock binary needed).

`conftest.py` registers the service dir as the `server` package so
`from server.settings import ...` works without a pip install. The docking
driver is stubbed by pointing LIGHTDOCK_PYTHON at /bin/true so submit returns a
job without running the real multi-step protocol.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

DATA_DIR = Path(__file__).resolve().parent / "data"
RECEPTOR = DATA_DIR / "receptor.pdb"
LIGAND = DATA_DIR / "ligand.pdb"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTDOCK_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("LIGHTDOCK_ROOT", str(tmp_path / "lightdock"))
    # Stub the driver interpreter so submit doesn't try to run the real pipeline.
    monkeypatch.setenv("LIGHTDOCK_PYTHON", "/bin/true")
    (tmp_path / "lightdock").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Healthcheck / manifest -----


def test_health(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "lightdock"
    assert "version" in body


def test_healthz_detail(client):
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "lightdock"
    # lightdock is not installed in the offline test env → not available, but
    # the endpoint must still return 200 with the readiness fields present.
    assert "lightdock_version" in body
    assert "scoring_functions" in body
    assert isinstance(body["scoring_functions"], list)


def test_manifest_service_name(client):
    assert client.get("/api/manifest").json()["service"] == "lightdock"


def test_manifest_lists_dock_endpoint(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/dock" in paths, f"missing /api/dock in {paths}"


def test_manifest_extras_has_tool_outputs(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "dock" in extras["tool_outputs"]
    assert "input_uri_schemes" in extras


def test_openapi_registers_task_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/tasks/dock" in paths, "task endpoint /api/tasks/dock not registered"


# ----- Settings -----


def test_settings_defaults():
    from server.settings import LightdockSettings

    class _Off(LightdockSettings):
        model_config = SettingsConfigDict(
            env_prefix="LIGHTDOCK_TEST_", env_file=None, extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/lightdock_jobs")
    assert s.root == Path("/opt/lightdock")
    assert s.bin_dir == Path("/opt/lightdock/.venv/bin")
    assert s.default_scoring == "fastdfire"
    assert s.default_cores == 8
    assert s.max_concurrent_jobs == 1


# ----- Endpoint smoke (submit returns a job; no real subprocess asserted) -----


def test_dock_returns_job(client):
    with open(RECEPTOR, "rb") as fr, open(LIGAND, "rb") as fl:
        resp = client.post(
            "/api/dock",
            files={
                "receptor": ("receptor.pdb", fr.read(), "chemical/x-pdb"),
                "ligand": ("ligand.pdb", fl.read(), "chemical/x-pdb"),
            },
            data={"swarms": "2", "glowworms": "5", "steps": "3", "top": "3"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["input_params"]["swarms"] == 2
    assert body["input_params"]["scoring_function"] == "fastdfire"


def test_dock_rejects_missing_receptor(client):
    with open(LIGAND, "rb") as fl:
        resp = client.post(
            "/api/dock",
            files={"ligand": ("ligand.pdb", fl.read(), "chemical/x-pdb")},
        )
    assert resp.status_code == 422


def test_dock_rejects_missing_ligand(client):
    with open(RECEPTOR, "rb") as fr:
        resp = client.post(
            "/api/dock",
            files={"receptor": ("receptor.pdb", fr.read(), "chemical/x-pdb")},
        )
    assert resp.status_code == 422


def test_dock_rejects_bad_scoring_function(client):
    with open(RECEPTOR, "rb") as fr, open(LIGAND, "rb") as fl:
        resp = client.post(
            "/api/dock",
            files={
                "receptor": ("receptor.pdb", fr.read(), "chemical/x-pdb"),
                "ligand": ("ligand.pdb", fl.read(), "chemical/x-pdb"),
            },
            data={"scoring_function": "Not A Function!"},
        )
    assert resp.status_code == 422

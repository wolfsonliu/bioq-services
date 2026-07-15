"""Offline tests for haddock3-server (no real haddock3 / CNS needed).

`conftest.py` registers the service dir as the `server` package so
`from server.settings import ...` works without a pip install.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

DATA_DIR = Path(__file__).resolve().parent / "data"
COMPLEX_PDB = DATA_DIR / "complex.pdb"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HADDOCK3_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("HADDOCK3_ROOT", str(tmp_path / "haddock3"))
    (tmp_path / "haddock3").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Healthcheck / manifest -----


def test_health(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "haddock3"
    assert "version" in body


def test_healthz_detail_reports_cns(client):
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "haddock3"
    # No CNS binary staged in the test env → not available, but the endpoint
    # must still return 200 and restraints must be flagged available.
    assert body["cns_available"] is False
    assert body["weights_loaded"] is False
    assert body["restraints_available"] is True


def test_manifest_service_name(client):
    assert client.get("/api/manifest").json()["service"] == "haddock3"


def test_manifest_lists_all_endpoints(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    for p in (
        "/api/dock",
        "/api/dock/protein-protein",
        "/api/score",
        "/api/restraints/restrain-bodies",
        "/api/restraints/active-passive-to-ambig",
    ):
        assert p in paths, f"missing {p} in {paths}"


def test_manifest_extras_has_tool_outputs(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    for key in ("dock", "score", "restrain-bodies", "actpass-to-ambig"):
        assert key in extras["tool_outputs"]


def test_openapi_registers_task_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in (
        "/api/tasks/dock",
        "/api/tasks/dock/protein-protein",
        "/api/tasks/score",
        "/api/tasks/restraints/restrain-bodies",
        "/api/tasks/restraints/active-passive-to-ambig",
    ):
        assert p in paths, f"task endpoint {p} not registered"


# ----- Settings -----


def test_settings_defaults():
    from server.settings import Haddock3Settings

    class _Off(Haddock3Settings):
        model_config = SettingsConfigDict(
            env_prefix="HADDOCK3_TEST_", env_file=None, extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/haddock3_jobs")
    assert s.root == Path("/opt/haddock3")
    assert s.weights_dir == Path("/data/models/haddock3")
    assert s.cns_exec == Path("/data/models/haddock3/cns/cns")
    assert s.default_ncores == 8


# ----- Endpoint smoke (submit returns a job; no real subprocess asserted) -----


def test_restrain_bodies_returns_job(client):
    with open(COMPLEX_PDB, "rb") as fh:
        resp = client.post(
            "/api/restraints/restrain-bodies",
            files={"structure": ("complex.pdb", fh, "chemical/x-pdb")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["input_params"] is not None


def test_actpass_to_ambig_returns_job(client):
    with open(DATA_DIR / "a.actpass", "rb") as fa, open(DATA_DIR / "b.actpass", "rb") as fb:
        resp = client.post(
            "/api/restraints/active-passive-to-ambig",
            files={
                "actpass1": ("a.actpass", fa.read(), "text/plain"),
                "actpass2": ("b.actpass", fb.read(), "text/plain"),
            },
            data={"segid1": "A", "segid2": "B"},
        )
    assert resp.status_code == 200, resp.text
    assert "job_id" in resp.json()


def test_score_returns_job(client):
    with open(COMPLEX_PDB, "rb") as fh:
        resp = client.post(
            "/api/score",
            files={"complex": ("complex.pdb", fh, "chemical/x-pdb")},
            data={"full": "true"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["input_params"]["full"] is True


def test_restrain_bodies_rejects_missing_input(client):
    resp = client.post("/api/restraints/restrain-bodies", data={})
    assert resp.status_code in (400, 422)

"""Offline tests for deeprank-ab-server (no real DeepRank-Ab pipeline needed)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPRANK_AB_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("DEEPRANK_AB_ROOT", str(tmp_path / "deeprank-ab"))
    monkeypatch.setenv("DEEPRANK_AB_PYTHON", "/bin/true")
    monkeypatch.setenv("DEEPRANK_AB_INFERENCE_SCRIPT", "/bin/true")
    (tmp_path / "deeprank-ab").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Healthcheck / manifest -----

def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "deeprank-ab"
    assert "version" in health
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "deeprank-ab"
    assert detail["version"] == health["version"]


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "deeprank-ab"


def test_manifest_lists_score_endpoint(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/score" in paths


def test_manifest_extras_has_tool_outputs(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "score" in extras["tool_outputs"]


def test_manifest_extras_has_scoring_legend(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    legend = extras["scoring_legend"]
    assert "predicted_dockq" in legend
    assert "quality_flag" in legend


def test_manifest_extras_has_model_info(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    info = extras["model_info"]
    assert "EGNN" in info["architecture"]
    assert "ESM-2" in info["sequence_encoder"]


def test_endpoint_examples_present(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert by_path["/api/score"]["examples"]


# ----- Settings -----

def test_settings_defaults():
    from server.settings import DeepRankAbSettings

    class _Off(DeepRankAbSettings):
        model_config = SettingsConfigDict(env_prefix="DEEPRANK_AB_TEST_", env_file=None, extra="ignore")

    s = _Off()
    assert s.jobs_base_dir == Path("/data/deeprank_ab_jobs")
    assert s.root == Path("/opt/deeprank-ab")
    assert s.python == "/opt/conda/envs/deeprank-ab/bin/python"
    assert s.oss_region == "cn-hangzhou"


def test_settings_env_override(monkeypatch):
    from server.settings import DeepRankAbSettings
    monkeypatch.setenv("DEEPRANK_AB_PYTHON", "/custom/python")
    monkeypatch.setenv("DEEPRANK_AB_INFERENCE_SCRIPT", "/custom/inference.py")
    s = DeepRankAbSettings()
    assert s.python == "/custom/python"
    assert s.inference_script == "/custom/inference.py"


# ----- Adapter -----

def test_adapter_name():
    from server.adapter import DeepRankAbAdapter
    from server.settings import DeepRankAbSettings

    class _Off(DeepRankAbSettings):
        model_config = SettingsConfigDict(env_prefix="DEEPRANK_AB_TEST_", env_file=None, extra="ignore")

    a = DeepRankAbAdapter(settings=_Off())
    assert a.name == "deeprank-ab"


def test_detect_outputs_csv(tmp_path):
    from server.adapter import DeepRankAbAdapter
    from server.settings import DeepRankAbSettings

    class _Off(DeepRankAbSettings):
        model_config = SettingsConfigDict(env_prefix="DEEPRANK_AB_TEST_", env_file=None, extra="ignore")

    a = DeepRankAbAdapter(settings=_Off())
    job = tmp_path / "j"
    workspace = job / "output" / "test-deeprank_ab_pred_HL_A"
    workspace.mkdir(parents=True)
    (workspace / "test-deeprank_ab_pred_HL_A_predictions.csv").write_text(
        "pdb_id,predicted_dockq,quality_flag\nmodel_0,0.65,ok\n"
    )
    assert a.detect_outputs(job) is True


def test_detect_outputs_empty(tmp_path):
    from server.adapter import DeepRankAbAdapter
    from server.settings import DeepRankAbSettings

    class _Off(DeepRankAbSettings):
        model_config = SettingsConfigDict(env_prefix="DEEPRANK_AB_TEST_", env_file=None, extra="ignore")

    a = DeepRankAbAdapter(settings=_Off())
    job = tmp_path / "j"
    (job / "output").mkdir(parents=True)
    assert a.detect_outputs(job) is False


def test_subprocess_env():
    from server.adapter import DeepRankAbAdapter
    from server.settings import DeepRankAbSettings

    class _Off(DeepRankAbSettings):
        model_config = SettingsConfigDict(env_prefix="DEEPRANK_AB_TEST_", env_file=None, extra="ignore")

    a = DeepRankAbAdapter(settings=_Off())
    env = a.subprocess_env()
    assert "PYTHONPATH" in env
    assert "DeepRank-Ab" in env["PYTHONPATH"]


# ----- Request models -----

def test_score_request_defaults():
    from server.models import ScoreRequest
    r = ScoreRequest()
    assert r.heavy_chain_id == "H"
    assert r.light_chain_id == "L"
    assert r.antigen_chain_id == "A"


def test_score_request_nanobody():
    from server.models import ScoreRequest
    r = ScoreRequest(light_chain_id="-")
    assert r.light_chain_id == "-"


def test_score_request_custom_chains():
    from server.models import ScoreRequest
    r = ScoreRequest(heavy_chain_id="B", light_chain_id="C", antigen_chain_id="D")
    assert r.heavy_chain_id == "B"
    assert r.light_chain_id == "C"
    assert r.antigen_chain_id == "D"


# ----- argv builders -----

def test_score_argv_basic(tmp_path):
    from server.models import ScoreRequest
    from server.settings import DeepRankAbSettings
    from server.tools import score_argv

    class _Off(DeepRankAbSettings):
        model_config = SettingsConfigDict(env_prefix="DEEPRANK_AB_TEST_", env_file=None, extra="ignore")

    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    argv = score_argv(
        ScoreRequest(),
        job_dir=job_dir,
        pdb_path=tmp_path / "input.pdb",
        settings=s,
    )
    assert argv[0] == "bash"
    assert argv[1] == "-c"
    cmd = argv[2]
    assert "cd " in cmd
    assert str(job_dir / "output") in cmd
    assert "inference.py" in cmd
    assert " H " in cmd or "'H'" in cmd
    assert " L " in cmd or "'L'" in cmd
    assert " A" in cmd or "'A'" in cmd


def test_score_argv_nanobody(tmp_path):
    from server.models import ScoreRequest
    from server.settings import DeepRankAbSettings
    from server.tools import score_argv

    class _Off(DeepRankAbSettings):
        model_config = SettingsConfigDict(env_prefix="DEEPRANK_AB_TEST_", env_file=None, extra="ignore")

    argv = score_argv(
        ScoreRequest(light_chain_id="-"),
        job_dir=tmp_path,
        pdb_path=tmp_path / "vhh.pdb",
        settings=_Off(),
    )
    cmd = argv[2]
    assert "'-'" in cmd or " - " in cmd


# ----- URI resolution -----

def test_uri_resolve_file(tmp_path):
    from server.settings import DeepRankAbSettings
    from server.uris import resolve_input

    class _Off(DeepRankAbSettings):
        model_config = SettingsConfigDict(env_prefix="DEEPRANK_AB_TEST_", env_file=None, extra="ignore")

    src = tmp_path / "src.pdb"
    src.write_text("ATOM\n")
    dest = tmp_path / "dest.pdb"
    out = resolve_input(None, f"file://{src}", dest, _Off())
    assert out.read_text() == "ATOM\n"


def test_uri_requires_input(tmp_path):
    from fastapi import HTTPException
    from server.settings import DeepRankAbSettings
    from server.uris import resolve_input

    class _Off(DeepRankAbSettings):
        model_config = SettingsConfigDict(env_prefix="DEEPRANK_AB_TEST_", env_file=None, extra="ignore")

    with pytest.raises(HTTPException) as exc:
        resolve_input(None, None, tmp_path / "x", _Off())
    assert exc.value.status_code == 422


# ----- endpoint smoke (no real pipeline) -----

def test_score_endpoint_returns_job(client):
    data_dir = Path(__file__).resolve().parent / "data"
    with open(data_dir / "test.pdb", "rb") as fh:
        resp = client.post(
            "/api/score",
            data={
                "heavy_chain_id": "H",
                "light_chain_id": "L",
                "antigen_chain_id": "A",
            },
            files={
                "input_pdb": ("test.pdb", fh, "chemical/x-pdb"),
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["created_at"] is not None
    assert body["input_params"] is not None
    assert body["input_params"]["heavy_chain_id"] == "H"
    assert body["input_params"]["light_chain_id"] == "L"
    assert body["input_params"]["antigen_chain_id"] == "A"

"""Offline tests for promera-server (no real algorithm / GPU needed).

``conftest.py`` registers the service dir as ``server`` package, so
``from server.settings import ...`` works without pip install.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_TARGET = DATA_DIR / "test_target.json"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMERA_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("PROMERA_ROOT", str(tmp_path / "promera"))
    monkeypatch.setenv("PROMERA_PYTHON", "/bin/true")
    monkeypatch.setenv("PROMERA_WEIGHTS", str(tmp_path / "weights.ckpt"))
    monkeypatch.setenv("PROMERA_LIGANDMPNN_DIR", str(tmp_path / "lmpnn"))
    (tmp_path / "promera").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Healthcheck / manifest -----


def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "promera"
    assert "version" in health


def test_health_detail(client):
    detail = client.get("/healthz/detail").json()
    assert detail["status"] == "ok"
    assert "active_jobs" in detail


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "promera"


def test_manifest_lists_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/cofold" in paths
    assert "/api/design" in paths


def test_manifest_extras_has_tool_outputs(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "cofold" in extras["tool_outputs"]
    assert "design" in extras["tool_outputs"]


def test_manifest_extras_has_design_types(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "minibinder" in extras["design_types"]
    assert "vhh" in extras["design_types"]


# ----- Settings -----


def test_settings_defaults():
    from server.settings import PromeraSettings

    class _Off(PromeraSettings):
        model_config = SettingsConfigDict(
            env_prefix="PROMERA_TEST_", env_file=None, extra="ignore"
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/promera_jobs")
    assert s.root == Path("/opt/promera")
    assert s.max_concurrent_jobs == 1


def test_settings_env_override(monkeypatch):
    from server.settings import PromeraSettings

    monkeypatch.setenv("PROMERA_X_MAX_CONCURRENT_JOBS", "4")

    class _Off(PromeraSettings):
        model_config = SettingsConfigDict(
            env_prefix="PROMERA_X_", env_file=None, extra="ignore"
        )

    s = _Off()
    assert s.max_concurrent_jobs == 4


# ----- Adapter -----


def test_adapter_name():
    from server.adapter import PromeraAdapter
    from server.settings import PromeraSettings

    class _Off(PromeraSettings):
        model_config = SettingsConfigDict(
            env_prefix="PROMERA_T2_", env_file=None, extra="ignore"
        )

    adapter = PromeraAdapter(settings=_Off())
    assert adapter.name == "promera"


def test_adapter_detect_outputs_empty(tmp_path):
    from server.adapter import PromeraAdapter
    from server.settings import PromeraSettings

    class _Off(PromeraSettings):
        model_config = SettingsConfigDict(
            env_prefix="PROMERA_T3_", env_file=None, extra="ignore"
        )

    adapter = PromeraAdapter(settings=_Off())
    job_dir = tmp_path / "job1"
    (job_dir / "output").mkdir(parents=True)
    assert not adapter.detect_outputs(job_dir)


def test_adapter_detect_outputs_with_cif(tmp_path):
    from server.adapter import PromeraAdapter
    from server.settings import PromeraSettings

    class _Off(PromeraSettings):
        model_config = SettingsConfigDict(
            env_prefix="PROMERA_T4_", env_file=None, extra="ignore"
        )

    adapter = PromeraAdapter(settings=_Off())
    job_dir = tmp_path / "job2"
    out = job_dir / "output" / "test"
    out.mkdir(parents=True)
    (out / "test_seed0_samp0.cif").write_text("data_test\n")
    assert adapter.detect_outputs(job_dir)


# ----- tools.py -----


def test_cofold_argv_structure():
    from server.models import CofoldRequest
    from server.settings import PromeraSettings
    from server.tools import cofold_argv

    class _Off(PromeraSettings):
        model_config = SettingsConfigDict(
            env_prefix="PROMERA_T5_", env_file=None, extra="ignore"
        )

    s = _Off()
    req = CofoldRequest(num_seeds=2, diffusion_samples=3)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        job_dir = Path(td) / "job"
        job_dir.mkdir()
        schema_path = Path(td) / "input" / "input.json"
        schema_path.parent.mkdir()
        schema_path.write_text("{}")

        argv = cofold_argv(req, job_dir=job_dir, schema_path=schema_path, settings=s)

    assert argv[0] == s.python
    assert "-m" in argv
    assert "promera" in argv
    assert "--weights" in argv
    assert any("num_seeds=2" in a for a in argv)
    assert any("diffusion_samples=3" in a for a in argv)
    assert any("assert_msa=false" in a for a in argv)


def test_design_argv_structure():
    from server.models import DesignRequest
    from server.settings import PromeraSettings
    from server.tools import build_design_config, design_argv, write_design_config

    class _Off(PromeraSettings):
        model_config = SettingsConfigDict(
            env_prefix="PROMERA_T6_", env_file=None, extra="ignore"
        )

    s = _Off()
    req = DesignRequest(design_type="vhh", num_backbones=5)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        job_dir = Path(td) / "job"
        target_dir = job_dir / "input" / "targets"
        output_dir = job_dir / "output"
        target_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        cfg = build_design_config(
            req, target_dir=target_dir, output_dir=output_dir, settings=s
        )
        config_path = write_design_config(cfg, job_dir / "input" / "task_config.yaml")
        argv = design_argv(req, job_dir=job_dir, config_path=config_path, settings=s)

    assert "--task" in argv
    assert "promera.inference.Design" in argv
    assert "--task_config" in argv


def test_build_design_config_minibinder():
    from server.models import DesignRequest
    from server.settings import PromeraSettings
    from server.tools import build_design_config

    class _Off(PromeraSettings):
        model_config = SettingsConfigDict(
            env_prefix="PROMERA_T7_", env_file=None, extra="ignore"
        )

    s = _Off()
    req = DesignRequest(
        design_type="minibinder",
        num_backbones=20,
        binder_length_min=50,
        binder_length_max=80,
        epitope_residues="10,20,30",
    )
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        target_dir = Path(td) / "targets"
        output_dir = Path(td) / "output"
        target_dir.mkdir()
        output_dir.mkdir()
        cfg = build_design_config(
            req, target_dir=target_dir, output_dir=output_dir, settings=s
        )

    assert cfg["binder"]["type"] == "protein"
    assert cfg["binder"]["length"] == [50, 80]
    assert cfg["num_backbones"] == 20
    assert cfg["epitope_residues"] == [10, 20, 30]
    assert cfg["inverse_folder"]["type"] == "solublempnn"


def test_build_design_config_vhh():
    from server.models import DesignRequest
    from server.settings import PromeraSettings
    from server.tools import build_design_config

    class _Off(PromeraSettings):
        model_config = SettingsConfigDict(
            env_prefix="PROMERA_T8_", env_file=None, extra="ignore"
        )

    s = _Off()
    req = DesignRequest(design_type="vhh", num_backbones=5)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        target_dir = Path(td) / "targets"
        output_dir = Path(td) / "output"
        target_dir.mkdir()
        output_dir.mkdir()
        cfg = build_design_config(
            req, target_dir=target_dir, output_dir=output_dir, settings=s
        )

    assert cfg["binder"]["type"] == "vhh"
    assert "framework" in cfg["binder"]
    assert "cdr_lengths" in cfg["binder"]
    assert cfg["binder"]["paratope_from_cdrs"] is True


# ----- Endpoint smoke (no real pipeline) -----


def test_cofold_returns_job(client):
    with open(TEST_TARGET, "rb") as fh:
        resp = client.post(
            "/api/cofold",
            files={"input_schema": ("test.json", fh, "application/json")},
            data={"num_seeds": "1", "diffusion_samples": "1"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["input_params"] is not None
    assert body["input_params"]["num_seeds"] == 1


def test_design_returns_job(client):
    with open(TEST_TARGET, "rb") as fh:
        resp = client.post(
            "/api/design",
            files={"target_schema": ("target.json", fh, "application/json")},
            data={
                "design_type": "minibinder",
                "num_backbones": "3",
                "binder_length_min": "50",
                "binder_length_max": "70",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["input_params"]["design_type"] == "minibinder"
    assert body["input_params"]["num_backbones"] == 3


def test_design_vhh_returns_job(client):
    with open(TEST_TARGET, "rb") as fh:
        resp = client.post(
            "/api/design",
            files={"target_schema": ("target.json", fh, "application/json")},
            data={"design_type": "vhh", "num_backbones": "2"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["input_params"]["design_type"] == "vhh"


def test_cofold_requires_input(client):
    resp = client.post("/api/cofold", data={"num_seeds": "1"})
    assert resp.status_code == 422

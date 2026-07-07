"""Offline tests for flowmol-server.

Real FlowMol sampling never runs offline — the subprocess is stubbed via
FLOWMOL_PYTHON=/bin/true so no GPU / weights needed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


def _touch_variant(weights_dir: Path, variant: str) -> None:
    """Create empty checkpoints/last.ckpt + config.yaml for a variant."""
    d = weights_dir / "trained_models" / variant / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    (d / "last.ckpt").write_bytes(b"\x00")
    (weights_dir / "trained_models" / variant / "config.yaml").write_text("dataset: geom\n")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Reload the app under a sandbox of tmp_path-based dirs + stubbed python."""
    monkeypatch.setenv("FLOWMOL_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("FLOWMOL_ROOT", str(tmp_path / "flowmol"))
    monkeypatch.setenv("FLOWMOL_PYTHON", "/bin/true")
    monkeypatch.setenv("FLOWMOL_INFERENCE_SCRIPT", str(tmp_path / "inference.py"))
    monkeypatch.setenv("FLOWMOL_WEIGHTS_DIR", str(tmp_path / "weights"))
    (tmp_path / "flowmol").mkdir(parents=True, exist_ok=True)
    # Stage 2 primary variants so healthz shows partial weights_loaded.
    _touch_variant(tmp_path / "weights", "flowmol3")
    _touch_variant(tmp_path / "weights", "fm3_nodistort")

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Health / manifest -----


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "flowmol"
    assert "version" in body


def test_healthz_detail_reports_weights(client):
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "flowmol"
    # Only 2 of 4 primaries staged → weights_loaded=False.
    assert body["weights_loaded"] is False
    assert "fm3_none_ckpt" in body["weights_missing"]
    assert "fm3_ahigh_ckpt" in body["weights_missing"]
    assert "flowmol3" in body["staged_variants"]
    assert "fm3_nodistort" in body["staged_variants"]


def test_healthz_detail_all_primaries_ok(tmp_path, monkeypatch):
    """Sanity: staging all 4 primary variants flips weights_loaded=true."""
    monkeypatch.setenv("FLOWMOL_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("FLOWMOL_ROOT", str(tmp_path / "flowmol"))
    monkeypatch.setenv("FLOWMOL_PYTHON", "/bin/true")
    monkeypatch.setenv("FLOWMOL_WEIGHTS_DIR", str(tmp_path / "weights"))
    (tmp_path / "flowmol").mkdir(parents=True, exist_ok=True)
    for v in ("flowmol3", "fm3_nodistort", "fm3_none", "fm3_ahigh"):
        _touch_variant(tmp_path / "weights", v)

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    c = TestClient(server_app.app)
    body = c.get("/healthz/detail").json()
    assert body["weights_loaded"] is True
    assert body["weights_missing"] == {}


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "flowmol"


def test_manifest_lists_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/generate" in paths


def test_manifest_extras_have_model_info(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "FlowMol3"
    assert "flowmol3" in extras["model"]["primary_variants"]
    assert "generate" in extras["tool_outputs"]


def test_manifest_extras_have_endpoint_examples(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    examples = by_path["/api/generate"]["examples"]
    assert len(examples) >= 2
    assert any("n_mols" in (e.get("curl") or "") for e in examples)


def test_openapi_served(client):
    r = client.get("/openapi.json")
    r.raise_for_status()
    paths = r.json()["paths"]
    assert "/api/generate" in paths
    assert "/api/tasks/generate" in paths


# ----- Validation errors -----


def test_generate_invalid_variant_returns_422(client):
    r = client.post("/api/generate", data={"model_variant": "not_a_real_variant"})
    assert r.status_code == 422


def test_generate_n_mols_out_of_range_returns_422(client):
    r = client.post("/api/generate", data={"n_mols": "0"})
    assert r.status_code == 422


def test_generate_n_mols_too_high_returns_422(client):
    r = client.post("/api/generate", data={"n_mols": "10000"})
    assert r.status_code == 422


def test_generate_n_timesteps_out_of_range_returns_422(client):
    r = client.post("/api/generate", data={"n_timesteps": "10"})
    assert r.status_code == 422


def test_generate_hc_thresh_out_of_range_returns_422(client):
    r = client.post("/api/generate", data={"hc_thresh": "1.5"})
    assert r.status_code == 422


# ----- Smoke (subprocess stubbed via /bin/true) -----


def test_generate_returns_job_with_input_params(client):
    r = client.post(
        "/api/generate",
        data={
            "n_mols": "50",
            "n_timesteps": "100",
            "model_variant": "flowmol3",
            "seed": "42",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["n_mols"] == 50
    assert body["input_params"]["n_timesteps"] == 100
    assert body["input_params"]["model_variant"] == "flowmol3"
    assert body["input_params"]["seed"] == 42


def test_generate_defaults_survive(client):
    """POST with no body should use pydantic defaults + submit."""
    r = client.post("/api/generate", data={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_params"]["n_mols"] == 100
    assert body["input_params"]["n_timesteps"] == 250
    assert body["input_params"]["model_variant"] == "flowmol3"


# ----- Settings -----


def test_settings_defaults():
    from server.settings import FlowMolSettings

    class _Off(FlowMolSettings):
        model_config = SettingsConfigDict(
            env_prefix="FLOWMOL_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/flowmol_jobs")
    assert s.root == Path("/opt/flowmol")
    # Weights externalized to NAS — default points at the FC mount path.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    assert s.weights_dir == Path("/data/models/flowmol")
    assert s.max_concurrent_jobs == 1
    assert s.default_variant == "flowmol3"


def test_settings_env_override(monkeypatch):
    from server.settings import FlowMolSettings
    monkeypatch.setenv("FLOWMOL_PYTHON", "/custom/python")
    monkeypatch.setenv("FLOWMOL_WEIGHTS_DIR", "/mnt/scratch/weights")
    s = FlowMolSettings()
    assert s.python == "/custom/python"
    assert s.weights_dir == Path("/mnt/scratch/weights")


# ----- tools.argv builder -----


def test_generate_argv_minimal_flags(tmp_path):
    from server.models import GenerateRequest
    from server.settings import FlowMolSettings
    from server.tools import generate_argv

    class _Off(FlowMolSettings):
        model_config = SettingsConfigDict(
            env_prefix="FLOWMOL_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off(
        python="/opt/foo/python",
        inference_script="/opt/foo/inference.py",
        weights_dir=tmp_path / "weights",
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    argv = generate_argv(
        GenerateRequest(n_mols=50, n_timesteps=100, model_variant="fm3_ahigh"),
        job_dir=job_dir,
        settings=s,
    )
    assert argv[0] == "/opt/foo/python"
    assert "/opt/foo/inference.py" in argv
    # Model dir points at the variant directory under NAS.
    assert "--model-dir" in argv
    idx = argv.index("--model-dir")
    assert argv[idx + 1] == str(s.weights_dir / "trained_models" / "fm3_ahigh")
    assert "--output-file" in argv
    assert str(job_dir / "output" / "molecules.sdf") in argv
    assert "--stats-file" in argv
    assert str(job_dir / "output" / "sampling_stats.json") in argv
    assert "--n-mols" in argv and "50" in argv
    assert "--n-timesteps" in argv and "100" in argv
    # Optional flags not present when their pydantic default is None.
    assert "--seed" not in argv
    assert "--stochasticity" not in argv
    assert "--hc-thresh" not in argv
    assert "--n-atoms-per-mol" not in argv


def test_generate_argv_includes_optional_flags(tmp_path):
    from server.models import GenerateRequest
    from server.settings import FlowMolSettings
    from server.tools import generate_argv

    class _Off(FlowMolSettings):
        model_config = SettingsConfigDict(
            env_prefix="FLOWMOL_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    argv = generate_argv(
        GenerateRequest(
            n_mols=10, n_timesteps=250, model_variant="flowmol3",
            n_atoms_per_mol=25, seed=42, stochasticity=1.5, hc_thresh=0.7,
            max_batch_size=32,
        ),
        job_dir=job_dir,
        settings=s,
    )
    assert "--seed" in argv and "42" in argv
    assert "--stochasticity" in argv and "1.5" in argv
    assert "--hc-thresh" in argv and "0.7" in argv
    assert "--n-atoms-per-mol" in argv and "25" in argv
    assert "--max-batch-size" in argv and "32" in argv

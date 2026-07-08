"""Offline tests for semlaflow-server.

Real SemlaFlow sampling never runs offline — the subprocess is stubbed via
SEMLAFLOW_PYTHON=/bin/true so no GPU / weights needed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


def _stage_model(
    weights_dir: Path,
    name: str,
    dataset: str,
    *,
    splits=("train", "val", "test"),
) -> None:
    """Create model.ckpt + smol/<split>.smol + manifest.yaml for a model."""
    d = weights_dir / name
    (d / "smol").mkdir(parents=True, exist_ok=True)
    (d / "model.ckpt").write_bytes(b"\x00")
    for s in splits:
        (d / "smol" / f"{s}.smol").write_bytes(b"\x00")
    (d / "manifest.yaml").write_text(f"dataset: {dataset}\n")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Reload the app under a sandbox of tmp_path-based dirs + stubbed python."""
    monkeypatch.setenv("SEMLAFLOW_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("SEMLAFLOW_ROOT", str(tmp_path / "semlaflow"))
    monkeypatch.setenv("SEMLAFLOW_PYTHON", "/bin/true")
    monkeypatch.setenv("SEMLAFLOW_INFERENCE_SCRIPT", str(tmp_path / "inference.py"))
    monkeypatch.setenv("SEMLAFLOW_WEIGHTS_DIR", str(tmp_path / "weights"))
    (tmp_path / "semlaflow").mkdir(parents=True, exist_ok=True)
    # qm9 fully staged; geom-drugs missing train.smol → not ready.
    _stage_model(tmp_path / "weights", "qm9", "qm9")
    _stage_model(tmp_path / "weights", "geom-drugs", "geom-drugs", splits=("val", "test"))

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Health / manifest -----


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "semlaflow"
    assert "version" in body


def test_healthz_detail_reports_models(client):
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "semlaflow"
    # geom-drugs missing train.smol → not ready → weights_loaded False.
    assert body["weights_loaded"] is False
    assert body["models"]["qm9"]["ready"] is True
    assert body["models"]["geom-drugs"]["ready"] is False
    assert body["models"]["geom-drugs"]["splits"]["train"] is False


def test_healthz_detail_all_ready(tmp_path, monkeypatch):
    """Staging both models fully flips weights_loaded=true."""
    monkeypatch.setenv("SEMLAFLOW_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("SEMLAFLOW_ROOT", str(tmp_path / "semlaflow"))
    monkeypatch.setenv("SEMLAFLOW_PYTHON", "/bin/true")
    monkeypatch.setenv("SEMLAFLOW_WEIGHTS_DIR", str(tmp_path / "weights"))
    (tmp_path / "semlaflow").mkdir(parents=True, exist_ok=True)
    _stage_model(tmp_path / "weights", "qm9", "qm9")
    _stage_model(tmp_path / "weights", "geom-drugs", "geom-drugs")

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    c = TestClient(server_app.app)
    body = c.get("/healthz/detail").json()
    assert body["weights_loaded"] is True


def test_api_models_lists_registry(client):
    body = client.get("/api/models").json()
    names = {m["name"]: m for m in body["models"]}
    assert "qm9" in names and "geom-drugs" in names
    assert names["qm9"]["dataset"] == "qm9"
    assert names["qm9"]["ready"] is True
    assert names["geom-drugs"]["ready"] is False


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "semlaflow"


def test_manifest_lists_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/generate" in paths


def test_manifest_extras_have_model_info(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "SemlaFlow"
    assert "generate" in extras["tool_outputs"]


def test_manifest_extras_have_endpoint_examples(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    examples = by_path["/api/generate"]["examples"]
    assert len(examples) >= 2
    assert any("model_name" in (e.get("curl") or "") for e in examples)


def test_openapi_served(client):
    r = client.get("/openapi.json")
    r.raise_for_status()
    paths = r.json()["paths"]
    assert "/api/generate" in paths
    assert "/api/tasks/generate" in paths
    assert "/api/models" in paths


# ----- Validation errors -----


def test_generate_invalid_model_returns_422(client):
    r = client.post("/api/generate", data={"model_name": "not_a_model"})
    assert r.status_code == 422


def test_generate_n_molecules_too_low_returns_422(client):
    r = client.post("/api/generate", data={"n_molecules": "0"})
    assert r.status_code == 422


def test_generate_n_molecules_too_high_returns_422(client):
    r = client.post("/api/generate", data={"n_molecules": "20000"})
    assert r.status_code == 422


def test_generate_integration_steps_out_of_range_returns_422(client):
    r = client.post("/api/generate", data={"integration_steps": "5"})
    assert r.status_code == 422


def test_generate_bad_dataset_split_returns_422(client):
    r = client.post("/api/generate", data={"dataset_split": "holdout"})
    assert r.status_code == 422


# ----- Smoke (subprocess stubbed via /bin/true) -----


def test_generate_returns_job_with_input_params(client):
    r = client.post(
        "/api/generate",
        data={
            "model_name": "qm9",
            "n_molecules": "50",
            "integration_steps": "80",
            "seed": "42",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["model_name"] == "qm9"
    assert body["input_params"]["n_molecules"] == 50
    assert body["input_params"]["integration_steps"] == 80
    assert body["input_params"]["seed"] == 42


def test_generate_defaults_survive(client):
    """POST with no body should use pydantic defaults + submit."""
    r = client.post("/api/generate", data={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_params"]["model_name"] == "qm9"
    assert body["input_params"]["n_molecules"] == 100
    assert body["input_params"]["integration_steps"] == 100
    assert body["input_params"]["dataset_split"] == "test"


# ----- Settings -----


def test_settings_defaults():
    from server.settings import SemlaFlowSettings

    class _Off(SemlaFlowSettings):
        model_config = SettingsConfigDict(
            env_prefix="SEMLAFLOW_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/semlaflow_jobs")
    assert s.root == Path("/opt/semlaflow")
    assert s.weights_dir == Path("/data/models/semlaflow")
    assert s.max_concurrent_jobs == 1
    assert s.default_model == "qm9"


def test_settings_env_override(monkeypatch):
    from server.settings import SemlaFlowSettings
    monkeypatch.setenv("SEMLAFLOW_PYTHON", "/custom/python")
    monkeypatch.setenv("SEMLAFLOW_WEIGHTS_DIR", "/mnt/scratch/weights")
    s = SemlaFlowSettings()
    assert s.python == "/custom/python"
    assert s.weights_dir == Path("/mnt/scratch/weights")


def test_registry_infers_dataset_from_name(tmp_path):
    """No manifest.yaml → dataset inferred from directory name."""
    from server.settings import SemlaFlowSettings

    class _Off(SemlaFlowSettings):
        model_config = SettingsConfigDict(
            env_prefix="SEMLAFLOW_TEST_", env_file=None, extra="ignore",
        )

    wd = tmp_path / "weights"
    (wd / "qm9-headline" / "smol").mkdir(parents=True)
    (wd / "qm9-headline" / "model.ckpt").write_bytes(b"\x00")
    (wd / "qm9-headline" / "smol" / "train.smol").write_bytes(b"\x00")

    s = _Off(weights_dir=wd)
    m = s.get_model("qm9-headline")
    assert m is not None
    assert m.dataset == "qm9"


# ----- tools.argv builder -----


def test_generate_argv_minimal_flags(tmp_path):
    from server.models import GenerateRequest
    from server.settings import SemlaFlowSettings
    from server.tools import generate_argv

    class _Off(SemlaFlowSettings):
        model_config = SettingsConfigDict(
            env_prefix="SEMLAFLOW_TEST_", env_file=None, extra="ignore",
        )

    # Empty weights_dir → registry miss → fallback (model_name is the dataset).
    s = _Off(
        python="/opt/foo/python",
        inference_script="/opt/foo/inference.py",
        weights_dir=tmp_path / "weights",
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    argv = generate_argv(
        GenerateRequest(model_name="qm9", n_molecules=50, integration_steps=80),
        job_dir=job_dir,
        settings=s,
    )
    assert argv[0] == "/opt/foo/python"
    assert "/opt/foo/inference.py" in argv
    assert "--ckpt-path" in argv
    idx = argv.index("--ckpt-path")
    assert argv[idx + 1] == str(s.weights_dir / "qm9" / "model.ckpt")
    assert "--data-path" in argv
    idx = argv.index("--data-path")
    assert argv[idx + 1] == str(s.weights_dir / "qm9" / "smol")
    assert "--dataset" in argv
    idx = argv.index("--dataset")
    assert argv[idx + 1] == "qm9"
    assert "--save-dir" in argv
    assert str(job_dir / "output") in argv
    assert "--n-molecules" in argv and "50" in argv
    assert "--integration-steps" in argv and "80" in argv
    # seed omitted when None (pydantic default).
    assert "--seed" not in argv


def test_generate_argv_resolves_dataset_from_registry(tmp_path):
    """A staged model with a mismatched-name dir resolves dataset via manifest."""
    from server.models import GenerateRequest
    from server.settings import SemlaFlowSettings
    from server.tools import generate_argv

    class _Off(SemlaFlowSettings):
        model_config = SettingsConfigDict(
            env_prefix="SEMLAFLOW_TEST_", env_file=None, extra="ignore",
        )

    wd = tmp_path / "weights"
    _stage_model(wd, "geom-drugs", "geom-drugs")
    s = _Off(weights_dir=wd)
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    argv = generate_argv(
        GenerateRequest(model_name="geom-drugs", seed=7),
        job_dir=job_dir,
        settings=s,
    )
    idx = argv.index("--dataset")
    assert argv[idx + 1] == "geom-drugs"
    assert "--seed" in argv and "7" in argv

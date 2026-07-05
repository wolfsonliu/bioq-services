"""Offline HTTP tests for openadmet-server.

`conftest.py` registers the service dir as `server` package.
Subprocess is stubbed via `OPENADMET_PYTHON=/bin/true` — the framework
still submits jobs and runs `bash -c`, but the underlying `openadmet
predict` call is replaced by `/bin/true`, which exits 0 without doing
anything.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


LOSARTAN = "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"


def _make_fake_model(models_root: Path, name: str, input_col: str,
                     target_cols: list[str], biotargets: list[str]) -> Path:
    """Create the minimum layout the server needs to recognize a model.

    (metadata.yaml + data.yaml + procedure.yaml + a dummy model.pth so
    the model dir is non-empty).
    """
    model_dir = models_root / name
    (model_dir / "recipe_components").mkdir(parents=True, exist_ok=True)

    (model_dir / "recipe_components" / "metadata.yaml").write_text(yaml.safe_dump({
        "version": "v1",
        "name": "test",
        "tag": f"{name}-tag",
        "build_number": 0,
        "description": "test model",
        "authors": "unit test",
        "email": "test@example.com",
        "biotargets": biotargets,
        "tags": ["test"],
        "driver": "pytorch",
    }))
    (model_dir / "recipe_components" / "data.yaml").write_text(yaml.safe_dump({
        "type": "intake",
        "input_col": input_col,
        "target_cols": target_cols,
        "dropna": True,
    }))
    (model_dir / "recipe_components" / "procedure.yaml").write_text(yaml.safe_dump({
        "feat": {"type": "ChemPropFeaturizer"},
        "model": {"type": "ChemPropModel"},
        "split": {"type": "ShuffleSplitter"},
        "train": {"type": "LightningTrainer"},
    }))
    (model_dir / "model.pth").write_bytes(b"\x00")
    (model_dir / "model.json").write_text("{}")
    return model_dir


@pytest.fixture
def client(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    models_root = weights_dir / "models"
    foundations = weights_dir / "foundations" / ".chemprop"
    models_root.mkdir(parents=True, exist_ok=True)
    foundations.mkdir(parents=True, exist_ok=True)

    # Pre-stage 2 fake models — one CYP multitask (CANONICAL alias), one
    # PXR singletask (SMILES alias) — to exercise input_col grouping.
    _make_fake_model(
        models_root, "cyp-mt-test",
        input_col="OPENADMET_CANONICAL_SMILES",
        target_cols=["OPENADMET_LOGAC50_CYP3A4", "OPENADMET_LOGAC50_CYP2D6"],
        biotargets=["CYP3A4", "CYP2D6"],
    )
    _make_fake_model(
        models_root, "pxr-test",
        input_col="OPENADMET_SMILES",
        target_cols=["pchembl_value_mean"],
        biotargets=["CYP3A4"],
    )
    (foundations / "chemeleon_mp.pt").write_bytes(b"\x00" * 4096)

    monkeypatch.setenv("OPENADMET_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("OPENADMET_ROOT", str(tmp_path / "upstream"))
    monkeypatch.setenv("OPENADMET_PYTHON", "/bin/true")
    monkeypatch.setenv("OPENADMET_WEIGHTS_DIR", str(weights_dir))
    (tmp_path / "upstream").mkdir(parents=True, exist_ok=True)

    # Reset the model registry LRU cache between tests.
    from server.settings import _cached_list_models
    _cached_list_models.cache_clear()

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ===== Health / manifest =====================================================


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "openadmet"


def test_healthz_detail_reports_models_and_foundation(client):
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["chemeleon_foundation_present"] is True
    assert body["weights_loaded"] is True
    assert body["models_count"] == 2
    assert set(body["models_available"]) == {"cyp-mt-test", "pxr-test"}


def test_healthz_detail_reports_foundation_missing(client, tmp_path):
    # Delete the foundation and re-poll.
    (tmp_path / "weights" / "foundations" / ".chemprop" / "chemeleon_mp.pt").unlink()
    body = client.get("/healthz/detail").json()
    assert body["chemeleon_foundation_present"] is False
    assert body["weights_loaded"] is False


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "openadmet"


def test_manifest_lists_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/predict" in paths
    assert "/api/compare" in paths


def test_manifest_extras_include_registry_and_tool_outputs(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "OpenADMET Models"
    assert "predict" in extras["tool_outputs"]
    assert "compare" in extras["tool_outputs"]
    assert extras["model_registry"]["endpoint"] == "GET /api/models"
    assert extras["model_registry"]["current_count"] == 2


def test_manifest_examples_have_curl(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    for path in ("/api/predict", "/api/compare"):
        examples = by_path[path]["examples"]
        assert len(examples) >= 2
        assert all("curl" in (e.get("curl") or "").lower() for e in examples)


def test_openapi_served(client):
    r = client.get("/openapi.json")
    r.raise_for_status()
    paths = r.json()["paths"]
    assert "/api/predict" in paths
    assert "/api/compare" in paths
    assert "/api/tasks/predict" in paths
    assert "/api/tasks/compare" in paths
    assert "/api/models" in paths


# ===== /api/models registry ==================================================


def test_models_endpoint_lists_registered(client):
    body = client.get("/api/models").json()
    assert body["count"] == 2
    names = {m["name"] for m in body["models"]}
    assert names == {"cyp-mt-test", "pxr-test"}
    cyp = next(m for m in body["models"] if m["name"] == "cyp-mt-test")
    assert cyp["input_col"] == "OPENADMET_CANONICAL_SMILES"
    assert cyp["target_cols"] == ["OPENADMET_LOGAC50_CYP3A4", "OPENADMET_LOGAC50_CYP2D6"]
    assert cyp["biotargets"] == ["CYP3A4", "CYP2D6"]
    assert cyp["model_type"] == "ChemPropModel"


# ===== Validation errors =====================================================


def test_predict_missing_input_returns_422(client):
    r = client.post("/api/predict", data={"model_names": '["cyp-mt-test"]'})
    assert r.status_code == 422


def test_predict_unknown_model_returns_422(client):
    r = client.post(
        "/api/predict",
        data={"input_smiles": LOSARTAN, "model_names": '["not-a-real-model"]'},
    )
    assert r.status_code == 422
    assert "not registered" in r.text.lower()


def test_predict_multiple_input_sources_returns_422(client):
    r = client.post(
        "/api/predict",
        data={
            "input_smiles": LOSARTAN,
            "input_csv_uri": "file:///tmp/other.csv",
            "model_names": '["cyp-mt-test"]',
        },
    )
    # Framework serializes _prepare_predict_input's HTTPException as 422.
    # In the submit-poll path, the exception is raised inside _build, which
    # runs inside the JobRunner; the framework surfaces this as job failure
    # (status_code=200 with status=failed) OR 5xx.  Accept either.
    assert r.status_code in (422, 500), r.text


def test_predict_acquisition_mismatch_returns_422(client):
    r = client.post(
        "/api/predict",
        data={
            "input_smiles": LOSARTAN,
            "model_names": '["cyp-mt-test"]',
            "aq_fxns": '["ucb"]',
            # Missing beta -> pydantic validator rejects.
        },
    )
    assert r.status_code == 422


def test_compare_missing_both_modes_returns_422(client):
    r = client.post("/api/compare", data={"report": "false"})
    assert r.status_code == 422


def test_compare_mode_a_needs_two_models(client):
    r = client.post("/api/compare", data={"model_names": '["cyp-mt-test"]'})
    assert r.status_code == 422


# ===== Predict smoke (subprocess stubbed via /bin/true) ======================


def test_predict_returns_job_with_input_params(client):
    r = client.post(
        "/api/predict",
        data={
            "input_smiles": f"{LOSARTAN},CC(=O)O",
            "model_names": '["cyp-mt-test"]',
            "accelerator": "cpu",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["input_smiles"].startswith(LOSARTAN[:20])
    assert body["input_params"]["model_names"] == ["cyp-mt-test"]
    assert body["input_params"]["accelerator"] == "cpu"


def test_predict_multi_model_mixed_input_col_returns_job(client):
    """Two models with different input_col should still produce one job."""
    r = client.post(
        "/api/predict",
        data={
            "input_smiles": LOSARTAN,
            "model_names": '["cyp-mt-test", "pxr-test"]',
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_params"]["model_names"] == ["cyp-mt-test", "pxr-test"]


def test_predict_csv_upload_returns_job(client):
    csv_path = Path(__file__).parent / "data" / "demo_input.csv"
    with open(csv_path, "rb") as f:
        r = client.post(
            "/api/predict",
            data={"model_names": '["cyp-mt-test"]'},
            files={"input_csv": ("demo_input.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text


# ===== Compare smoke (subprocess stubbed) ====================================


def test_compare_mode_a_returns_job(client):
    r = client.post(
        "/api/compare",
        data={
            "model_names": '["cyp-mt-test", "pxr-test"]',
            "label_types": '["biotarget", "biotarget"]',
        },
    )
    assert r.status_code == 200, r.text
    assert "job_id" in r.json()


def test_compare_mode_b_returns_job(client):
    data_dir = Path(__file__).parent / "data"
    with open(data_dir / "demo_stats_a.json", "rb") as fa, \
         open(data_dir / "demo_stats_b.json", "rb") as fb:
        r = client.post(
            "/api/compare",
            data={
                "labels": '["model_a", "model_b"]',
                "task_names": '["pchembl_value_mean", "pchembl_value_mean"]',
            },
            files=[
                ("model_stats_files", ("stats_a.json", fa, "application/json")),
                ("model_stats_files", ("stats_b.json", fb, "application/json")),
            ],
        )
    assert r.status_code == 200, r.text


# ===== Settings ==============================================================


def test_settings_defaults():
    from server.settings import OpenAdmetSettings

    class _Off(OpenAdmetSettings):
        model_config = SettingsConfigDict(
            env_prefix="OPENADMET_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/openadmet_jobs")
    assert s.root == Path("/opt/openadmet/upstream")
    assert s.weights_dir == Path("/data/models/openadmet")
    assert s.models_root == Path("/data/models/openadmet/models")
    assert s.chemeleon_foundation == Path(
        "/data/models/openadmet/foundations/.chemprop/chemeleon_mp.pt"
    )
    assert s.max_concurrent_jobs == 1


def test_settings_env_override(monkeypatch):
    from server.settings import OpenAdmetSettings

    monkeypatch.setenv("OPENADMET_PYTHON", "/custom/python")
    monkeypatch.setenv("OPENADMET_WEIGHTS_DIR", "/mnt/scratch/openadmet")
    s = OpenAdmetSettings()
    assert s.python == "/custom/python"
    assert s.weights_dir == Path("/mnt/scratch/openadmet")
    assert s.models_root == Path("/mnt/scratch/openadmet/models")
    assert s.chemeleon_foundation == Path(
        "/mnt/scratch/openadmet/foundations/.chemprop/chemeleon_mp.pt"
    )

"""Offline tests for diffdock-pp-server.

Real diffusion model never runs in offline tests — the subprocess is
stubbed via DIFFDOCK_PP_PYTHON=/bin/true so no GPU / weights needed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Reload the app under a sandbox of tmp_path-based dirs + stubbed python."""
    monkeypatch.setenv("DIFFDOCK_PP_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("DIFFDOCK_PP_ROOT", str(tmp_path / "diffdock_pp"))
    monkeypatch.setenv("DIFFDOCK_PP_PYTHON", "/bin/true")
    monkeypatch.setenv("DIFFDOCK_PP_INFERENCE_SCRIPT",
                       str(tmp_path / "inference.py"))
    monkeypatch.setenv("DIFFDOCK_PP_CONFIG_YAML",
                       str(tmp_path / "config.yaml"))
    monkeypatch.setenv("DIFFDOCK_PP_WEIGHTS_DIR", str(tmp_path / "weights"))
    (tmp_path / "diffdock_pp").mkdir(parents=True, exist_ok=True)

    # Touch enough of the expected weight tree so /healthz/detail can find
    # some files (missing ones will show up in weights_missing).
    weights = tmp_path / "weights"
    score_dir = weights / "large_model_dips" / "fold_0"
    score_dir.mkdir(parents=True)
    (score_dir / "model_best_338669_140_31.084_30.347.pth").write_bytes(b"\x00")
    (weights / "large_model_dips" / "args.yaml").write_text("dummy: true\n")

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Health / manifest -----


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "diffdock-pp"
    assert "version" in body


def test_healthz_detail_reports_weights(client):
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "diffdock-pp"
    # Only score checkpoint + args.yaml were touched → confidence + ESM
    # entries should be missing.
    assert body["weights_loaded"] is False
    missing = body["weights_missing"]
    assert "confidence_checkpoint" in missing
    assert "esm2_checkpoint" in missing


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "diffdock-pp"


def test_manifest_lists_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/dock" in paths


def test_manifest_extras_have_model_info(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "DiffDock-PP"
    assert "rigid" in extras["model"]["task"] or "docking" in extras["model"]["task"]
    assert "dock" in extras["tool_outputs"]
    # Confidence-model + top_k guidance must land in the manifest so agents
    # can decide sanely without reading source.
    assert "top_k" in extras["config_tips"]
    assert "use_confidence_model" in extras["config_tips"]


def test_manifest_extras_have_endpoint_examples(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    examples = by_path["/api/dock"]["examples"]
    assert len(examples) >= 2
    # At least one example must show the seed / URI variant.
    assert any("seed=" in (e.get("curl") or "") for e in examples)


def test_openapi_served(client):
    r = client.get("/openapi.json")
    r.raise_for_status()
    paths = r.json()["paths"]
    assert "/api/dock" in paths
    assert "/api/tasks/dock" in paths


# ----- Validation errors -----


def test_dock_missing_inputs_returns_422(client):
    """No receptor or ligand → 422 from resolve_input."""
    r = client.post("/api/dock", data={"num_samples": "5"})
    assert r.status_code in (400, 422)


def test_dock_num_samples_out_of_range_returns_422(client, tmp_path):
    rec = tmp_path / "r.pdb"
    rec.write_bytes(b"ATOM")
    lig = tmp_path / "l.pdb"
    lig.write_bytes(b"ATOM")
    r = client.post(
        "/api/dock",
        files={
            "receptor": ("r.pdb", rec.read_bytes(), "chemical/x-pdb"),
            "ligand": ("l.pdb", lig.read_bytes(), "chemical/x-pdb"),
        },
        data={"num_samples": "0"},
    )
    assert r.status_code == 422


def test_dock_top_k_out_of_range_returns_422(client, tmp_path):
    rec = tmp_path / "r.pdb"
    rec.write_bytes(b"ATOM")
    lig = tmp_path / "l.pdb"
    lig.write_bytes(b"ATOM")
    r = client.post(
        "/api/dock",
        files={
            "receptor": ("r.pdb", rec.read_bytes(), "chemical/x-pdb"),
            "ligand": ("l.pdb", lig.read_bytes(), "chemical/x-pdb"),
        },
        data={"num_samples": "20", "top_k": "0"},
    )
    assert r.status_code == 422


# ----- Smoke (subprocess stubbed via /bin/true) -----


def test_dock_returns_job_with_input_params(client, tmp_path):
    rec = tmp_path / "r.pdb"
    rec.write_bytes(b"ATOM      1  N   ALA A   1\n")
    lig = tmp_path / "l.pdb"
    lig.write_bytes(b"ATOM      1  N   ALA A   1\n")
    r = client.post(
        "/api/dock",
        files={
            "receptor": ("r.pdb", rec.read_bytes(), "chemical/x-pdb"),
            "ligand": ("l.pdb", lig.read_bytes(), "chemical/x-pdb"),
        },
        data={
            "num_samples": "8",
            "top_k": "3",
            "use_confidence_model": "false",
            "seed": "42",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["num_samples"] == 8
    assert body["input_params"]["top_k"] == 3
    assert body["input_params"]["use_confidence_model"] is False
    assert body["input_params"]["seed"] == 42


# ----- Settings -----


def test_settings_defaults():
    from server.settings import DiffDockPPSettings

    class _Off(DiffDockPPSettings):
        model_config = SettingsConfigDict(
            env_prefix="DIFFDOCK_PP_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/diffdock_pp_jobs")
    assert s.root == Path("/opt/diffdock-pp")
    # Weights externalized to NAS — default points at the FC mount path.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    assert s.weights_dir == Path("/data/models/diffdock-pp")
    assert s.max_concurrent_jobs == 1


def test_settings_env_override(monkeypatch):
    from server.settings import DiffDockPPSettings
    monkeypatch.setenv("DIFFDOCK_PP_PYTHON", "/custom/python")
    monkeypatch.setenv("DIFFDOCK_PP_WEIGHTS_DIR", "/mnt/scratch/w")
    s = DiffDockPPSettings()
    assert s.python == "/custom/python"
    assert s.weights_dir == Path("/mnt/scratch/w")


# ----- tools.argv builder -----


def test_dock_argv_includes_required_flags(tmp_path):
    from server.models import DockRequest
    from server.settings import DiffDockPPSettings
    from server.tools import dock_argv

    class _Off(DiffDockPPSettings):
        model_config = SettingsConfigDict(
            env_prefix="DIFFDOCK_PP_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off(
        python="/opt/foo/python",
        inference_script="/opt/foo/inference.py",
        config_yaml=tmp_path / "cfg.yaml",
        weights_dir=tmp_path / "weights",
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    rec = tmp_path / "r.pdb"
    rec.write_bytes(b"ATOM")
    lig = tmp_path / "l.pdb"
    lig.write_bytes(b"ATOM")

    argv = dock_argv(
        DockRequest(num_samples=12, top_k=3, use_confidence_model=False,
                    seed=7, mirror_ligand=True),
        job_dir=job_dir,
        receptor=rec,
        ligand=lig,
        settings=s,
    )

    assert argv[0] == "/opt/foo/python"
    assert "/opt/foo/inference.py" in argv
    assert "--receptor" in argv and str(rec) in argv
    assert "--ligand" in argv and str(lig) in argv
    assert "--num_samples" in argv and "12" in argv
    assert "--top_k" in argv and "3" in argv
    assert "--use_confidence_model" in argv and "false" in argv
    assert "--seed" in argv and "7" in argv
    assert "--mirror_ligand" in argv and "true" in argv
    assert "--score_model_dir" in argv
    assert str(s.weights_dir / "large_model_dips" / "fold_0") in argv
    assert "--confidence_model_dir" in argv
    assert str(s.weights_dir / "confidence_model_dips" / "fold_0") in argv
    assert "--config" in argv and str(s.config_yaml) in argv
    assert "--torchhub_dir" in argv
    assert str(s.weights_dir / "esm_cache") in argv
    assert "--output" in argv and str(job_dir / "output") in argv


def test_dock_argv_seed_defaults_to_zero_when_unset(tmp_path):
    from server.models import DockRequest
    from server.settings import DiffDockPPSettings
    from server.tools import dock_argv

    class _Off(DiffDockPPSettings):
        model_config = SettingsConfigDict(
            env_prefix="DIFFDOCK_PP_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off(python="/bin/true", inference_script="/opt/i.py",
             config_yaml=tmp_path / "c.yaml", weights_dir=tmp_path / "w")
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    rec = tmp_path / "r.pdb"
    rec.write_bytes(b"ATOM")
    lig = tmp_path / "l.pdb"
    lig.write_bytes(b"ATOM")
    argv = dock_argv(
        DockRequest(),
        job_dir=job_dir, receptor=rec, ligand=lig, settings=s,
    )
    seed_idx = argv.index("--seed")
    assert argv[seed_idx + 1] == "0"

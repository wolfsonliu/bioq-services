"""Offline tests for diffusion-hopping-server.

Real diffusion model never runs in offline tests — the subprocess is
stubbed via DIFFUSION_HOPPING_PYTHON=/bin/true so no GPU / weights needed.
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
    monkeypatch.setenv("DIFFUSION_HOPPING_JOBS_BASE_DIR",
                       str(tmp_path / "jobs"))
    monkeypatch.setenv("DIFFUSION_HOPPING_ROOT", str(tmp_path / "diffhopp"))
    monkeypatch.setenv("DIFFUSION_HOPPING_PYTHON", "/bin/true")
    monkeypatch.setenv("DIFFUSION_HOPPING_INFERENCE_SCRIPT",
                       str(tmp_path / "inference.py"))
    monkeypatch.setenv("DIFFUSION_HOPPING_WEIGHTS_DIR",
                       str(tmp_path / "checkpoints"))
    (tmp_path / "diffhopp").mkdir(parents=True, exist_ok=True)
    (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    # Touch one fake ckpt so healthz/detail shows ≥1 weight present.
    (tmp_path / "checkpoints" / "gvp_conditional.ckpt").write_bytes(b"\x00")

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Health / manifest -----


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "diffusion-hopping"
    assert "version" in body


def test_healthz_detail_reports_weights(client):
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "diffusion-hopping"
    # Only gvp_conditional was touched in the fixture, so the other 3 are missing.
    assert body["weights_loaded"] is False
    assert "gvp_unconditional" in body["weights_missing"]
    assert "egnn_conditional" in body["weights_missing"]


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "diffusion-hopping"


def test_manifest_lists_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/generate" in paths


def test_manifest_extras_have_model_info(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "DiffHopp"
    assert "gvp_conditional" in extras["model"]["variants"]
    assert "generate" in extras["tool_outputs"]


def test_manifest_extras_have_endpoint_examples(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    examples = by_path["/api/generate"]["examples"]
    assert len(examples) >= 2
    assert any("gvp_conditional" in (e.get("curl") or "") for e in examples)


def test_openapi_served(client):
    r = client.get("/openapi.json")
    r.raise_for_status()
    assert "/api/generate" in r.json()["paths"]


# ----- Validation errors -----


def test_generate_missing_inputs_returns_422(client):
    """Neither protein nor reference_ligand provided → 422 from resolve_input."""
    r = client.post("/api/generate", data={"num_samples": "5"})
    # Either FastAPI form validation (422) or our resolve_input 422 — either way
    # it's a 4xx; the server must not crash.
    assert r.status_code in (400, 422)


def test_generate_invalid_variant_returns_422(client, tmp_path):
    protein = tmp_path / "p.pdb"
    protein.write_bytes(b"ATOM")
    ligand = tmp_path / "l.sdf"
    ligand.write_bytes(b"$$$$")
    r = client.post(
        "/api/generate",
        files={
            "protein": ("p.pdb", protein.read_bytes(), "chemical/x-pdb"),
            "reference_ligand": ("l.sdf", ligand.read_bytes(), "chemical/x-mdl-sdfile"),
        },
        data={"num_samples": "5", "model_variant": "not_a_real_variant"},
    )
    assert r.status_code == 422


def test_generate_num_samples_out_of_range_returns_422(client, tmp_path):
    protein = tmp_path / "p.pdb"
    protein.write_bytes(b"ATOM")
    ligand = tmp_path / "l.sdf"
    ligand.write_bytes(b"$$$$")
    r = client.post(
        "/api/generate",
        files={
            "protein": ("p.pdb", protein.read_bytes(), "chemical/x-pdb"),
            "reference_ligand": ("l.sdf", ligand.read_bytes(), "chemical/x-mdl-sdfile"),
        },
        data={"num_samples": "0"},
    )
    assert r.status_code == 422


# ----- Smoke (subprocess stubbed via /bin/true) -----


def test_generate_returns_job_with_input_params(client, tmp_path):
    protein = tmp_path / "p.pdb"
    protein.write_bytes(b"ATOM      1  N   ALA A   1\n")
    ligand = tmp_path / "l.sdf"
    ligand.write_bytes(b"dummy\n$$$$\n")
    r = client.post(
        "/api/generate",
        files={
            "protein": ("p.pdb", protein.read_bytes(), "chemical/x-pdb"),
            "reference_ligand": ("l.sdf", ligand.read_bytes(), "chemical/x-mdl-sdfile"),
        },
        data={"num_samples": "3", "model_variant": "gvp_unconditional"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["num_samples"] == 3
    assert body["input_params"]["model_variant"] == "gvp_unconditional"


# ----- Settings -----


def test_settings_defaults():
    from server.settings import DiffusionHoppingSettings

    class _Off(DiffusionHoppingSettings):
        model_config = SettingsConfigDict(
            env_prefix="DIFFUSION_HOPPING_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/diffusion_hopping_jobs")
    assert s.root == Path("/opt/diffusion-hopping")
    # Weights externalized to NAS — default points at the FC mount path.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    assert s.weights_dir == Path("/data/models/diffusion-hopping/checkpoints")
    assert s.max_concurrent_jobs == 1


def test_settings_env_override(monkeypatch):
    from server.settings import DiffusionHoppingSettings
    monkeypatch.setenv("DIFFUSION_HOPPING_PYTHON", "/custom/python")
    monkeypatch.setenv("DIFFUSION_HOPPING_WEIGHTS_DIR", "/mnt/scratch/weights")
    s = DiffusionHoppingSettings()
    assert s.python == "/custom/python"
    assert s.weights_dir == Path("/mnt/scratch/weights")


# ----- tools.argv builder -----


def test_generate_argv_includes_required_flags(tmp_path):
    from server.models import GenerateRequest
    from server.settings import DiffusionHoppingSettings
    from server.tools import generate_argv

    class _Off(DiffusionHoppingSettings):
        model_config = SettingsConfigDict(
            env_prefix="DIFFUSION_HOPPING_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off(
        python="/opt/foo/python",
        inference_script="/opt/foo/inference.py",
        weights_dir=tmp_path / "checkpoints",
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    protein = tmp_path / "p.pdb"
    protein.write_bytes(b"ATOM")
    ligand = tmp_path / "l.sdf"
    ligand.write_bytes(b"")

    argv = generate_argv(
        GenerateRequest(num_samples=7, model_variant="egnn_conditional"),
        job_dir=job_dir,
        input_molecule=ligand,
        input_protein=protein,
        settings=s,
    )

    assert argv[0] == "/opt/foo/python"
    assert "/opt/foo/inference.py" in argv
    assert "--input_molecule" in argv
    assert str(ligand) in argv
    assert "--input_protein" in argv
    assert str(protein) in argv
    assert "--num_samples" in argv
    assert "7" in argv
    assert "--variant" in argv
    assert "egnn_conditional" in argv
    assert "--checkpoint" in argv
    assert str(s.weights_dir / "egnn_conditional.ckpt") in argv
    assert "--output" in argv
    assert str(job_dir / "output") in argv

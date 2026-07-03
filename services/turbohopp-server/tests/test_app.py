"""Offline tests for turbohopp-server.

Real consistency-model sampling never runs in offline tests — the subprocess
is stubbed via TURBOHOPP_PYTHON=/bin/true so no GPU / weights needed.
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
    monkeypatch.setenv("TURBOHOPP_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("TURBOHOPP_ROOT", str(tmp_path / "turbohopp"))
    monkeypatch.setenv("TURBOHOPP_PYTHON", "/bin/true")
    monkeypatch.setenv("TURBOHOPP_INFERENCE_SCRIPT", str(tmp_path / "inference.py"))
    monkeypatch.setenv("TURBOHOPP_WEIGHTS_DIR", str(tmp_path / "checkpoints"))
    (tmp_path / "turbohopp").mkdir(parents=True, exist_ok=True)
    (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


@pytest.fixture
def client_with_weights(tmp_path, monkeypatch):
    """Same as `client` but with one fake .ckpt so weights_loaded=True."""
    monkeypatch.setenv("TURBOHOPP_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("TURBOHOPP_ROOT", str(tmp_path / "turbohopp"))
    monkeypatch.setenv("TURBOHOPP_PYTHON", "/bin/true")
    monkeypatch.setenv("TURBOHOPP_INFERENCE_SCRIPT", str(tmp_path / "inference.py"))
    monkeypatch.setenv("TURBOHOPP_WEIGHTS_DIR", str(tmp_path / "checkpoints"))
    (tmp_path / "turbohopp").mkdir(parents=True, exist_ok=True)
    (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (tmp_path / "checkpoints" / "turbohopp_consistency.ckpt").write_bytes(b"\x00")

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Health / manifest -----


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "turbohopp"
    assert "version" in body


def test_healthz_detail_reports_missing_weights(client):
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "turbohopp"
    assert body["weights_loaded"] is False
    assert body["files_found"] == 0
    assert body["weights_dir"].endswith("checkpoints")


def test_healthz_detail_reports_present_weights(client_with_weights):
    body = client_with_weights.get("/healthz/detail").json()
    assert body["weights_loaded"] is True
    assert body["files_found"] == 1


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "turbohopp"


def test_manifest_lists_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/generate" in paths


def test_manifest_extras_have_model_info(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "TurboHopp"
    assert "consistency" in extras["model"]["task"].lower()
    assert "generate" in extras["tool_outputs"]


def test_manifest_extras_have_endpoint_examples(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    examples = by_path["/api/generate"]["examples"]
    assert len(examples) >= 2
    assert any("num_sampling_steps" in (e.get("curl") or "") for e in examples)


def test_manifest_config_tips_present(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    tips = extras["config_tips"]
    assert "num_sampling_steps" in tips
    assert "find_best" in tips


def test_openapi_served(client):
    r = client.get("/openapi.json")
    r.raise_for_status()
    assert "/api/generate" in r.json()["paths"]
    # Task endpoint auto-registered under settings.task_endpoints_enabled default.
    assert "/api/tasks/generate" in r.json()["paths"]


# ----- Validation errors -----


def test_generate_missing_inputs_returns_422(client):
    """Neither protein nor reference_ligand → 422 from resolve_input."""
    r = client.post("/api/generate", data={"num_samples": "5"})
    assert r.status_code in (400, 422)


def test_generate_missing_ligand_returns_422(client, tmp_path):
    protein = tmp_path / "p.pdb"
    protein.write_bytes(b"ATOM")
    r = client.post(
        "/api/generate",
        files={"protein": ("p.pdb", protein.read_bytes(), "chemical/x-pdb")},
        data={"num_samples": "5"},
    )
    assert r.status_code in (400, 422)


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


def test_generate_num_sampling_steps_out_of_range_returns_422(client, tmp_path):
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
        data={"num_samples": "5", "num_sampling_steps": "0"},
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
        data={
            "num_samples": "3",
            "num_sampling_steps": "5",
            "find_best": "true",
            "seed": "42",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["num_samples"] == 3
    assert body["input_params"]["num_sampling_steps"] == 5
    assert body["input_params"]["find_best"] is True
    assert body["input_params"]["seed"] == 42


# ----- Settings -----


def test_settings_defaults():
    from server.settings import TurboHoppSettings

    class _Off(TurboHoppSettings):
        model_config = SettingsConfigDict(
            env_prefix="TURBOHOPP_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/turbohopp_jobs")
    assert s.root == Path("/opt/turbohopp")
    # Weights externalized to NAS under a versioned subdir.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    assert s.weights_dir == Path("/data/models/turbohopp/checkpoints/v1")
    assert s.max_concurrent_jobs == 1


def test_settings_env_override(monkeypatch):
    from server.settings import TurboHoppSettings

    monkeypatch.setenv("TURBOHOPP_PYTHON", "/custom/python")
    monkeypatch.setenv("TURBOHOPP_WEIGHTS_DIR", "/mnt/scratch/weights")
    s = TurboHoppSettings()
    assert s.python == "/custom/python"
    assert s.weights_dir == Path("/mnt/scratch/weights")


# ----- tools.argv builder -----


def _off_settings(tmp_path: Path):
    from server.settings import TurboHoppSettings

    class _Off(TurboHoppSettings):
        model_config = SettingsConfigDict(
            env_prefix="TURBOHOPP_TEST_",
            env_file=None,
            extra="ignore",
        )

    return _Off(
        python="/opt/foo/python",
        inference_script="/opt/foo/inference.py",
        weights_dir=tmp_path / "checkpoints",
    )


def test_generate_argv_includes_required_flags(tmp_path):
    from server.models import GenerateRequest
    from server.tools import generate_argv

    s = _off_settings(tmp_path)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "turbohopp_consistency.ckpt").write_bytes(b"\x00")

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    protein = tmp_path / "p.pdb"
    protein.write_bytes(b"ATOM")
    ligand = tmp_path / "l.sdf"
    ligand.write_bytes(b"")

    argv = generate_argv(
        GenerateRequest(num_samples=7, num_sampling_steps=12, find_best=True, seed=17),
        job_dir=job_dir,
        input_protein=protein,
        input_molecule=ligand,
        settings=s,
    )

    assert argv[0] == "/opt/foo/python"
    assert "/opt/foo/inference.py" in argv
    assert "--input_protein" in argv
    assert str(protein) in argv
    assert "--input_molecule" in argv
    assert str(ligand) in argv
    assert "--num_samples" in argv and "7" in argv
    assert "--num_sampling_steps" in argv and "12" in argv
    assert "--checkpoint" in argv
    # Auto-picked from weights_dir (single .ckpt present).
    assert str(ckpt_dir / "turbohopp_consistency.ckpt") in argv
    assert "--output" in argv and str(job_dir / "output") in argv
    assert "--find_best" in argv
    assert "--seed" in argv and "17" in argv


def test_generate_argv_omits_find_best_when_false(tmp_path):
    from server.models import GenerateRequest
    from server.tools import generate_argv

    s = _off_settings(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "x.ckpt").write_bytes(b"\x00")
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    argv = generate_argv(
        GenerateRequest(num_samples=3, num_sampling_steps=5, find_best=False),
        job_dir=job_dir,
        input_protein=tmp_path / "p.pdb",
        input_molecule=tmp_path / "l.sdf",
        settings=s,
    )
    assert "--find_best" not in argv
    assert "--seed" not in argv  # default seed=None → omitted


def test_generate_argv_uses_checkpoint_name_override(tmp_path, monkeypatch):
    from server.models import GenerateRequest
    from server.tools import generate_argv

    s = _off_settings(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "auto.ckpt").write_bytes(b"\x00")
    (tmp_path / "checkpoints" / "pinned.ckpt").write_bytes(b"\x00")
    monkeypatch.setenv("TURBOHOPP_CHECKPOINT_NAME", "pinned.ckpt")

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    argv = generate_argv(
        GenerateRequest(),
        job_dir=job_dir,
        input_protein=tmp_path / "p.pdb",
        input_molecule=tmp_path / "l.sdf",
        settings=s,
    )
    assert str(tmp_path / "checkpoints" / "pinned.ckpt") in argv

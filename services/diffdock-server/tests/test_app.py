"""Offline tests for diffdock-server (no real subprocess / GPU needed)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFDOCK_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("DIFFDOCK_ROOT", str(tmp_path / "diffdock"))
    monkeypatch.setenv("DIFFDOCK_WEIGHTS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DIFFDOCK_PYTHON", "/bin/true")  # stub subprocess
    (tmp_path / "diffdock").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


def _off_settings():
    from server.settings import DiffdockSettings

    class _Off(DiffdockSettings):
        model_config = SettingsConfigDict(
            env_prefix="DIFFDOCK_OFFLINE_", env_file=None, extra="ignore",
        )

    return _Off()


# ----- Healthcheck / manifest smoke -----

def test_health(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "diffdock"
    assert "version" in body


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "diffdock"


def test_healthz_detail_reports_missing_weights(client):
    """Weights + LUT probe should not raise when NAS mount is absent."""
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "diffdock"
    assert body["weights_loaded"] is False
    assert "score_model_ckpt" in body["weights_missing"]
    assert "confidence_model_ckpt" in body["weights_missing"]
    assert "esm2_650M" in body["weights_missing"]
    # ESMFold is soft — reported but not in weights_missing
    assert "esmfold_available" in body
    assert body["esmfold_available"] is False
    assert body["so3_cache_ok"] is False
    assert body["torus_cache_ok"] is False


def test_healthz_detail_flips_when_weights_appear(client, tmp_path):
    """Once the NAS layout exists, weights_loaded flips to True."""
    weights = tmp_path / "models"
    score_dir = weights / "score_model"
    conf_dir = weights / "confidence_model"
    esm_dir = weights / "esm_cache/hub/checkpoints"
    for d in (score_dir, conf_dir, esm_dir):
        d.mkdir(parents=True)
    (score_dir / "best_ema_inference_epoch_model.pt").write_bytes(b"x")
    (score_dir / "model_parameters.yml").write_text("x: 1\n")
    (conf_dir / "best_model_epoch75.pt").write_bytes(b"x")
    (conf_dir / "model_parameters.yml").write_text("x: 1\n")
    (esm_dir / "esm2_t33_650M_UR50D.pt").write_bytes(b"x")

    body = client.get("/healthz/detail").json()
    assert body["weights_loaded"] is True
    assert body["weights_missing"] == {}


# ----- Settings -----

def test_settings_defaults():
    s = _off_settings()
    assert s.jobs_base_dir == Path("/data/diffdock_jobs")
    assert s.root == Path("/opt/diffdock")
    assert s.python == "/opt/conda/envs/diffdock/bin/python"
    assert s.inference_script == Path("/opt/diffdock/server/run_inference.py")
    assert s.config_yaml == Path("/opt/diffdock/server/default_inference_args.yaml")
    assert s.weights_dir == Path("/data/models/diffdock")
    assert s.score_model_dir == Path("/data/models/diffdock/score_model")
    assert s.confidence_model_dir == Path(
        "/data/models/diffdock/confidence_model"
    )
    assert s.esm_cache_dir == Path("/data/models/diffdock/esm_cache")


# ----- Request model validators -----


def test_dock_request_defaults_ok():
    from server.models import DockRequest

    # No sides set: model itself does not enforce (endpoint layer does).
    r = DockRequest()
    assert r.samples_per_complex == 10
    assert r.inference_steps == 20
    assert r.actual_steps == 19
    assert r.batch_size == 10
    assert r.no_final_step_noise is True
    assert r.save_visualisation is False


def test_dock_request_rejects_actual_steps_gt_inference():
    from pydantic import ValidationError

    from server.models import DockRequest

    with pytest.raises(ValidationError, match="actual_steps"):
        DockRequest(inference_steps=15, actual_steps=20)


def test_dock_request_rejects_bad_complex_name():
    from pydantic import ValidationError

    from server.models import DockRequest

    with pytest.raises(ValidationError, match="complex_name"):
        DockRequest(complex_name="bad name/../etc")


def test_dock_request_accepts_valid_complex_name():
    from server.models import DockRequest

    r = DockRequest(complex_name="1a0q_test-42")
    assert r.complex_name == "1a0q_test-42"


def test_dock_request_rejects_two_protein_text_inputs():
    """URI + sequence both set → ValidationError."""
    from pydantic import ValidationError

    from server.models import DockRequest

    with pytest.raises(ValidationError, match="protein_uri or protein_sequence"):
        DockRequest(
            protein_uri="file:///data/target.pdb",
            protein_sequence="MKW" * 30,
        )


def test_dock_request_rejects_two_ligand_text_inputs():
    """URI + description both set → ValidationError."""
    from pydantic import ValidationError

    from server.models import DockRequest

    with pytest.raises(ValidationError, match="ligand_uri or ligand_description"):
        DockRequest(
            ligand_uri="file:///data/lig.sdf",
            ligand_description="CCO",
        )


def test_dock_request_rejects_out_of_range_samples():
    from pydantic import ValidationError

    from server.models import DockRequest

    with pytest.raises(ValidationError):
        DockRequest(samples_per_complex=0)
    with pytest.raises(ValidationError):
        DockRequest(samples_per_complex=200)


# ----- tools.dock_argv builder -----


def test_dock_argv_pdb_and_sdf_paths(tmp_path):
    from server.models import DockRequest
    from server.tools import dock_argv

    settings = _off_settings()
    req = DockRequest(complex_name="test_c")
    argv = dock_argv(
        protein_path=tmp_path / "target.pdb",
        protein_sequence=None,
        ligand_arg=str(tmp_path / "lig.sdf"),
        out_dir=tmp_path / "out",
        params=req,
        settings=settings,
    )
    assert argv[0] == settings.python
    assert argv[1] == str(settings.inference_script)
    assert "--protein_path" in argv
    assert str(tmp_path / "target.pdb") in argv
    assert "--ligand" in argv
    assert str(tmp_path / "lig.sdf") in argv
    assert "--complex_name" in argv
    assert "test_c" in argv
    # Model/config/torchhub paths come from settings, not request
    assert "--model_dir" in argv
    assert "--confidence_model_dir" in argv
    assert "--config" in argv
    assert "--torchhub_dir" in argv


def test_dock_argv_sequence_and_smiles(tmp_path):
    from server.models import DockRequest
    from server.tools import dock_argv

    settings = _off_settings()
    req = DockRequest(protein_sequence="MKW" * 30, ligand_description="CCO")
    argv = dock_argv(
        protein_path=None,
        protein_sequence=req.protein_sequence,
        ligand_arg=req.ligand_description,
        out_dir=tmp_path / "out",
        params=req,
        settings=settings,
    )
    assert "--protein_sequence" in argv
    assert "MKW" * 30 in argv
    assert "--protein_path" not in argv
    assert "--ligand" in argv
    assert "CCO" in argv


def test_dock_argv_rejects_both_protein_forms(tmp_path):
    from server.models import DockRequest
    from server.tools import dock_argv

    with pytest.raises(ValueError, match="Exactly one of protein_path"):
        dock_argv(
            protein_path=tmp_path / "p.pdb",
            protein_sequence="MKW",
            ligand_arg="CCO",
            out_dir=tmp_path,
            params=DockRequest(),
            settings=_off_settings(),
        )


def test_dock_argv_rejects_neither_protein_form(tmp_path):
    from server.models import DockRequest
    from server.tools import dock_argv

    with pytest.raises(ValueError, match="Exactly one of protein_path"):
        dock_argv(
            protein_path=None,
            protein_sequence=None,
            ligand_arg="CCO",
            out_dir=tmp_path,
            params=DockRequest(),
            settings=_off_settings(),
        )


def test_dock_argv_visualisation_flag(tmp_path):
    from server.models import DockRequest
    from server.tools import dock_argv

    req = DockRequest(save_visualisation=True, ligand_description="CCO",
                      protein_sequence="MKW" * 30)
    argv = dock_argv(
        protein_path=None, protein_sequence=req.protein_sequence,
        ligand_arg=req.ligand_description, out_dir=tmp_path,
        params=req, settings=_off_settings(),
    )
    # save_visualisation is passed as 'true' string; upstream flag handling
    # happens in run_inference.py (store_true only when true).
    idx = argv.index("--save_visualisation")
    assert argv[idx + 1] == "true"


# ----- Adapter -----


def test_adapter_name():
    from server.adapter import DiffdockAdapter

    a = DiffdockAdapter(settings=_off_settings())
    assert a.name == "diffdock"


def test_adapter_detect_outputs_empty(tmp_path):
    from server.adapter import DiffdockAdapter

    a = DiffdockAdapter(settings=_off_settings())
    job_dir = tmp_path / "job"
    (job_dir / "output").mkdir(parents=True)
    assert a.detect_outputs(job_dir) is False


def test_adapter_detect_outputs_finds_rank1(tmp_path):
    """rank1.sdf under output/<complex_name>/ is the target file."""
    from server.adapter import DiffdockAdapter

    a = DiffdockAdapter(settings=_off_settings())
    job_dir = tmp_path / "job"
    nested = job_dir / "output" / "my_complex"
    nested.mkdir(parents=True)
    (nested / "rank1.sdf").write_text("valid_sdf\n$$$$\n")
    # write a job.json sidecar so _infer_complex_name works
    (job_dir / "job.json").write_text(
        '{"input_params": {"complex_name": "my_complex"}}'
    )
    assert a.detect_outputs(job_dir) is True


def test_adapter_detect_outputs_scans_when_no_sidecar(tmp_path):
    """Even without job.json, rglob finds rank1.sdf."""
    from server.adapter import DiffdockAdapter

    a = DiffdockAdapter(settings=_off_settings())
    job_dir = tmp_path / "job"
    nested = job_dir / "output" / "any_name"
    nested.mkdir(parents=True)
    (nested / "rank1.sdf").write_text("x")
    assert a.detect_outputs(job_dir) is True


def test_adapter_subprocess_env():
    from server.adapter import DiffdockAdapter

    a = DiffdockAdapter(settings=_off_settings())
    env = a.subprocess_env()
    assert env["PYTHONPATH"] == str(a.settings.root)
    assert env["TORCH_HOME"] == str(a.settings.esm_cache_dir)
    assert env["REPOSITORY_URL"] == "file:///dev/null"


def test_adapter_manifest_extras_shape():
    from server.adapter import DiffdockAdapter

    a = DiffdockAdapter(settings=_off_settings())
    extras = a.manifest_extras()
    assert "DiffDock-L" in extras["model"]["name"]
    assert extras["model"]["license"] == "MIT"
    assert "dock" in extras["tool_outputs"]
    assert "config_tips" in extras
    assert "samples_per_complex" in extras["config_tips"]


def test_adapter_endpoint_examples_shape():
    from server.adapter import DiffdockAdapter

    a = DiffdockAdapter(settings=_off_settings())
    ex = a.endpoint_examples()
    assert set(ex.keys()) == {"/api/dock", "/api/tasks/dock"}
    for path, examples in ex.items():
        assert examples, f"no examples for {path}"
        assert examples[0].curl.startswith("curl")


# ----- Endpoint smoke via TestClient -----


def test_manifest_lists_both_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/dock" in paths
    assert "/api/tasks/dock" in paths


def test_dock_rejects_no_protein_input_422(client):
    """No protein file / URI / sequence → 422."""
    resp = client.post(
        "/api/dock",
        files={"ligand": ("lig.sdf", b"$$$$", "chemical/x-mdl-sdfile")},
        data={"complex_name": "test"},
    )
    assert resp.status_code == 422
    assert "protein" in resp.json()["detail"].lower()


def test_dock_rejects_no_ligand_input_422(client, tmp_path):
    """No ligand file / URI / description → 422."""
    resp = client.post(
        "/api/dock",
        files={"protein": ("p.pdb", b"HEADER TEST\nEND\n", "chemical/x-pdb")},
        data={"complex_name": "test"},
    )
    assert resp.status_code == 422
    assert "ligand" in resp.json()["detail"].lower()


def test_dock_rejects_two_protein_inputs_422(client):
    """Uploading protein file AND passing protein_sequence → 422."""
    resp = client.post(
        "/api/dock",
        files={
            "protein": ("p.pdb", b"HEADER\nEND\n", "chemical/x-pdb"),
            "ligand": ("l.sdf", b"$$$$", "chemical/x-mdl-sdfile"),
        },
        data={
            "protein_sequence": "MKW" * 30,
            "complex_name": "test",
        },
    )
    assert resp.status_code == 422
    assert "protein" in resp.json()["detail"].lower()


def test_dock_smiles_and_sequence_input_accepted(client, monkeypatch):
    """SMILES + protein_sequence submits fine; wrapper subprocess stub returns 0.

    We only check the 202 submission path — the stub /bin/true won't
    produce outputs, so the eventual job will FAIL, but the submission
    itself must succeed.
    """
    resp = client.post(
        "/api/dock",
        data={
            "protein_sequence": "MKW" * 30,
            "ligand_description": "COc(cc1)ccc1C#N",
            "complex_name": "smoke",
            "samples_per_complex": "2",
            "inference_steps": "10",
            "actual_steps": "10",
        },
    )
    # Framework returns 200 for sync submit endpoint (JobInfo body)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("pending", "running", "failed", "completed")
    assert "job_id" in body


def test_dock_task_smiles_sequence_accepted(client):
    """Async task endpoint accepts the same shape."""
    resp = client.post(
        "/api/tasks/dock",
        data={
            "protein_sequence": "MKW" * 30,
            "ligand_description": "CCO",
            "complex_name": "smoke_async",
        },
        headers={"X-Fc-Invocation-Type": "Async"},
    )
    # Async task returns 200 as well (framework wraps the response)
    assert resp.status_code in (200, 202)

"""Offline tests for odesign-server (no real ODesign pipeline needed)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ODESIGN_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("ODESIGN_ROOT", str(tmp_path / "odesign"))
    monkeypatch.setenv("ODESIGN_PYTHON", "/bin/true")
    monkeypatch.setenv("ODESIGN_INFERENCE_SCRIPT", "/bin/true")
    monkeypatch.setenv("ODESIGN_CKPT_ROOT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("ODESIGN_DATA_ROOT_DIR", str(tmp_path / "data"))
    (tmp_path / "odesign").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ckpt").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


DATA_DIR = Path(__file__).resolve().parent / "data"


# ----- Healthcheck / manifest -----

def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "odesign"
    assert "version" in health
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "odesign"
    assert detail["version"] == health["version"]


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "odesign"


def test_manifest_lists_endpoint(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert paths == {"/api/design", "/api/tasks/design"}


def test_manifest_models(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    models = extras["models"]
    assert "odesign_base_prot_flex" in models
    assert "odesign_base_prot_rigid" in models
    assert "odesign_base_ligand_rigid" in models
    assert "odesign_base_na_rigid" in models
    assert len(models) == 4


def test_manifest_task_types(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    tasks = extras["task_types"]
    assert "binder_design" in tasks
    assert "rna_aptamer" in tasks
    assert "motif_scaffolding" in tasks


def test_manifest_has_endpoint_examples(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert by_path["/api/design"]["examples"]


# ----- Settings -----

def test_settings_defaults():
    from server.settings import ODesignSettings

    class _Off(ODesignSettings):
        model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")

    s = _Off()
    assert s.jobs_base_dir == Path("/data/odesign_jobs")
    assert s.root == Path("/opt/odesign/ODesign")
    assert s.python == "/opt/conda/envs/odesign/bin/python"
    assert s.inference_script == "/opt/odesign/ODesign/scripts/inference.py"
    assert s.ckpt_root_dir == Path("/opt/odesign/ckpt")
    assert s.data_root_dir == Path("/opt/odesign/data")
    assert s.oss_region == "cn-hangzhou"


def test_settings_env_override(monkeypatch):
    from server.settings import ODesignSettings
    monkeypatch.setenv("ODESIGN_PYTHON", "/custom/python")
    monkeypatch.setenv("ODESIGN_INFERENCE_SCRIPT", "/custom/inference.py")
    s = ODesignSettings()
    assert s.python == "/custom/python"
    assert s.inference_script == "/custom/inference.py"


# ----- Adapter -----

def test_adapter_name():
    from server.adapter import ODesignAdapter
    from server.settings import ODesignSettings

    class _Off(ODesignSettings):
        model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")

    a = ODesignAdapter(settings=_Off())
    assert a.name == "odesign"


def test_detect_outputs_with_cif(tmp_path):
    from server.adapter import ODesignAdapter
    from server.settings import ODesignSettings

    class _Off(ODesignSettings):
        model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")

    a = ODesignAdapter(settings=_Off())
    job = tmp_path / "j"
    preds = job / "output" / "fc_test" / "seed_42" / "predictions"
    preds.mkdir(parents=True)
    (preds / "fc_test_seed_42_bb_0_seq_0.cif").write_text("data_test\n")
    assert a.detect_outputs(job) is True


def test_detect_outputs_empty(tmp_path):
    from server.adapter import ODesignAdapter
    from server.settings import ODesignSettings

    class _Off(ODesignSettings):
        model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")

    a = ODesignAdapter(settings=_Off())
    job = tmp_path / "j"
    (job / "output").mkdir(parents=True)
    assert a.detect_outputs(job) is False


# ----- Request models -----

def test_design_request_defaults():
    from server.models import DesignRequest
    r = DesignRequest()
    assert r.model == "odesign_base_prot_flex"
    assert r.design_modality is None
    assert r.n_sample == 5
    assert r.seeds == "[42]"
    assert r.num_workers == 4
    assert r.invfold_topk == 1
    assert r.invfold_temp == 1.0
    assert r.enable_partial_diff is False
    assert r.partial_diff_snr == 0.1


def test_design_request_custom():
    from server.models import DesignRequest
    r = DesignRequest(
        model="odesign_base_na_rigid",
        design_modality="rna",
        n_sample=10,
        seeds="[42,101]",
        invfold_topk=3,
    )
    assert r.model == "odesign_base_na_rigid"
    assert r.design_modality == "rna"
    assert r.n_sample == 10
    assert r.seeds == "[42,101]"
    assert r.invfold_topk == 3


# ----- argv builders -----

def test_design_argv_prot_flex(tmp_path):
    from server.models import DesignRequest
    from server.settings import ODesignSettings
    from server.tools import design_argv

    class _Off(ODesignSettings):
        model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")

    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    json_path = tmp_path / "input.json"
    json_path.write_text('[{"name":"test"}]')

    argv = design_argv(DesignRequest(), job_dir=job_dir, json_path=json_path, settings=s)

    assert argv[0] == s.python
    assert argv[1] == s.inference_script
    assert "exp=train_odesign_base_prot_flex" in argv
    assert "exp.design_modality=protein" in argv
    assert f"exp.input_json_path={json_path}" in argv
    assert "exp.model.sample_diffusion.N_sample=5" in argv
    assert "exp.use_msa=false" in argv
    assert any(a.startswith("hydra.run.dir=") for a in argv)


def test_design_argv_na_requires_modality(tmp_path):
    from server.models import DesignRequest
    from server.settings import ODesignSettings
    from server.tools import design_argv

    class _Off(ODesignSettings):
        model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")

    req = DesignRequest(model="odesign_base_na_rigid")
    with pytest.raises(ValueError, match="design_modality is required"):
        design_argv(req, job_dir=tmp_path, json_path=tmp_path / "x.json", settings=_Off())


def test_design_argv_na_with_modality(tmp_path):
    from server.models import DesignRequest
    from server.settings import ODesignSettings
    from server.tools import design_argv

    class _Off(ODesignSettings):
        model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")

    s = _Off()
    req = DesignRequest(model="odesign_base_na_rigid", design_modality="rna")
    argv = design_argv(req, job_dir=tmp_path, json_path=tmp_path / "x.json", settings=s)
    assert "exp=train_odesign_base_na_rigid" in argv
    assert "exp.design_modality=rna" in argv


def test_design_argv_ligand(tmp_path):
    from server.models import DesignRequest
    from server.settings import ODesignSettings
    from server.tools import design_argv

    class _Off(ODesignSettings):
        model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")

    req = DesignRequest(model="odesign_base_ligand_rigid")
    argv = design_argv(req, job_dir=tmp_path, json_path=tmp_path / "x.json", settings=_Off())
    assert "exp=train_odesign_base_ligand_rigid" in argv
    assert "exp.design_modality=ligand" in argv


def test_design_argv_partial_diff(tmp_path):
    from server.models import DesignRequest
    from server.settings import ODesignSettings
    from server.tools import design_argv

    class _Off(ODesignSettings):
        model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")

    req = DesignRequest(enable_partial_diff=True, partial_diff_snr=0.5)
    argv = design_argv(req, job_dir=tmp_path, json_path=tmp_path / "x.json", settings=_Off())
    assert any("partial_diffusion.enable=true" in a for a in argv)
    assert any("partial_diffusion.snr=0.5" in a for a in argv)


# ----- URI resolution -----

def test_uri_resolve_file(tmp_path):
    from server.settings import ODesignSettings
    from server.uris import resolve_input

    class _Off(ODesignSettings):
        model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")

    src = tmp_path / "input.json"
    src.write_text('[{"name":"test"}]')
    dest = tmp_path / "dest.json"
    out = resolve_input(None, f"file://{src}", dest, _Off())
    assert out.read_text() == '[{"name":"test"}]'


def test_uri_requires_input(tmp_path):
    from fastapi import HTTPException
    from server.settings import ODesignSettings
    from server.uris import resolve_input

    class _Off(ODesignSettings):
        model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")

    with pytest.raises(HTTPException) as exc:
        resolve_input(None, None, tmp_path / "x", _Off())
    assert exc.value.status_code == 422


# ----- JSON ref_file rewriting -----

def test_rewrite_ref_files(tmp_path):
    from server.uris import rewrite_ref_files

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    json_path = input_dir / "input.json"
    json_path.write_text(json.dumps([
        {"name": "test", "ref_file": "./examples/target.pdb"},
        {"name": "test2", "ref_file": ""},
    ]))

    rewrite_ref_files(json_path, input_dir)

    result = json.loads(json_path.read_text())
    assert result[0]["ref_file"] == str(input_dir / "target.pdb")
    assert result[1]["ref_file"] == ""


def test_rewrite_ref_files_no_ref(tmp_path):
    from server.uris import rewrite_ref_files

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    json_path = input_dir / "input.json"
    original = json.dumps([{"name": "test"}])
    json_path.write_text(original)

    rewrite_ref_files(json_path, input_dir)

    assert json_path.read_text() == original


# ----- endpoint smoke (no real pipeline) -----

def test_design_endpoint_returns_job(client):
    with open(DATA_DIR / "fc_design.json", "rb") as fh:
        resp = client.post(
            "/api/design",
            data={
                "model": "odesign_base_prot_flex",
                "n_sample": "2",
                "seeds": "[42]",
            },
            files={
                "input_json": ("fc_design.json", fh, "application/json"),
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["created_at"] is not None
    assert body["input_params"] is not None
    assert body["input_params"]["model"] == "odesign_base_prot_flex"
    assert body["input_params"]["n_sample"] == 2


def test_design_endpoint_na_model(client):
    with open(DATA_DIR / "fc_design.json", "rb") as fh:
        resp = client.post(
            "/api/design",
            data={
                "model": "odesign_base_na_rigid",
                "design_modality": "rna",
                "n_sample": "2",
            },
            files={
                "input_json": ("fc_design.json", fh, "application/json"),
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["input_params"]["model"] == "odesign_base_na_rigid"
    assert body["input_params"]["design_modality"] == "rna"


# ----- task endpoint smoke (synchronous; /bin/true so it returns immediately) -----

def test_design_task_endpoint_returns_terminal_status(client):
    """POST /api/tasks/design blocks until subprocess exits."""
    with open(DATA_DIR / "fc_design.json", "rb") as fh:
        resp = client.post(
            "/api/tasks/design",
            data={},
            files={"input_json": ("fc_design.json", fh, "application/json")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_design_task_endpoint_honors_job_id_header(client):
    with open(DATA_DIR / "fc_design.json", "rb") as fh:
        resp = client.post(
            "/api/tasks/design",
            data={},
            files={"input_json": ("fc_design.json", fh, "application/json")},
            headers={"X-Bioagent-Job-Id": "odesign-task-001"},
        )
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "odesign-task-001"


def test_design_task_endpoint_duplicate_returns_existing(client):
    hdrs = {"X-Bioagent-Job-Id": "odesign-dup-001"}
    with open(DATA_DIR / "fc_design.json", "rb") as fh:
        r1 = client.post(
            "/api/tasks/design",
            data={},
            files={"input_json": ("fc_design.json", fh, "application/json")},
            headers=hdrs,
        )
    with open(DATA_DIR / "fc_design.json", "rb") as fh:
        r2 = client.post(
            "/api/tasks/design",
            data={},
            files={"input_json": ("fc_design.json", fh, "application/json")},
            headers=hdrs,
        )
    assert r1.json()["job_id"] == r2.json()["job_id"]
    assert r1.json()["created_at"] == r2.json()["created_at"]

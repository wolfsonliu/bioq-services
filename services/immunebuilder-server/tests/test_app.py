"""Offline tests for immunebuilder-server (no real ImmuneBuilder needed).

Tests run with IMMUNEBUILDER_VENV_BIN pointing at a directory containing
a dummy `ABodyBuilder2` (just /bin/true), so no GPU or model weights needed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


# =====================================================================
# Example sequences for tests (short but valid single-letter AA codes)
# =====================================================================

HEAVY_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
LIGHT_SEQ = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
NANOBODY_SEQ = "QVQLVESGGGLVQAGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAINSGGGSTYYPDSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAKDGYGGSFDYWGQGTQVTVSS"
ALPHA_SEQ = "METLLGVSLVILWLQLARVNSQQGEEDPQALSIQEGENATMNCSYKTSINNLQWYRQNSGRGLVHLILIRSNEREKHSGRLRVTLDTSKKSSSLLITASRAADTASYFCAVNFGGGKLIFGQGTELSVKP"
BETA_SEQ = "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSFSTCSANYGYTFGSGTRLTVVEDL"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("IMMUNEBUILDER_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("IMMUNEBUILDER_VENV_BIN", str(tmp_path / "bin"))
    (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
    # Create dummy CLI entry points so argv construction works
    for name in ("ABodyBuilder2", "NanoBodyBuilder2", "TCRBuilder2"):
        (tmp_path / "bin" / name).write_text("#!/bin/sh\ntrue\n")
        (tmp_path / "bin" / name).chmod(0o755)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# =====================================================================
# Healthcheck / manifest
# =====================================================================

def test_healthz(client):
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "immunebuilder"
    assert "version" in body


def test_healthz_detail(client):
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "immunebuilder"


def test_manifest_lists_three_endpoints(client):
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/predict_antibody", "/api/predict_nanobody", "/api/predict_tcr"}


def test_manifest_predictors(client):
    body = client.get("/api/manifest").json()
    predictors = body["service_specific"]["predictors"]
    assert "antibody" in predictors
    assert "nanobody" in predictors
    assert "tcr" in predictors


def test_manifest_tool_outputs(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "final_model" in extras["tool_outputs"]
    assert "unrefined_models" in extras["tool_outputs"]
    assert "error_estimates" in extras["tool_outputs"]


def test_manifest_numbering_schemes(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    schemes = extras["numbering_schemes"]
    assert "imgt" in schemes
    assert "chothia" in schemes


def test_endpoint_examples(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert by_path["/api/predict_antibody"]["examples"]
    assert by_path["/api/predict_nanobody"]["examples"]
    assert by_path["/api/predict_tcr"]["examples"]


def test_openapi_served(client):
    r = client.get("/openapi.json")
    r.raise_for_status()
    schema = r.json()
    assert "paths" in schema


# =====================================================================
# Settings
# =====================================================================

def test_settings_defaults():
    from server.settings import ImmuneBuilderSettings

    class _Off(ImmuneBuilderSettings):
        model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")

    s = _Off()
    assert s.jobs_base_dir == Path("/data/immunebuilder_jobs")
    assert s.venv_bin == Path("/opt/conda/envs/immunebuilder/bin")
    assert s.oss_region == "cn-hangzhou"


def test_settings_env_override(monkeypatch):
    from server.settings import ImmuneBuilderSettings
    monkeypatch.setenv("IMMUNEBUILDER_VENV_BIN", "/custom/bin")
    s = ImmuneBuilderSettings()
    assert s.venv_bin == Path("/custom/bin")


# =====================================================================
# Adapter
# =====================================================================

def test_adapter_name():
    from server.adapter import ImmuneBuilderAdapter
    from server.settings import ImmuneBuilderSettings

    class _Off(ImmuneBuilderSettings):
        model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")

    a = ImmuneBuilderAdapter(settings=_Off())
    assert a.name == "immunebuilder"


def test_detect_outputs_pdb(tmp_path):
    from server.adapter import ImmuneBuilderAdapter
    from server.settings import ImmuneBuilderSettings

    class _Off(ImmuneBuilderSettings):
        model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")

    a = ImmuneBuilderAdapter(settings=_Off())
    job = tmp_path / "j"
    out = job / "output"
    out.mkdir(parents=True)
    (out / "final_model.pdb").write_text("REMARK ImmuneBuilder\nATOM\n")
    assert a.detect_outputs(job) is True


def test_detect_outputs_empty(tmp_path):
    from server.adapter import ImmuneBuilderAdapter
    from server.settings import ImmuneBuilderSettings

    class _Off(ImmuneBuilderSettings):
        model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")

    a = ImmuneBuilderAdapter(settings=_Off())
    job = tmp_path / "j"
    (job / "output").mkdir(parents=True)
    assert a.detect_outputs(job) is False


# =====================================================================
# FASTA construction
# =====================================================================

def test_write_fasta_antibody(tmp_path):
    from server.tools import write_fasta
    dest = tmp_path / "input.fasta"
    write_fasta({"H": "EVQL", "L": "DIQM"}, dest)
    content = dest.read_text()
    assert ">H\nEVQL\n" in content
    assert ">L\nDIQM\n" in content


def test_write_fasta_nanobody(tmp_path):
    from server.tools import write_fasta
    dest = tmp_path / "input.fasta"
    write_fasta({"H": "QVQL"}, dest)
    content = dest.read_text()
    assert ">H\nQVQL\n" in content
    assert ">L" not in content


def test_write_fasta_tcr(tmp_path):
    from server.tools import write_fasta
    dest = tmp_path / "input.fasta"
    write_fasta({"A": "METL", "B": "NAGV"}, dest)
    content = dest.read_text()
    assert ">A\nMETL\n" in content
    assert ">B\nNAGV\n" in content


# =====================================================================
# argv builders
# =====================================================================

def test_predict_antibody_argv(tmp_path):
    from server.models import AntibodyRequest
    from server.settings import ImmuneBuilderSettings
    from server.tools import predict_antibody_argv

    class _Off(ImmuneBuilderSettings):
        model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")

    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">H\nEVQL\n>L\nDIQM\n")

    argv = predict_antibody_argv(
        AntibodyRequest(heavy_sequence=HEAVY_SEQ, light_sequence=LIGHT_SEQ),
        job_dir=job_dir, fasta_path=fasta, settings=s,
    )
    assert "ABodyBuilder2" in argv[0]
    assert "-f" in argv
    assert "--to_directory" in argv
    assert "-n" in argv
    idx = argv.index("-n")
    assert argv[idx + 1] == "imgt"


def test_predict_nanobody_argv(tmp_path):
    from server.models import NanobodyRequest
    from server.settings import ImmuneBuilderSettings
    from server.tools import predict_nanobody_argv

    class _Off(ImmuneBuilderSettings):
        model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")

    argv = predict_nanobody_argv(
        NanobodyRequest(heavy_sequence=NANOBODY_SEQ),
        job_dir=tmp_path,
        fasta_path=tmp_path / "input.fasta",
        settings=_Off(),
    )
    assert "NanoBodyBuilder2" in argv[0]


def test_predict_tcr_argv(tmp_path):
    from server.models import TCRRequest
    from server.settings import ImmuneBuilderSettings
    from server.tools import predict_tcr_argv

    class _Off(ImmuneBuilderSettings):
        model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")

    argv = predict_tcr_argv(
        TCRRequest(alpha_sequence=ALPHA_SEQ, beta_sequence=BETA_SEQ),
        job_dir=tmp_path,
        fasta_path=tmp_path / "input.fasta",
        settings=_Off(),
    )
    assert "TCRBuilder2" in argv[0]


def test_predict_argv_no_save_all(tmp_path):
    from server.models import AntibodyRequest
    from server.settings import ImmuneBuilderSettings
    from server.tools import predict_antibody_argv

    class _Off(ImmuneBuilderSettings):
        model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")

    argv = predict_antibody_argv(
        AntibodyRequest(
            heavy_sequence=HEAVY_SEQ,
            light_sequence=LIGHT_SEQ,
            save_all_models=False,
        ),
        job_dir=tmp_path,
        fasta_path=tmp_path / "input.fasta",
        settings=_Off(),
    )
    assert "--to_directory" not in argv
    assert any("final_model.pdb" in a for a in argv)


def test_predict_argv_no_sidechain_check(tmp_path):
    from server.models import AntibodyRequest
    from server.settings import ImmuneBuilderSettings
    from server.tools import predict_antibody_argv

    class _Off(ImmuneBuilderSettings):
        model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")

    argv = predict_antibody_argv(
        AntibodyRequest(
            heavy_sequence=HEAVY_SEQ,
            light_sequence=LIGHT_SEQ,
            no_sidechain_bond_check=True,
        ),
        job_dir=tmp_path,
        fasta_path=tmp_path / "input.fasta",
        settings=_Off(),
    )
    assert "-u" in argv


def test_predict_argv_n_threads(tmp_path):
    from server.models import AntibodyRequest
    from server.settings import ImmuneBuilderSettings
    from server.tools import predict_antibody_argv

    class _Off(ImmuneBuilderSettings):
        model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")

    argv = predict_antibody_argv(
        AntibodyRequest(
            heavy_sequence=HEAVY_SEQ,
            light_sequence=LIGHT_SEQ,
            n_threads=4,
        ),
        job_dir=tmp_path,
        fasta_path=tmp_path / "input.fasta",
        settings=_Off(),
    )
    assert "--n_threads" in argv
    idx = argv.index("--n_threads")
    assert argv[idx + 1] == "4"


def test_predict_argv_numbering_scheme(tmp_path):
    from server.models import AntibodyRequest
    from server.settings import ImmuneBuilderSettings
    from server.tools import predict_antibody_argv

    class _Off(ImmuneBuilderSettings):
        model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")

    argv = predict_antibody_argv(
        AntibodyRequest(
            heavy_sequence=HEAVY_SEQ,
            light_sequence=LIGHT_SEQ,
            numbering_scheme="chothia",
        ),
        job_dir=tmp_path,
        fasta_path=tmp_path / "input.fasta",
        settings=_Off(),
    )
    idx = argv.index("-n")
    assert argv[idx + 1] == "chothia"


# =====================================================================
# Request model validation
# =====================================================================

def test_antibody_request_defaults():
    from server.models import AntibodyRequest
    r = AntibodyRequest(heavy_sequence=HEAVY_SEQ, light_sequence=LIGHT_SEQ)
    assert r.numbering_scheme == "imgt"
    assert r.save_all_models is True
    assert r.no_sidechain_bond_check is False
    assert r.n_threads == -1
    assert r.name == "prediction"


def test_antibody_request_rejects_bad_aa():
    from pydantic import ValidationError
    from server.models import AntibodyRequest
    with pytest.raises(ValidationError):
        AntibodyRequest(heavy_sequence="EVQLX" + "A" * 20, light_sequence=LIGHT_SEQ)


def test_antibody_request_rejects_short_seq():
    from pydantic import ValidationError
    from server.models import AntibodyRequest
    with pytest.raises(ValidationError):
        AntibodyRequest(heavy_sequence="EVQL", light_sequence=LIGHT_SEQ)


def test_tcr_request_fields():
    from server.models import TCRRequest
    r = TCRRequest(alpha_sequence=ALPHA_SEQ, beta_sequence=BETA_SEQ)
    assert r.alpha_sequence == ALPHA_SEQ
    assert r.beta_sequence == BETA_SEQ


# =====================================================================
# Endpoint smoke (no real pipeline — /bin/true exits immediately)
# =====================================================================

def test_antibody_endpoint_returns_job(client):
    resp = client.post(
        "/api/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "name": "test_ab",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["input_params"]["name"] == "test_ab"
    assert body["input_params"]["heavy_sequence"] == HEAVY_SEQ
    assert body["input_params"]["light_sequence"] == LIGHT_SEQ


def test_nanobody_endpoint_returns_job(client):
    resp = client.post(
        "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "test_nb",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["input_params"]["heavy_sequence"] == NANOBODY_SEQ


def test_tcr_endpoint_returns_job(client):
    resp = client.post(
        "/api/predict_tcr",
        data={
            "alpha_sequence": ALPHA_SEQ,
            "beta_sequence": BETA_SEQ,
            "name": "test_tcr",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["input_params"]["alpha_sequence"] == ALPHA_SEQ
    assert body["input_params"]["beta_sequence"] == BETA_SEQ


def test_unknown_job_returns_404(client):
    assert client.get("/api/jobs/nonexistent-id").status_code == 404

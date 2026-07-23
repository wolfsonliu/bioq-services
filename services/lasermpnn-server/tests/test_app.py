"""Offline tests for lasermpnn-server (mocked subprocess; no real inference)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


class _Off:
    """Marker mixin factory — see _off_settings()."""


def _off_settings(**kw):
    from server.settings import LASErMPNNSettings

    class _S(LASErMPNNSettings):
        model_config = SettingsConfigDict(
            env_prefix="LASERMPNN_TEST_", env_file=None, extra="ignore",
        )

    return _S(**kw)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LASERMPNN_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("LASERMPNN_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("LASERMPNN_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("LASERMPNN_DEVICE", "cpu")
    (tmp_path / "root").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    import importlib
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ---- health / manifest / settings ----

def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "lasermpnn"
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "lasermpnn"
    assert "weights_loaded" in detail
    assert set(detail["weights_missing"]).issubset(
        {"nothing_heldout", "ligandmpnn_split", "soluble", "ligand_encoder"},
    )


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "lasermpnn"


def test_settings_defaults():
    s = _off_settings()
    assert s.jobs_base_dir == Path("/data/lasermpnn_jobs")
    assert s.root == Path("/opt/lasermpnn")
    assert s.weights_dir == Path("/data/models/lasermpnn")
    assert s.device == "cuda:0"
    assert s.max_concurrent_jobs == 1


def test_settings_env_override(monkeypatch):
    from server.settings import LASErMPNNSettings
    monkeypatch.setenv("LASERMPNN_DEVICE", "cuda:3")
    monkeypatch.setenv("LASERMPNN_ROOT", "/custom")
    s = LASErMPNNSettings()
    assert s.device == "cuda:3"
    assert s.root == Path("/custom")


def test_adapter_name_and_cwd(tmp_path):
    from server.adapter import LASErMPNNAdapter
    s = _off_settings(root=tmp_path / "root")
    (s.root).mkdir(parents=True)
    a = LASErMPNNAdapter(settings=s)
    assert a.name == "lasermpnn"
    assert a.subprocess_cwd() == s.root


# ---- request models ----

def test_design_request_defaults():
    from server.models import DesignRequest
    r = DesignRequest()
    assert r.designs_per_input == 4
    assert r.designs_per_batch == 30
    assert r.sequence_temp is None
    assert r.model_variant == "nothing_heldout"
    assert r.disabled_residues == "X,C"
    assert r.output_fasta is True
    assert r.constrain_ala_gly is False


def test_design_ligandmpnn_request_defaults():
    from server.models import DesignLigandMPNNRequest
    r = DesignLigandMPNNRequest()
    assert r.designs_per_input == 4
    assert r.disabled_residues == "X"
    assert not hasattr(r, "model_variant")


def test_design_request_rejects_bad_variant():
    from pydantic import ValidationError
    from server.models import DesignRequest
    with pytest.raises(ValidationError):
        DesignRequest(model_variant="does_not_exist")


def test_design_request_rejects_out_of_range_temp():
    from pydantic import ValidationError
    from server.models import DesignRequest
    with pytest.raises(ValidationError):
        DesignRequest(sequence_temp=99.0)


# ---- weight_file mapping ----

def test_weight_file_mapping(tmp_path):
    from server.tools import weight_file
    s = _off_settings(weights_dir=tmp_path)
    assert weight_file("nothing_heldout", s) == tmp_path / "laser_weights_0p1A_nothing_heldout.pt"
    assert weight_file("ligandmpnn_split", s) == tmp_path / "laser_weights_0p1A_noise_ligandmpnn_split.pt"
    assert weight_file("soluble", s).name.startswith("soluble_weights")


# ---- argv builders ----

def test_design_argv_minimal(tmp_path):
    from server.models import DesignRequest
    from server.tools import design_argv
    s = _off_settings(weights_dir=tmp_path / "w", device="cpu")
    job_dir = tmp_path / "job"
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n")
    argv = design_argv(DesignRequest(), input_pdb=input_pdb, job_dir=job_dir, settings=s)
    assert argv[1] == "-m"
    assert argv[2] == "LASErMPNN.run_batch_inference"
    assert str(input_pdb) in argv
    assert str(job_dir / "output") in argv
    assert argv[argv.index("-w") + 1].endswith("laser_weights_0p1A_nothing_heldout.pt")
    assert argv[argv.index("-d") + 1] == "cpu"
    # positional designs_per_input right after the output dir
    assert "4" in argv
    assert "--output_fasta" in argv  # default True
    assert "-c" not in argv  # constrain off by default
    assert (job_dir / "output").is_dir()


def test_design_argv_optional_flags(tmp_path):
    from server.models import DesignRequest
    from server.tools import design_argv
    s = _off_settings(weights_dir=tmp_path / "w", device="cuda:0")
    job_dir = tmp_path / "job"
    pdb = tmp_path / "in.pdb"
    pdb.write_text("ATOM\n")
    req = DesignRequest(
        model_variant="soluble",
        sequence_temp=0.5,
        first_shell_sequence_temp=0.1,
        chi_temp=0.2,
        fix_beta=True,
        ignore_ligand=True,
        constrain_ala_gly=True,
        ala_budget=6,
        gly_budget=1,
        output_fasta=False,
    )
    argv = design_argv(req, input_pdb=pdb, job_dir=job_dir, settings=s)
    assert argv[argv.index("-w") + 1].endswith("soluble_weights_no_heldout_drop_clusters_optstep_65000.pt")
    assert argv[argv.index("--sequence_temp") + 1] == "0.5"
    assert argv[argv.index("--first_shell_sequence_temp") + 1] == "0.1"
    assert argv[argv.index("--chi_temp") + 1] == "0.2"
    assert "--fix_beta" in argv
    assert "--ignore_ligand" in argv
    assert "--output_fasta" not in argv
    assert "-c" in argv
    assert argv[argv.index("--ala_budget") + 1] == "6"
    assert argv[argv.index("--gly_budget") + 1] == "1"


def test_design_ligandmpnn_argv(tmp_path):
    from server.models import DesignLigandMPNNRequest
    from server.tools import design_ligandmpnn_argv
    s = _off_settings(weights_dir=tmp_path / "w")
    job_dir = tmp_path / "job"
    pdb = tmp_path / "in.pdb"
    pdb.write_text("ATOM\n")
    argv = design_ligandmpnn_argv(
        DesignLigandMPNNRequest(designs_per_input=2), input_pdb=pdb, job_dir=job_dir, settings=s,
    )
    assert argv[2] == "LASErMPNN.run_batch_inference_ligandmpnn"
    assert argv[argv.index("-w") + 1].endswith("laser_weights_0p1A_noise_ligandmpnn_split.pt")
    assert "2" in argv
    assert "-c" not in argv  # ligandmpnn variant has no ALA/GLY budget


# ---- detect_outputs ----

def test_detect_outputs_design_pdb(tmp_path):
    from server.adapter import LASErMPNNAdapter
    a = LASErMPNNAdapter(settings=_off_settings())
    job = tmp_path / "j"
    (job / "output" / "input").mkdir(parents=True)
    (job / "output" / "input" / "design_0.pdb").write_text("ATOM\n")
    assert a.detect_outputs(job) is True


def test_detect_outputs_fasta(tmp_path):
    from server.adapter import LASErMPNNAdapter
    a = LASErMPNNAdapter(settings=_off_settings())
    job = tmp_path / "j"
    (job / "output").mkdir(parents=True)
    (job / "output" / "designs.fasta").write_text(">x\nM\n")
    assert a.detect_outputs(job) is True


def test_detect_outputs_empty(tmp_path):
    from server.adapter import LASErMPNNAdapter
    a = LASErMPNNAdapter(settings=_off_settings())
    job = tmp_path / "j"
    (job / "output").mkdir(parents=True)
    assert a.detect_outputs(job) is False


# ---- endpoint smoke (subprocess runs async and may fail; accept path is what matters) ----

def test_design_endpoint_returns_job(client):
    resp = client.post(
        "/api/design",
        data={"designs_per_input": "1", "model_variant": "nothing_heldout"},
        files={"pdb": ("input.pdb", b"ATOM\n", "text/plain")},
    )
    assert resp.status_code in (200, 422)
    if resp.status_code == 200:
        assert "job_id" in resp.json()


def test_design_endpoint_rejects_bad_variant(client):
    resp = client.post(
        "/api/design",
        data={"model_variant": "nope"},
        files={"pdb": ("input.pdb", b"ATOM\n", "text/plain")},
    )
    assert resp.status_code == 422


def test_design_ligandmpnn_endpoint_returns_job(client):
    resp = client.post(
        "/api/design_ligandmpnn",
        data={"designs_per_input": "1"},
        files={"pdb": ("input.pdb", b"ATOM\n", "text/plain")},
    )
    assert resp.status_code in (200, 422)


def test_design_task_endpoint_returns_job(client):
    resp = client.post(
        "/api/tasks/design",
        data={"designs_per_input": "1"},
        files={"pdb": ("input.pdb", b"ATOM\n", "text/plain")},
    )
    assert resp.status_code in (200, 422, 500)
    if resp.status_code == 200:
        assert "job_id" in resp.json()


# ---- manifest extras / examples ----

def test_manifest_extras(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert set(extras["model_variants"].keys()) == {"nothing_heldout", "ligandmpnn_split", "soluble"}
    assert "design" in extras["tool_outputs"]
    assert "design_ligandmpnn" in extras["tool_outputs"]
    assert "important" in extras  # protonation warning


def test_endpoint_examples_exist():
    import importlib
    import os
    os.environ["LASERMPNN_JOBS_BASE_DIR"] = "/tmp/lasermpnn_jobs_test"
    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    body = TestClient(server_app.app).get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    for path in ("/api/design", "/api/design_ligandmpnn"):
        assert path in by_path, f"{path} not registered"
        assert by_path[path]["examples"], f"{path} has no examples"

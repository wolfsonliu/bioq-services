"""Offline tests for openbpmd-server (subprocess stubbed via OPENBPMD_PYTHON=/bin/true).

Real OpenMM / OpenBPMD is never invoked here. The `/healthz/detail` OpenMM
probe is monkeypatched.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


# ---------------------------------------------------------------------------
# Fixture bytes (content is irrelevant — subprocess is /bin/true)
# ---------------------------------------------------------------------------


def _rst7_bytes() -> bytes:
    return b"default_name\n     3\n  1.0  2.0  3.0  4.0  5.0  6.0\n"


def _prm7_bytes() -> bytes:
    return b"%VERSION  VERSION_STAMP = V0001.000\n%FLAG TITLE\n"


def _gro_bytes() -> bytes:
    return b"test\n    3\n    1MOL    C1    1   0.000   0.000   0.000\n  1.0 1.0 1.0\n"


def _top_bytes() -> bytes:
    return b"; topology\n[ defaults ]\n1 2 yes\n"


# ---------------------------------------------------------------------------
# Client fixture — recreates the app in an isolated tmp env.
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBPMD_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("OPENBPMD_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("OPENBPMD_PYTHON", "/bin/true")
    monkeypatch.setenv("OPENBPMD_INFERENCE_SCRIPT", str(tmp_path / "inference.py"))
    monkeypatch.setenv("OPENBPMD_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("OPENBPMD_MAX_CONCURRENT_JOBS", "1")
    monkeypatch.setenv("OPENBPMD_TASK_ENDPOINTS_ENABLED", "true")
    monkeypatch.setenv("OPENBPMD_PLATFORM", "CPU")
    (tmp_path / "root").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ---------------------------------------------------------------------------
# Health / manifest
# ---------------------------------------------------------------------------


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "openbpmd"


def test_healthz_detail_probes_openmm(client, monkeypatch):
    import server.app as app_mod
    monkeypatch.setattr(app_mod, "_probe_openmm", lambda: ("8.1.1", ["Reference", "CPU", "CUDA"]))
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["openmm_version"] == "8.1.1"
    assert body["cuda_available"] is True
    assert body["weights_loaded"] is True
    assert body["task_endpoints_enabled"] is True


def test_healthz_detail_no_cuda(client, monkeypatch):
    import server.app as app_mod
    monkeypatch.setattr(app_mod, "_probe_openmm", lambda: ("8.1.1", ["Reference", "CPU"]))
    body = client.get("/healthz/detail").json()
    assert body["cuda_available"] is False


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "openbpmd"


def test_manifest_lists_score_endpoint(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/score" in paths


def test_task_endpoint_registered(client):
    paths = set(client.get("/openapi.json").json()["paths"].keys())
    assert "/api/tasks/score" in paths


def test_manifest_extras_have_score_semantics(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["long_running"] is True
    assert "CompScore" in extras["model"]["score_semantics"]
    assert "results.csv" in extras["tool_outputs"]


def test_manifest_examples_have_curl(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert len(by_path["/api/score"]["examples"]) >= 1
    assert any(
        "solvated" in (e.get("curl") or "")
        for e in by_path["/api/score"]["examples"]
    )


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_structure_returns_422(client):
    r = client.post(
        "/api/score",
        files={"parameters": ("solvated.prm7", _prm7_bytes(), "application/octet-stream")},
        data={"lig_resname": "MOL"},
    )
    assert r.status_code == 422


def test_missing_parameters_returns_422(client):
    r = client.post(
        "/api/score",
        files={"structure": ("solvated.rst7", _rst7_bytes(), "application/octet-stream")},
    )
    assert r.status_code == 422


def test_bad_lig_resname_returns_422(client):
    r = client.post(
        "/api/score",
        files={
            "structure": ("solvated.rst7", _rst7_bytes(), "application/octet-stream"),
            "parameters": ("solvated.prm7", _prm7_bytes(), "application/octet-stream"),
        },
        data={"lig_resname": "has space"},
    )
    assert r.status_code == 422


def test_hill_height_out_of_range_returns_422(client):
    r = client.post(
        "/api/score",
        files={
            "structure": ("solvated.rst7", _rst7_bytes(), "application/octet-stream"),
            "parameters": ("solvated.prm7", _prm7_bytes(), "application/octet-stream"),
        },
        data={"hill_height": "9.0"},
    )
    assert r.status_code == 422


def test_nreps_out_of_range_returns_422(client):
    r = client.post(
        "/api/score",
        files={
            "structure": ("solvated.rst7", _rst7_bytes(), "application/octet-stream"),
            "parameters": ("solvated.prm7", _prm7_bytes(), "application/octet-stream"),
        },
        data={"nreps": "50"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Smoke (subprocess stubbed via /bin/true)
# ---------------------------------------------------------------------------


def test_score_submit_returns_job(client):
    r = client.post(
        "/api/score",
        files={
            "structure": ("solvated.rst7", _rst7_bytes(), "application/octet-stream"),
            "parameters": ("solvated.prm7", _prm7_bytes(), "application/octet-stream"),
        },
        data={"lig_resname": "MOL", "nreps": "2", "hill_height": "0.3"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["lig_resname"] == "MOL"
    assert body["input_params"]["nreps"] == 2


def test_score_submit_gromacs(client):
    r = client.post(
        "/api/score",
        files={
            "structure": ("solvated.gro", _gro_bytes(), "application/octet-stream"),
            "parameters": ("solvated.top", _top_bytes(), "application/octet-stream"),
        },
        data={"lig_resname": "LIG", "system_format": "gromacs"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["input_params"]["system_format"] == "gromacs"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _off_settings():
    from server.settings import OpenBPMDSettings

    class _S(OpenBPMDSettings):
        model_config = SettingsConfigDict(
            env_prefix="OPENBPMD_TEST_",
            env_file=None,
            extra="ignore",
        )

    return _S


def test_settings_defaults():
    s = _off_settings()()
    assert s.jobs_base_dir == Path("/data/openbpmd_jobs")
    assert s.root == Path("/opt/openbpmd")
    assert s.python == "/opt/conda/envs/openbpmd/bin/python"
    assert s.weights_dir == Path("/data/models/openbpmd")
    assert s.platform == "CUDA"
    assert s.max_concurrent_jobs == 1
    # Modern GPU service — task endpoints default ON.
    assert s.task_endpoints_enabled is True
    assert s.subprocess_timeout_s == 48 * 3600


def test_settings_env_override(monkeypatch):
    from server.settings import OpenBPMDSettings
    monkeypatch.setenv("OPENBPMD_PLATFORM", "CPU")
    monkeypatch.setenv("OPENBPMD_MAX_CONCURRENT_JOBS", "2")
    s = OpenBPMDSettings()
    assert s.platform == "CPU"
    assert s.max_concurrent_jobs == 2


# ---------------------------------------------------------------------------
# tools.score_argv
# ---------------------------------------------------------------------------


def test_score_argv_amber_flags(tmp_path):
    from server.models import ScoreRequest
    from server.tools import score_argv

    s = _off_settings()(python="/bin/true", inference_script=str(tmp_path / "inference.py"))
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    structure = tmp_path / "solvated.rst7"
    structure.write_bytes(_rst7_bytes())
    parameters = tmp_path / "solvated.prm7"
    parameters.write_bytes(_prm7_bytes())

    argv = score_argv(
        ScoreRequest(lig_resname="MOL", nreps=3, hill_height=0.3),
        job_dir=job_dir,
        structure=structure,
        parameters=parameters,
        settings=s,
    )
    assert argv[0] == "/bin/true"
    assert str(s.inference_script) in argv
    assert "--structure" in argv and str(structure) in argv
    assert "--parameters" in argv and str(parameters) in argv
    assert "--lig-resname" in argv and "MOL" in argv
    assert "--nreps" in argv and "3" in argv
    assert "--platform" in argv and "CUDA" in argv
    # Advanced knobs unset → flags absent
    assert "--sim-ns" not in argv
    assert "--equil-steps" not in argv
    assert "--system-format" not in argv


def test_score_argv_advanced_knobs(tmp_path):
    from server.models import ScoreRequest
    from server.tools import score_argv

    s = _off_settings()()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    structure = tmp_path / "solvated.gro"
    structure.write_bytes(_gro_bytes())
    parameters = tmp_path / "solvated.top"
    parameters.write_bytes(_top_bytes())

    argv = score_argv(
        ScoreRequest(sim_ns=0.02, equil_steps=500, system_format="gromacs"),
        job_dir=job_dir,
        structure=structure,
        parameters=parameters,
        settings=s,
    )
    assert "--sim-ns" in argv and "0.02" in argv
    assert "--equil-steps" in argv and "500" in argv
    assert "--system-format" in argv and "gromacs" in argv


# ---------------------------------------------------------------------------
# inference.py: validation (no OpenMM import)
# ---------------------------------------------------------------------------


def test_inference_validate_amber_ok(tmp_path):
    from server import inference
    structure = tmp_path / "solvated.rst7"
    structure.write_bytes(_rst7_bytes())
    parameters = tmp_path / "solvated.prm7"
    parameters.write_bytes(_prm7_bytes())
    args = inference.parse_args([
        "--structure", str(structure),
        "--parameters", str(parameters),
        "--output-dir", str(tmp_path / "out"),
        "--lig-resname", "MOL",
    ])
    inference.validate(args)  # must not raise
    assert (tmp_path / "out").is_dir()


def test_inference_validate_missing_file_exits_2(tmp_path):
    from server import inference
    args = inference.parse_args([
        "--structure", str(tmp_path / "nope.rst7"),
        "--parameters", str(tmp_path / "nope.prm7"),
        "--output-dir", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as e:
        inference.validate(args)
    assert e.value.code == 2


def test_inference_validate_mixed_extensions_exits_2(tmp_path):
    from server import inference
    structure = tmp_path / "solvated.rst7"
    structure.write_bytes(_rst7_bytes())
    parameters = tmp_path / "solvated.top"  # gromacs parm with amber coord
    parameters.write_bytes(_top_bytes())
    args = inference.parse_args([
        "--structure", str(structure),
        "--parameters", str(parameters),
        "--output-dir", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as e:
        inference.validate(args)
    assert e.value.code == 2


def test_inference_detect_format(tmp_path):
    from server import inference
    gro = tmp_path / "s.gro"
    gro.write_bytes(_gro_bytes())
    top = tmp_path / "s.top"
    top.write_bytes(_top_bytes())
    args = inference.parse_args([
        "--structure", str(gro),
        "--parameters", str(top),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert inference._detect_format(args) == "gromacs"

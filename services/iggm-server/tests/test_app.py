"""Offline tests for iggm-server.

Real IgGM inference never runs offline — the subprocess is stubbed via
IGGM_PYTHON=/bin/true so no GPU / weights needed.  Fake (empty) .pth files are
staged so the endpoint-level checkpoint pre-check passes.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

DATA = Path(__file__).resolve().parent / "data"

CHECKPOINT_NAMES = (
    "esm_ppi_650m_ab",
    "antibody_design_trunk",
    "antibody_inverse_design_trunk",
    "antibody_fr_design_trunk",
    "igso3_buffer",
)


def _stage_ckpts(weights_dir: Path, names=CHECKPOINT_NAMES) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        (weights_dir / f"{n}.pth").write_bytes(b"\x00")


def _ab_files():
    """(fasta, antigen) multipart tuples for an antibody design call."""
    return {
        "fasta": ("ab.fasta", (DATA / "ab_CDR_H3.fasta").read_bytes(), "text/plain"),
        "antigen": ("antigen.pdb", (DATA / "antigen.pdb").read_bytes(), "text/plain"),
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("IGGM_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("IGGM_ROOT", str(tmp_path / "iggm"))
    monkeypatch.setenv("IGGM_PYTHON", "/bin/true")
    monkeypatch.setenv("IGGM_DESIGN_SCRIPT", str(tmp_path / "run_design.py"))
    monkeypatch.setenv("IGGM_EPITOPE_SCRIPT", str(tmp_path / "epitope.py"))
    monkeypatch.setenv("IGGM_WEIGHTS_DIR", str(tmp_path / "weights"))
    (tmp_path / "iggm").mkdir(parents=True, exist_ok=True)
    _stage_ckpts(tmp_path / "weights")

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Health / manifest -----


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "iggm"


def test_healthz_detail_all_ckpts(client):
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["weights_loaded"] is True
    assert body["checkpoints"]["esm_ppi_650m_ab"] is True
    assert body["weights_missing"] == {}


def test_healthz_detail_missing_ckpts(tmp_path, monkeypatch):
    monkeypatch.setenv("IGGM_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("IGGM_PYTHON", "/bin/true")
    monkeypatch.setenv("IGGM_WEIGHTS_DIR", str(tmp_path / "weights"))
    # only stage two of five
    _stage_ckpts(tmp_path / "weights", names=("esm_ppi_650m_ab", "igso3_buffer"))

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    c = TestClient(server_app.app)
    body = c.get("/healthz/detail").json()
    assert body["weights_loaded"] is False
    assert "antibody_design_trunk" in body["weights_missing"]


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "iggm"


def test_manifest_extras_have_model_info(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "IgGM"
    assert "design" in extras["tool_outputs"]
    assert "epitope" in extras["tool_outputs"]


def test_openapi_served(client):
    r = client.get("/openapi.json")
    r.raise_for_status()
    paths = r.json()["paths"]
    for p in (
        "/api/design",
        "/api/affinity-maturation",
        "/api/epitope",
        "/api/tasks/design",
        "/api/tasks/affinity-maturation",
        "/api/tasks/epitope",
    ):
        assert p in paths, p


# ----- Validation errors -----


def test_design_bad_run_task_returns_422(client):
    r = client.post(
        "/api/design", data={"run_task": "not_a_task"}, files=_ab_files()
    )
    assert r.status_code == 422


def test_design_steps_out_of_range_returns_422(client):
    r = client.post("/api/design", data={"steps": "500"}, files=_ab_files())
    assert r.status_code == 422


def test_design_num_samples_too_high_returns_422(client):
    r = client.post("/api/design", data={"num_samples": "1000"}, files=_ab_files())
    assert r.status_code == 422


def test_design_missing_checkpoints_returns_422(tmp_path, monkeypatch):
    """No staged weights → endpoint pre-check rejects with 422."""
    monkeypatch.setenv("IGGM_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("IGGM_PYTHON", "/bin/true")
    monkeypatch.setenv("IGGM_WEIGHTS_DIR", str(tmp_path / "empty_weights"))
    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    c = TestClient(server_app.app)
    r = c.post("/api/design", data={"run_task": "design"}, files=_ab_files())
    assert r.status_code == 422
    assert "Checkpoints missing" in r.text


def test_affinity_maturation_requires_fasta_origin(client):
    r = client.post(
        "/api/affinity-maturation", data={"num_samples": "5"}, files=_ab_files()
    )
    assert r.status_code == 422
    assert "fasta_origin" in r.text


# ----- Smoke (subprocess stubbed via /bin/true) -----


def test_design_returns_job_with_input_params(client):
    r = client.post(
        "/api/design",
        data={"run_task": "design", "steps": "5", "num_samples": "2", "seed": "42"},
        files=_ab_files(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["run_task"] == "design"
    assert body["input_params"]["steps"] == 5
    assert body["input_params"]["num_samples"] == 2
    assert body["input_params"]["seed"] == 42


def test_design_defaults_survive(client):
    r = client.post("/api/design", files=_ab_files())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_params"]["run_task"] == "design"
    assert body["input_params"]["steps"] == 10
    assert body["input_params"]["num_samples"] == 1


def test_design_with_epitope(client):
    r = client.post(
        "/api/design",
        data={"run_task": "design", "epitope": "[7, 8, 9, 10, 11]"},
        files=_ab_files(),
    )
    assert r.status_code == 200, r.text
    assert r.json()["input_params"]["epitope"] == [7, 8, 9, 10, 11]


def test_affinity_maturation_submits(client):
    files = _ab_files()
    files["fasta_origin"] = (
        "origin.fasta", (DATA / "ab_native.fasta").read_bytes(), "text/plain"
    )
    r = client.post("/api/affinity-maturation", data={"num_samples": "3"}, files=files)
    assert r.status_code == 200, r.text
    assert r.json()["input_params"]["num_samples"] == 3


def test_epitope_submits(client):
    r = client.post("/api/epitope", files=_ab_files())
    assert r.status_code == 200, r.text
    assert "job_id" in r.json()


# ----- Settings -----


def test_settings_defaults():
    from server.settings import IgGMSettings

    class _Off(IgGMSettings):
        model_config = SettingsConfigDict(
            env_prefix="IGGM_TEST_", env_file=None, extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/iggm_jobs")
    assert s.root == Path("/opt/iggm")
    assert s.weights_dir == Path("/data/models/iggm")
    assert s.max_concurrent_jobs == 1


def test_settings_missing_checkpoints_per_task(tmp_path):
    from server.settings import IgGMSettings

    class _Off(IgGMSettings):
        model_config = SettingsConfigDict(
            env_prefix="IGGM_TEST_", env_file=None, extra="ignore",
        )

    wd = tmp_path / "weights"
    _stage_ckpts(wd, names=("esm_ppi_650m_ab", "igso3_buffer"))
    s = _Off(weights_dir=wd)
    # design needs antibody_design_trunk (absent) + common (present)
    assert s.missing_checkpoints("design") == ["antibody_design_trunk"]
    # inverse_design needs the inverse trunk
    assert s.missing_checkpoints("inverse_design") == ["antibody_inverse_design_trunk"]


# ----- tools.argv builders -----


def test_design_argv_flags(tmp_path):
    from server.models import DesignRequest
    from server.settings import IgGMSettings
    from server.tools import design_argv

    class _Off(IgGMSettings):
        model_config = SettingsConfigDict(
            env_prefix="IGGM_TEST_", env_file=None, extra="ignore",
        )

    s = _Off(python="/opt/foo/python", design_script="/opt/foo/run_design.py")
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    argv = design_argv(
        DesignRequest(run_task="inverse_design", steps=5, num_samples=2, epitope=[7, 8]),
        job_dir=job_dir,
        fasta_path=Path("/in/ab.fasta"),
        antigen_path=Path("/in/ag.pdb"),
        settings=s,
        run_task="inverse_design",
    )
    assert argv[0] == "/opt/foo/python"
    assert "/opt/foo/run_design.py" in argv
    assert "--run_task" in argv
    assert argv[argv.index("--run_task") + 1] == "inverse_design"
    assert "--fasta" in argv and "/in/ab.fasta" in argv
    assert "--antigen" in argv and "/in/ag.pdb" in argv
    assert str(job_dir / "output") in argv
    assert "--epitope" in argv
    # epitope ints follow the flag
    ei = argv.index("--epitope")
    assert argv[ei + 1] == "7" and argv[ei + 2] == "8"
    # seed omitted when None
    assert "--seed" not in argv


def test_design_argv_affinity_with_origin(tmp_path):
    from server.models import AffinityMaturationRequest
    from server.settings import IgGMSettings
    from server.tools import design_argv

    class _Off(IgGMSettings):
        model_config = SettingsConfigDict(
            env_prefix="IGGM_TEST_", env_file=None, extra="ignore",
        )

    s = _Off()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    argv = design_argv(
        AffinityMaturationRequest(num_samples=5, seed=1),
        job_dir=job_dir,
        fasta_path=Path("/in/ab.fasta"),
        antigen_path=Path("/in/ag.pdb"),
        settings=s,
        run_task="affinity_maturation",
        fasta_origin_path=Path("/in/origin.fasta"),
    )
    assert argv[argv.index("--run_task") + 1] == "affinity_maturation"
    assert "--fasta_origin" in argv and "/in/origin.fasta" in argv
    assert "--seed" in argv and "1" in argv


def test_epitope_argv(tmp_path):
    from server.settings import IgGMSettings
    from server.tools import epitope_argv

    class _Off(IgGMSettings):
        model_config = SettingsConfigDict(
            env_prefix="IGGM_TEST_", env_file=None, extra="ignore",
        )

    s = _Off(python="/opt/foo/python", epitope_script="/opt/foo/epitope.py")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    argv = epitope_argv(
        job_dir=job_dir,
        fasta_path=Path("/in/c.fasta"),
        antigen_path=Path("/in/c.pdb"),
        settings=s,
    )
    assert argv[0] == "/opt/foo/python"
    assert "/opt/foo/epitope.py" in argv
    assert "--fasta" in argv and "/in/c.fasta" in argv
    assert str(job_dir / "output") in argv

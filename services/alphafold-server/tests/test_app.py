"""Offline tests for alphafold-server (no real AlphaFold model / GPU needed).

The endpoint handlers' subprocess call is replaced with a synchronous stub via
`monkeypatch` so we can exercise the FASTA-saving + argv-assembly logic
without a GPU or the actual AlphaFold model installed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


# ---- Shared fixtures ----


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAFOLD_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("ALPHAFOLD_ROOT", str(tmp_path / "alphafold"))
    monkeypatch.setenv("ALPHAFOLD_PYTHON", "/bin/true")
    monkeypatch.setenv("ALPHAFOLD_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "alphafold").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


@pytest.fixture
def captured_argv(monkeypatch):
    captured: list[dict] = []

    def _fake_submit(build_argv, label, **kwargs):
        from bioq_service import JobInfo, JobStatus

        job_id = f"stub-{label}-{len(captured)}"
        import tempfile

        job_dir = Path(tempfile.mkdtemp(prefix="alphafold-test-"))
        argv = build_argv(job_id, job_dir)
        captured.append({"job_id": job_id, "label": label, "argv": argv, "job_dir": job_dir})
        return JobInfo(job_id=job_id, status=JobStatus.PENDING)

    return captured, _fake_submit


@pytest.fixture
def client_with_stub_runner(client, captured_argv):
    captured, fake_submit = captured_argv
    client.app.state.runner.submit = fake_submit
    return client, captured


# ---- Health / manifest ----


def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "alphafold"
    assert "version" in health


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "alphafold"


def test_manifest_lists_fold_endpoint(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/fold" in paths


def test_manifest_model_is_alphafold(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "AlphaFold" in extras["model"]["name"]
    assert "PDB" in extras["model"]["output_format"]


def test_manifest_has_config_tips(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "model_preset" in extras["config_tips"]
    assert "db_preset" in extras["config_tips"]


def test_endpoint_examples_present(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert by_path["/api/fold"]["examples"]


# ---- Settings ----


def test_settings_defaults():
    from server.settings import AlphaFoldSettings

    class _Off(AlphaFoldSettings):
        model_config = SettingsConfigDict(
            env_prefix="ALPHAFOLD_TEST_", env_file=None, extra="ignore"
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/alphafold_jobs")
    assert s.root == Path("/opt/alphafold")
    assert s.python == "/opt/conda/bin/python"
    assert s.data_dir == Path("/data/models/alphafold")
    assert s.n_cpu == 8
    assert s.max_concurrent_jobs == 1


def test_settings_env_override(monkeypatch):
    from server.settings import AlphaFoldSettings

    monkeypatch.setenv("ALPHAFOLD_PYTHON", "/custom/python")
    monkeypatch.setenv("ALPHAFOLD_DATA_DIR", "/nas/data")
    s = AlphaFoldSettings()
    assert s.python == "/custom/python"
    assert s.data_dir == Path("/nas/data")


# ---- Model validation ----


def test_fold_request_defaults():
    from server.models import FoldRequest

    req = FoldRequest()
    assert req.model_preset == "monomer_ptm"
    assert req.db_preset == "reduced_dbs"
    assert req.max_template_date == "2022-01-01"
    assert req.num_multimer_predictions_per_model == 1
    assert req.models_to_relax == "best"
    assert req.use_precomputed_msas is False
    assert req.random_seed is None
    assert req.use_gpu_relax is True


def test_fold_request_valid_multimer():
    from server.models import FoldRequest

    req = FoldRequest(
        model_preset="multimer",
        db_preset="full_dbs",
        num_multimer_predictions_per_model=5,
    )
    assert req.model_preset == "multimer"
    assert req.db_preset == "full_dbs"
    assert req.num_multimer_predictions_per_model == 5


def test_fold_request_invalid_preset():
    from server.models import FoldRequest

    with pytest.raises(ValueError):
        FoldRequest(model_preset="invalid")


def test_fold_request_invalid_db_preset():
    from server.models import FoldRequest

    with pytest.raises(ValueError):
        FoldRequest(db_preset="invalid")


# ---- argv assembly ----


def test_fold_argv_monomer_reduced(tmp_path):
    from server.models import FoldRequest
    from server.settings import AlphaFoldSettings
    from server.tools import fold_argv

    settings = AlphaFoldSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "alphafold",
        python="/bin/true",
        data_dir=tmp_path / "data",
    )
    req = FoldRequest(model_preset="monomer_ptm", db_preset="reduced_dbs")
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">A\nMKT\n")

    argv = fold_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)
    assert argv[0] == "/bin/true"
    assert "run_alphafold.py" in argv[1]
    assert "--fasta_paths" in argv
    assert "--model_preset" in argv
    assert argv[argv.index("--model_preset") + 1] == "monomer_ptm"
    assert "--small_bfd_database_path" in argv
    assert "--bfd_database_path" not in argv
    assert "--pdb70_database_path" in argv
    assert "--pdb_seqres_database_path" not in argv


def test_fold_argv_multimer_full(tmp_path):
    from server.models import FoldRequest
    from server.settings import AlphaFoldSettings
    from server.tools import fold_argv

    settings = AlphaFoldSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "alphafold",
        python="/bin/true",
        data_dir=tmp_path / "data",
    )
    req = FoldRequest(
        model_preset="multimer",
        db_preset="full_dbs",
        num_multimer_predictions_per_model=3,
    )
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">A\nMKT\n>B\nAVL\n")

    argv = fold_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)
    assert "--bfd_database_path" in argv
    assert "--uniref30_database_path" in argv
    assert "--small_bfd_database_path" not in argv
    assert "--pdb_seqres_database_path" in argv
    assert "--uniprot_database_path" in argv
    assert "--pdb70_database_path" not in argv
    assert "--num_multimer_predictions_per_model" in argv
    assert argv[argv.index("--num_multimer_predictions_per_model") + 1] == "3"


def test_fold_argv_with_seed(tmp_path):
    from server.models import FoldRequest
    from server.settings import AlphaFoldSettings
    from server.tools import fold_argv

    settings = AlphaFoldSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "alphafold",
        python="/bin/true",
        data_dir=tmp_path / "data",
    )
    req = FoldRequest(random_seed=42)
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">A\nMKT\n")

    argv = fold_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)
    assert "--random_seed" in argv
    assert argv[argv.index("--random_seed") + 1] == "42"


def test_fold_argv_omits_none_seed(tmp_path):
    from server.models import FoldRequest
    from server.settings import AlphaFoldSettings
    from server.tools import fold_argv

    settings = AlphaFoldSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "alphafold",
        python="/bin/true",
        data_dir=tmp_path / "data",
    )
    req = FoldRequest()
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">A\nMKT\n")

    argv = fold_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)
    assert "--random_seed" not in argv


def test_fold_argv_precomputed_msas(tmp_path):
    from server.models import FoldRequest
    from server.settings import AlphaFoldSettings
    from server.tools import fold_argv

    settings = AlphaFoldSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "alphafold",
        python="/bin/true",
        data_dir=tmp_path / "data",
    )
    req = FoldRequest(use_precomputed_msas=True)
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">A\nMKT\n")

    argv = fold_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)
    assert "--use_precomputed_msas=true" in argv


def test_fold_argv_gpu_relax(tmp_path):
    from server.models import FoldRequest
    from server.settings import AlphaFoldSettings
    from server.tools import fold_argv

    settings = AlphaFoldSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "alphafold",
        python="/bin/true",
        data_dir=tmp_path / "data",
    )
    req = FoldRequest(use_gpu_relax=False)
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">A\nMKT\n")

    argv = fold_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)
    assert "--use_gpu_relax=false" in argv


# ---- Adapter ----


def test_adapter_name():
    from server.adapter import AlphaFoldAdapter
    from server.settings import AlphaFoldSettings

    class _Off(AlphaFoldSettings):
        model_config = SettingsConfigDict(
            env_prefix="ALPHAFOLD_TEST_", env_file=None, extra="ignore"
        )

    adapter = AlphaFoldAdapter(settings=_Off())
    assert adapter.name == "alphafold"


def test_adapter_detect_outputs(tmp_path):
    from server.adapter import AlphaFoldAdapter
    from server.settings import AlphaFoldSettings

    settings = AlphaFoldSettings(jobs_base_dir=tmp_path / "jobs")
    adapter = AlphaFoldAdapter(settings=settings)

    job_dir = tmp_path / "jobs" / "j1"
    assert not adapter.detect_outputs(job_dir)

    out = job_dir / "output" / "input"
    out.mkdir(parents=True)
    assert not adapter.detect_outputs(job_dir)

    (out / "ranked_0.pdb").write_text("ATOM ...")
    assert adapter.detect_outputs(job_dir)


def test_adapter_subprocess_env():
    from server.adapter import AlphaFoldAdapter
    from server.settings import AlphaFoldSettings

    class _Off(AlphaFoldSettings):
        model_config = SettingsConfigDict(
            env_prefix="ALPHAFOLD_TEST_", env_file=None, extra="ignore"
        )

    adapter = AlphaFoldAdapter(settings=_Off())
    env = adapter.subprocess_env()
    assert env["TF_FORCE_UNIFIED_MEMORY"] == "1"
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "4.0"
    assert env["OPENMM_CPU_THREADS"] == "8"


# ---- Endpoint smoke (uses stubbed runner) ----


def test_fold_endpoint_smoke(client_with_stub_runner, tmp_path):
    client, captured = client_with_stub_runner
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">A\nMKTAYIAKQRQISFVKSHFSRQLE\n")

    with open(fasta, "rb") as f:
        r = client.post("/api/fold", data={}, files={"input_fasta": ("test.fasta", f, "text/plain")})
    assert r.status_code == 200, r.text
    assert r.json()["status"] in ("pending", "running")
    assert len(captured) == 1
    assert captured[0]["label"] == "fold"
    assert "--fasta_paths" in captured[0]["argv"]


def test_fold_endpoint_with_params(client_with_stub_runner, tmp_path):
    client, captured = client_with_stub_runner
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">A\nMKT\n")

    with open(fasta, "rb") as f:
        r = client.post(
            "/api/fold",
            data={
                "model_preset": "multimer",
                "db_preset": "full_dbs",
                "num_multimer_predictions_per_model": "3",
            },
            files={"input_fasta": ("test.fasta", f, "text/plain")},
        )
    assert r.status_code == 200, r.text

    argv = captured[0]["argv"]
    assert argv[argv.index("--model_preset") + 1] == "multimer"
    assert argv[argv.index("--db_preset") + 1] == "full_dbs"


def test_fold_endpoint_saves_fasta(client_with_stub_runner, tmp_path):
    client, captured = client_with_stub_runner
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">A\nMKT\n")

    with open(fasta, "rb") as f:
        r = client.post("/api/fold", data={}, files={"input_fasta": ("test.fasta", f, "text/plain")})
    assert r.status_code == 200, r.text

    saved = captured[0]["job_dir"] / "input" / "input.fasta"
    assert saved.exists()
    assert ">A" in saved.read_text()


def test_fold_endpoint_rejects_no_fasta(client):
    r = client.post("/api/fold", data={})
    assert r.status_code == 422


# ----- task endpoint smoke (synchronous; /bin/true so it returns immediately) -----


def test_fold_task_endpoint_returns_terminal_status(client):
    """POST /api/tasks/fold blocks until /bin/true exits, returns terminal JobInfo."""
    # Use a minimal valid fasta for the request.
    fasta_bytes = b">test\nMKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF\n"
    resp = client.post(
        "/api/tasks/fold",
        data={},
        files={"input_fasta": ("test.fasta", fasta_bytes, "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_fold_task_endpoint_honors_job_id_header(client):
    fasta_bytes = b">x\nMKQ\n"
    resp = client.post(
        "/api/tasks/fold",
        data={},
        files={"input_fasta": ("x.fasta", fasta_bytes, "text/plain")},
        headers={"X-Bioagent-Job-Id": "af-task-001"},
    )
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "af-task-001"


def test_fold_task_endpoint_duplicate_returns_existing(client):
    hdrs = {"X-Bioagent-Job-Id": "af-dup-001"}
    fasta_a = b">a\nMKQ\n"
    fasta_b = b">b\nLLL\n"
    r1 = client.post(
        "/api/tasks/fold",
        data={},
        files={"input_fasta": ("a.fasta", fasta_a, "text/plain")},
        headers=hdrs,
    )
    r2 = client.post(
        "/api/tasks/fold",
        data={},
        files={"input_fasta": ("b.fasta", fasta_b, "text/plain")},
        headers=hdrs,
    )
    assert r1.json()["job_id"] == r2.json()["job_id"]
    assert r1.json()["created_at"] == r2.json()["created_at"]

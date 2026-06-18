"""Offline tests for dockq-server (no real DockQ binary needed)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Import `server.app` fresh against patched env vars.

    The endpoint signatures use `Annotated[Model, Form()]` which is resolved
    against module-level globals at import time, so we re-import after
    `monkeypatch.setenv` to pick up the test paths.
    """
    monkeypatch.setenv("DOCKQ_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("DOCKQ_ROOT", str(tmp_path / "dockq"))
    monkeypatch.setenv("DOCKQ_BINARY", "/bin/true")  # never actually executed in offline tests
    (tmp_path / "dockq").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Healthcheck / manifest -----

def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "dockq"
    assert "version" in health
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "dockq"
    assert detail["version"] == health["version"]


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "dockq"


def test_manifest_lists_both_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/score" in paths
    assert "/api/score_batch" in paths


def test_manifest_extras_has_tool_outputs(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "score" in extras["tool_outputs"]
    assert "score_batch" in extras["tool_outputs"]


def test_manifest_extras_has_scoring_legend(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    legend = extras["scoring_legend"]
    for k in ("DockQ", "iRMSD", "LRMSD", "fnat", "clashes"):
        assert k in legend


def test_endpoint_examples_present(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert by_path["/api/score"]["examples"]
    assert by_path["/api/score_batch"]["examples"]


# ----- Settings -----

def test_settings_defaults():
    from server.settings import DockQSettings

    class _Off(DockQSettings):
        model_config = SettingsConfigDict(env_prefix="DOCKQ_TEST_", env_file=None, extra="ignore")

    s = _Off()
    assert s.jobs_base_dir == Path("/data/dockq_jobs")
    assert s.root == Path("/opt/dockq")
    assert s.binary == "DockQ"
    assert s.default_n_cpu == 8
    assert s.max_batch_size == 200
    assert s.oss_region == "cn-hangzhou"


def test_settings_env_override(monkeypatch):
    from server.settings import DockQSettings
    monkeypatch.setenv("DOCKQ_BINARY", "/custom/DockQ")
    monkeypatch.setenv("DOCKQ_DEFAULT_N_CPU", "16")
    s = DockQSettings()
    assert s.binary == "/custom/DockQ"
    assert s.default_n_cpu == 16


# ----- Adapter -----

def test_adapter_name():
    from server.adapter import DockQAdapter
    from server.settings import DockQSettings

    class _Off(DockQSettings):
        model_config = SettingsConfigDict(env_prefix="DOCKQ_TEST_", env_file=None, extra="ignore")

    a = DockQAdapter(settings=_Off())
    assert a.name == "dockq"


def test_detect_outputs_single(tmp_path):
    from server.adapter import DockQAdapter
    from server.settings import DockQSettings

    class _Off(DockQSettings):
        model_config = SettingsConfigDict(env_prefix="DOCKQ_TEST_", env_file=None, extra="ignore")

    a = DockQAdapter(settings=_Off())
    job = tmp_path / "j"
    (job / "output").mkdir(parents=True)
    (job / "output" / "result.json").write_text('{"total_DockQ": 0.5}')
    assert a.detect_outputs(job) is True


def test_detect_outputs_batch(tmp_path):
    from server.adapter import DockQAdapter
    from server.settings import DockQSettings

    class _Off(DockQSettings):
        model_config = SettingsConfigDict(env_prefix="DOCKQ_TEST_", env_file=None, extra="ignore")

    a = DockQAdapter(settings=_Off())
    job = tmp_path / "j"
    (job / "output").mkdir(parents=True)
    (job / "output" / "scores.csv").write_text("model,DockQ\n")
    assert a.detect_outputs(job) is True


def test_detect_outputs_empty(tmp_path):
    from server.adapter import DockQAdapter
    from server.settings import DockQSettings

    class _Off(DockQSettings):
        model_config = SettingsConfigDict(env_prefix="DOCKQ_TEST_", env_file=None, extra="ignore")

    a = DockQAdapter(settings=_Off())
    job = tmp_path / "j"
    (job / "output").mkdir(parents=True)
    assert a.detect_outputs(job) is False


# ----- Request models -----

def test_score_request_defaults():
    from server.models import ScoreRequest
    r = ScoreRequest()
    assert r.name == "run"
    assert r.mapping is None
    assert r.small_molecule is False
    assert r.capri_peptide is False
    assert r.no_align is False
    assert r.allowed_mismatches == 0
    assert r.optDockQF1 is False
    assert r.n_cpu is None


def test_score_batch_request_defaults():
    from server.models import ScoreBatchRequest
    r = ScoreBatchRequest()
    assert r.sort_by == "DockQ"


def test_score_request_name_validation():
    from pydantic import ValidationError
    from server.models import ScoreRequest
    with pytest.raises(ValidationError):
        ScoreRequest(name="bad/name")


# ----- argv builders -----

def test_score_argv_basic(tmp_path):
    from server.models import ScoreRequest
    from server.settings import DockQSettings
    from server.tools import score_argv

    class _Off(DockQSettings):
        model_config = SettingsConfigDict(env_prefix="DOCKQ_TEST_", env_file=None, extra="ignore")

    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    argv = score_argv(
        ScoreRequest(),
        job_dir=job_dir,
        model_path=tmp_path / "m.pdb",
        native_path=tmp_path / "n.pdb",
        settings=s,
    )
    assert argv[0] == "DockQ"
    assert "--json" in argv
    assert argv[argv.index("--json") + 1].endswith("run.json")
    assert "--short" in argv
    assert "--n_cpu" in argv
    assert argv[argv.index("--n_cpu") + 1] == "8"


def test_score_argv_with_flags(tmp_path):
    from server.models import ScoreRequest
    from server.settings import DockQSettings
    from server.tools import score_argv

    class _Off(DockQSettings):
        model_config = SettingsConfigDict(env_prefix="DOCKQ_TEST_", env_file=None, extra="ignore")

    job_dir = tmp_path / "j"
    job_dir.mkdir()
    argv = score_argv(
        ScoreRequest(
            mapping="HLA:BCX",
            small_molecule=True,
            no_align=True,
            allowed_mismatches=2,
            optDockQF1=True,
            n_cpu=8,
        ),
        job_dir=job_dir,
        model_path=tmp_path / "m.pdb",
        native_path=tmp_path / "n.pdb",
        settings=_Off(),
    )
    assert "--mapping" in argv
    assert argv[argv.index("--mapping") + 1] == "HLA:BCX"
    assert "--small_molecule" in argv
    assert "--no_align" in argv
    assert "--optDockQF1" in argv
    assert "--allowed_mismatches" in argv
    assert argv[argv.index("--allowed_mismatches") + 1] == "2"
    assert argv[argv.index("--n_cpu") + 1] == "8"


def test_batch_argv_basic(tmp_path):
    from server.models import ScoreBatchRequest
    from server.settings import DockQSettings
    from server.tools import batch_argv

    class _Off(DockQSettings):
        model_config = SettingsConfigDict(env_prefix="DOCKQ_TEST_", env_file=None, extra="ignore")

    job_dir = tmp_path / "j"
    job_dir.mkdir()
    argv = batch_argv(
        ScoreBatchRequest(sort_by="iRMSD"),
        job_dir=job_dir,
        native_path=tmp_path / "n.pdb",
        models_dir=tmp_path / "models",
        settings=_Off(),
    )
    assert argv[0] == sys.executable
    assert argv[1].endswith("batch_score.py")
    assert "--native" in argv
    assert "--models-dir" in argv
    assert "--output-dir" in argv
    assert "--sort-by" in argv
    assert argv[argv.index("--sort-by") + 1] == "iRMSD"
    assert "--n_cpu" in argv  # forwarded to DockQ via the driver


# ----- batch driver smoke -----

def test_batch_driver_summarize_single_interface():
    from server import batch_score
    row = batch_score._summarize(
        "model_001",
        {"total_DockQ": 0.75, "best_result": {("A", "B"): {"DockQ": 0.75, "iRMSD": 1.2, "fnat": 0.8}}},
    )
    assert row["model"] == "model_001"
    assert row["DockQ"] == 0.75
    assert row["iRMSD"] == 1.2
    assert row["fnat"] == 0.8
    assert row["n_interfaces"] == 1


def test_batch_driver_summarize_multi_interface_averages():
    from server import batch_score
    row = batch_score._summarize(
        "m",
        {
            "total_DockQ": 0.6,
            "best_result": {
                ("A", "B"): {"DockQ": 0.8, "iRMSD": 1.0},
                ("A", "C"): {"DockQ": 0.4, "iRMSD": 2.0},
            },
        },
    )
    # Headline DockQ is the total field; per-interface metrics are averaged.
    assert row["DockQ"] == 0.6
    assert row["iRMSD"] == pytest.approx(1.5)
    assert row["n_interfaces"] == 2


def test_batch_driver_summarize_dockq2_global_key():
    """DockQ 2.x renamed `total_DockQ` to `GlobalDockQ` (mean across interfaces).
    Verify the summary picks it up; otherwise scores.csv DockQ column shows NaN."""
    from server import batch_score
    row = batch_score._summarize(
        "m",
        {
            "best_dockq": 1.95,           # sum (DockQ 2.x)
            "GlobalDockQ": 0.65,          # mean (DockQ 2.x — the headline)
            "best_result": {
                "AB": {"DockQ": 0.99, "iRMSD": 0.0, "F1": 0.98},
                "AC": {"DockQ": 0.51, "iRMSD": 1.2, "F1": 0.50},
                "BC": {"DockQ": 0.45, "iRMSD": 2.1, "F1": 0.64},
            },
        },
    )
    assert row["DockQ"] == 0.65
    assert row["iRMSD"] == pytest.approx(1.1, abs=0.01)
    assert row["F1"] == pytest.approx(0.7066, abs=0.001)
    assert row["n_interfaces"] == 3


# ----- URI resolution -----

def test_uri_resolve_file(tmp_path):
    from server.settings import DockQSettings
    from server.uris import resolve_input

    class _Off(DockQSettings):
        model_config = SettingsConfigDict(env_prefix="DOCKQ_TEST_", env_file=None, extra="ignore")

    src = tmp_path / "src.pdb"
    src.write_text("ATOM\n")
    dest = tmp_path / "dest.pdb"
    out = resolve_input(None, f"file://{src}", dest, _Off())
    assert out.read_text() == "ATOM\n"


def test_uri_requires_input(tmp_path):
    from fastapi import HTTPException
    from server.settings import DockQSettings
    from server.uris import resolve_input

    class _Off(DockQSettings):
        model_config = SettingsConfigDict(env_prefix="DOCKQ_TEST_", env_file=None, extra="ignore")

    with pytest.raises(HTTPException) as exc:
        resolve_input(None, None, tmp_path / "x", _Off())
    assert exc.value.status_code == 422


# ----- endpoint smoke (no real DockQ) -----

def test_score_endpoint_returns_job(client):
    data_dir = Path(__file__).resolve().parent / "data"
    with open(data_dir / "model.pdb", "rb") as fm, open(data_dir / "native.pdb", "rb") as fn:
        resp = client.post(
            "/api/score",
            data={"name": "demo"},
            files={
                "model": ("model.pdb", fm, "chemical/x-pdb"),
                "native": ("native.pdb", fn, "chemical/x-pdb"),
            },
        )
    # Job is accepted; the subprocess (DOCKQ_BIN=/bin/true) exits 0 but won't
    # produce JSON, so the job may end up FAILED with no_outputs — that's
    # outside the scope of this smoke test.
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["created_at"] is not None
    assert body["input_params"] is not None
    assert body["input_params"]["name"] == "demo"


def test_score_batch_endpoint_requires_models(client):
    data_dir = Path(__file__).resolve().parent / "data"
    with open(data_dir / "native.pdb", "rb") as fn:
        resp = client.post(
            "/api/score_batch",
            files={"native": ("native.pdb", fn, "chemical/x-pdb")},
        )
    assert resp.status_code == 422
    assert "models" in resp.json()["detail"].lower()


def test_score_batch_endpoint_returns_job(client):
    data_dir = Path(__file__).resolve().parent / "data"
    with open(data_dir / "native.pdb", "rb") as fn, \
         open(data_dir / "model.pdb", "rb") as fm1, \
         open(data_dir / "model_alt.pdb", "rb") as fm2:
        resp = client.post(
            "/api/score_batch",
            data={"sort_by": "DockQ"},
            files=[
                ("native", ("native.pdb", fn, "chemical/x-pdb")),
                ("models", ("m1.pdb", fm1, "chemical/x-pdb")),
                ("models", ("m2.pdb", fm2, "chemical/x-pdb")),
            ],
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["created_at"] is not None
    assert body["input_params"] is not None
    assert body["input_params"]["sort_by"] == "DockQ"
    assert body["input_params"]["num_models"] == 2


# ----- task endpoint smoke -----

def test_score_task_endpoint_returns_terminal_status(client):
    """POST /api/tasks/score blocks until subprocess exits."""
    pdb_bytes = b"REMARK fake pdb\nATOM\nEND\n"
    resp = client.post(
        "/api/tasks/score",
        data={},
        files={
            "model": ("model.pdb", pdb_bytes, "chemical/x-pdb"),
            "native": ("native.pdb", pdb_bytes, "chemical/x-pdb"),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_score_batch_task_endpoint_returns_terminal_status(client):
    """POST /api/tasks/score_batch blocks until subprocess exits."""
    pdb_bytes = b"REMARK\nEND\n"
    resp = client.post(
        "/api/tasks/score_batch",
        data={},
        files=[
            ("native", ("native.pdb", pdb_bytes, "chemical/x-pdb")),
            ("models", ("model_0.pdb", pdb_bytes, "chemical/x-pdb")),
            ("models", ("model_1.pdb", pdb_bytes, "chemical/x-pdb")),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_score_batch_task_endpoint_rejects_empty_models(client):
    """422 returned up-front before allocating a job."""
    pdb_bytes = b"REMARK\nEND\n"
    resp = client.post(
        "/api/tasks/score_batch",
        data={},
        files={"native": ("native.pdb", pdb_bytes, "chemical/x-pdb")},
    )
    assert resp.status_code == 422

"""Offline tests for boltzgen-server (no real BoltzGen pipeline needed)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BOLTZGEN_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("BOLTZGEN_ROOT", str(tmp_path / "boltzgen"))
    monkeypatch.setenv("BOLTZGEN_PYTHON", "/bin/true")
    monkeypatch.setenv("BOLTZGEN_CLI", "/bin/true")
    monkeypatch.setenv("BOLTZGEN_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("BOLTZGEN_MOLDIR", str(tmp_path / "moldir"))
    (tmp_path / "boltzgen").mkdir(parents=True, exist_ok=True)
    (tmp_path / "weights").mkdir(parents=True, exist_ok=True)
    (tmp_path / "moldir").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


DATA_DIR = Path(__file__).resolve().parent / "data"


# ----- Healthcheck / manifest -----

def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "boltzgen"
    assert "version" in health
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "boltzgen"
    assert detail["version"] == health["version"]


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "boltzgen"


def test_manifest_lists_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert paths == {
        "/api/design",
        "/api/inverse_fold",
        "/api/tasks/design",
        "/api/tasks/inverse_fold",
    }


def test_manifest_protocols(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    protocols = extras["protocols"]
    assert "protein-anything" in protocols
    assert "nanobody-anything" in protocols
    assert "protein-small_molecule" in protocols
    assert len(protocols) == 6


def test_manifest_models(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    models = extras["models"]
    assert "design-diverse" in models
    assert "inverse-fold" in models
    assert "folding" in models
    assert "affinity" in models
    assert len(models) == 5


def test_manifest_has_endpoint_examples(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert by_path["/api/design"]["examples"]
    assert by_path["/api/inverse_fold"]["examples"]


# ----- Settings -----

def test_settings_defaults():
    from server.settings import BoltzGenSettings

    class _Off(BoltzGenSettings):
        model_config = SettingsConfigDict(env_prefix="BOLTZGEN_TEST_", env_file=None, extra="ignore")

    s = _Off()
    assert s.jobs_base_dir == Path("/data/boltzgen_jobs")
    assert s.root == Path("/opt/boltzgen")
    assert s.python == "/opt/conda/envs/boltzgen/bin/python"
    assert s.cli == "/opt/conda/envs/boltzgen/bin/boltzgen"
    # Weights are externalized to NAS — defaults point at the FC mount path.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    assert s.weights_dir == Path("/data/models/boltzgen/weights")
    assert s.moldir == Path("/data/models/boltzgen/moldir")
    assert s.oss_region == "cn-hangzhou"


def test_settings_env_override(monkeypatch):
    from server.settings import BoltzGenSettings
    monkeypatch.setenv("BOLTZGEN_PYTHON", "/custom/python")
    monkeypatch.setenv("BOLTZGEN_CLI", "/custom/boltzgen")
    s = BoltzGenSettings()
    assert s.python == "/custom/python"
    assert s.cli == "/custom/boltzgen"


# ----- Adapter -----

def test_adapter_name():
    from server.adapter import BoltzGenAdapter
    from server.settings import BoltzGenSettings

    class _Off(BoltzGenSettings):
        model_config = SettingsConfigDict(env_prefix="BOLTZGEN_TEST_", env_file=None, extra="ignore")

    a = BoltzGenAdapter(settings=_Off())
    assert a.name == "boltzgen"


def test_detect_outputs_final_ranked(tmp_path):
    from server.adapter import BoltzGenAdapter
    from server.settings import BoltzGenSettings

    class _Off(BoltzGenSettings):
        model_config = SettingsConfigDict(env_prefix="BOLTZGEN_TEST_", env_file=None, extra="ignore")

    a = BoltzGenAdapter(settings=_Off())
    job = tmp_path / "j"
    final = job / "output" / "final_ranked_designs" / "final_30_designs"
    final.mkdir(parents=True)
    (final / "design_0.cif").write_text("data_test\n")
    assert a.detect_outputs(job) is True


def test_detect_outputs_intermediate(tmp_path):
    from server.adapter import BoltzGenAdapter
    from server.settings import BoltzGenSettings

    class _Off(BoltzGenSettings):
        model_config = SettingsConfigDict(env_prefix="BOLTZGEN_TEST_", env_file=None, extra="ignore")

    a = BoltzGenAdapter(settings=_Off())
    job = tmp_path / "j"
    intermediate = job / "output" / "intermediate_designs"
    intermediate.mkdir(parents=True)
    (intermediate / "design_0_0.cif").write_text("data_test\n")
    assert a.detect_outputs(job) is True


def test_detect_outputs_empty(tmp_path):
    from server.adapter import BoltzGenAdapter
    from server.settings import BoltzGenSettings

    class _Off(BoltzGenSettings):
        model_config = SettingsConfigDict(env_prefix="BOLTZGEN_TEST_", env_file=None, extra="ignore")

    a = BoltzGenAdapter(settings=_Off())
    job = tmp_path / "j"
    (job / "output").mkdir(parents=True)
    assert a.detect_outputs(job) is False


# ----- Request models -----

def test_design_request_defaults():
    from server.models import DesignRequest
    r = DesignRequest()
    assert r.protocol == "protein-anything"
    assert r.num_designs == 100
    assert r.budget == 30
    assert r.use_kernels == "auto"
    assert r.skip_inverse_folding is False
    assert r.inverse_fold_num_sequences == 1
    assert r.reuse is False


def test_inverse_fold_request_defaults():
    from server.models import InverseFoldRequest
    r = InverseFoldRequest()
    assert r.protocol == "protein-anything"
    assert r.num_designs == 100


def test_design_request_custom():
    from server.models import DesignRequest
    r = DesignRequest(
        protocol="nanobody-anything",
        num_designs=50,
        budget=10,
        use_kernels="false",
        skip_inverse_folding=True,
    )
    assert r.protocol == "nanobody-anything"
    assert r.num_designs == 50
    assert r.budget == 10
    assert r.use_kernels == "false"
    assert r.skip_inverse_folding is True


# ----- argv builders -----

def test_design_argv_structure(tmp_path):
    from server.models import DesignRequest
    from server.settings import BoltzGenSettings
    from server.tools import design_argv

    class _Off(BoltzGenSettings):
        model_config = SettingsConfigDict(env_prefix="BOLTZGEN_TEST_", env_file=None, extra="ignore")

    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    yaml_path = tmp_path / "spec.yaml"
    yaml_path.write_text("entities: []\n")

    argv = design_argv(DesignRequest(), job_dir=job_dir, yaml_path=yaml_path, settings=s)

    assert argv[0] == s.cli
    assert "run" in argv
    assert str(yaml_path) in argv
    assert "--protocol" in argv
    assert "protein-anything" in argv
    assert "--num_designs" in argv
    assert "100" in argv
    assert "--no_subprocess" in argv
    assert "--devices" in argv
    assert "1" in argv
    assert "--design_checkpoints" in argv
    assert "--inverse_fold_checkpoint" in argv
    assert "--folding_checkpoint" in argv
    assert "--affinity_checkpoint" in argv


def test_design_argv_with_options(tmp_path):
    from server.models import DesignRequest
    from server.settings import BoltzGenSettings
    from server.tools import design_argv

    class _Off(BoltzGenSettings):
        model_config = SettingsConfigDict(env_prefix="BOLTZGEN_TEST_", env_file=None, extra="ignore")

    req = DesignRequest(
        step_scale="1.8",
        skip_inverse_folding=True,
        inverse_fold_avoid="C",
        alpha=0.01,
        reuse=True,
    )
    argv = design_argv(req, job_dir=tmp_path, yaml_path=tmp_path / "s.yaml", settings=_Off())

    assert "--step_scale" in argv
    assert "1.8" in argv
    assert "--skip_inverse_folding" in argv
    assert "--inverse_fold_avoid" in argv
    assert "C" in argv
    assert "--alpha" in argv
    assert "0.01" in argv
    assert "--reuse" in argv


def test_inverse_fold_argv_structure(tmp_path):
    from server.models import InverseFoldRequest
    from server.settings import BoltzGenSettings
    from server.tools import inverse_fold_argv

    class _Off(BoltzGenSettings):
        model_config = SettingsConfigDict(env_prefix="BOLTZGEN_TEST_", env_file=None, extra="ignore")

    s = _Off()
    argv = inverse_fold_argv(
        InverseFoldRequest(), job_dir=tmp_path, yaml_path=tmp_path / "s.yaml", settings=s,
    )

    assert argv[0] == s.cli
    assert "run" in argv
    assert "--only_inverse_fold" in argv
    assert "--no_subprocess" in argv
    assert "--design_checkpoints" not in argv


# ----- URI resolution -----

def test_uri_resolve_file(tmp_path):
    from server.settings import BoltzGenSettings
    from server.uris import resolve_input

    class _Off(BoltzGenSettings):
        model_config = SettingsConfigDict(env_prefix="BOLTZGEN_TEST_", env_file=None, extra="ignore")

    src = tmp_path / "src.yaml"
    src.write_text("entities: []\n")
    dest = tmp_path / "dest.yaml"
    out = resolve_input(None, f"file://{src}", dest, _Off())
    assert out.read_text() == "entities: []\n"


def test_uri_requires_input(tmp_path):
    from fastapi import HTTPException
    from server.settings import BoltzGenSettings
    from server.uris import resolve_input

    class _Off(BoltzGenSettings):
        model_config = SettingsConfigDict(env_prefix="BOLTZGEN_TEST_", env_file=None, extra="ignore")

    with pytest.raises(HTTPException) as exc:
        resolve_input(None, None, tmp_path / "x", _Off())
    assert exc.value.status_code == 422


# ----- endpoint smoke (no real pipeline) -----

def test_design_endpoint_returns_job(client):
    with open(DATA_DIR / "vanilla.yaml", "rb") as fh:
        resp = client.post(
            "/api/design",
            data={
                "protocol": "protein-anything",
                "num_designs": "10",
                "budget": "5",
            },
            files={
                "design_yaml": ("vanilla.yaml", fh, "application/x-yaml"),
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["created_at"] is not None
    assert body["input_params"] is not None
    assert body["input_params"]["protocol"] == "protein-anything"
    assert body["input_params"]["num_designs"] == 10
    assert body["input_params"]["budget"] == 5


def test_inverse_fold_endpoint_returns_job(client):
    with open(DATA_DIR / "inverse_fold.yaml", "rb") as fh:
        resp = client.post(
            "/api/inverse_fold",
            data={
                "inverse_fold_num_sequences": "5",
            },
            files={
                "design_yaml": ("inverse_fold.yaml", fh, "application/x-yaml"),
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["input_params"]["inverse_fold_num_sequences"] == 5


# ----- task endpoint smoke (synchronous; /bin/true so it returns immediately) -----

def test_design_task_endpoint_returns_terminal_status(client):
    """POST /api/tasks/design should block until /bin/true exits and return terminal JobInfo.

    With BOLTZGEN_CLI=/bin/true the "pipeline" runs instantly, exits rc=0, and
    produces no CIF/CSV → adapter.detect_outputs returns False → JobInfo
    status=failed with failure_kind=no_outputs.  Either outcome (completed or
    failed) means the synchronous endpoint waited for the subprocess.
    """
    with open(DATA_DIR / "vanilla.yaml", "rb") as fh:
        resp = client.post(
            "/api/tasks/design",
            data={
                "protocol": "protein-anything",
                "num_designs": "10",
                "budget": "5",
            },
            files={
                "design_yaml": ("vanilla.yaml", fh, "application/x-yaml"),
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    # Synchronous endpoint → JobInfo is already terminal (not PENDING/RUNNING).
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None
    assert body["input_params"]["protocol"] == "protein-anything"


def test_design_task_endpoint_honors_job_id_header(client):
    """X-Bioagent-Job-Id should become the response job_id."""
    with open(DATA_DIR / "vanilla.yaml", "rb") as fh:
        resp = client.post(
            "/api/tasks/design",
            data={"protocol": "protein-anything", "num_designs": "10", "budget": "5"},
            files={"design_yaml": ("vanilla.yaml", fh, "application/x-yaml")},
            headers={"X-Bioagent-Job-Id": "boltzgen-task-001"},
        )
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "boltzgen-task-001"


def test_design_task_endpoint_duplicate_returns_existing(client):
    """Two requests with the same task_id: second returns the first's JobInfo.

    Proves the framework-level idempotency check works through boltzgen-server's
    custom-signature handler (i.e. execute_task's idempotency wraps cleanly
    around the UploadFile parsing).
    """
    hdrs = {"X-Bioagent-Job-Id": "boltzgen-dup-001"}
    with open(DATA_DIR / "vanilla.yaml", "rb") as fh:
        r1 = client.post(
            "/api/tasks/design",
            data={"protocol": "protein-anything", "num_designs": "10", "budget": "5"},
            files={"design_yaml": ("vanilla.yaml", fh, "application/x-yaml")},
            headers=hdrs,
        )
    with open(DATA_DIR / "vanilla.yaml", "rb") as fh:
        r2 = client.post(
            "/api/tasks/design",
            data={"protocol": "nanobody-anything", "num_designs": "99", "budget": "5"},
            files={"design_yaml": ("vanilla.yaml", fh, "application/x-yaml")},
            headers=hdrs,
        )
    assert r1.json()["job_id"] == r2.json()["job_id"]
    # First request's params should win — proves no re-run.
    assert r2.json()["input_params"]["protocol"] == "protein-anything"
    assert r2.json()["input_params"]["num_designs"] == 10
    # created_at identity proves store.create wasn't called the second time.
    assert r1.json()["created_at"] == r2.json()["created_at"]

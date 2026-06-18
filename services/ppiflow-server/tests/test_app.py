"""End-to-end smoke for ppiflow-server.

We can't invoke the real `sample_*.py` scripts here (need GPU + the conda env
+ checkpoints). Instead we hook a stub endpoint into the framework to verify
the wiring; the real endpoints are exercised at deploy-time against FC.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

from server.adapter import PPIFlowAdapter
from server.settings import PPIFlowSettings


class _OfflineSettings(PPIFlowSettings):
    model_config = SettingsConfigDict(
        env_prefix="PPIFLOW_TEST_", env_file=None, extra="ignore",
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    from bioagent_service import create_app

    settings = _OfflineSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "ppiflow",
        ckpt_dir=tmp_path / "ppiflow" / "checkpoint",
        config_dir=tmp_path / "ppiflow" / "configs",
    )
    # `subprocess_cwd` returns settings.root; subprocess.Popen needs it to exist.
    settings.root.mkdir(parents=True, exist_ok=True)
    adapter = PPIFlowAdapter(settings=settings)
    app = create_app(adapter, settings, title="PPIFlow Test")

    @app.post("/api/stub")
    def _stub():
        def _build(_job_id: str, job_dir: Path) -> list[str]:
            out = job_dir / "output" / "stub"
            out.mkdir(parents=True, exist_ok=True)
            return ["bash", "-c", f"echo ATOM > {out / 'sample_0.pdb'}"]
        return app.state.runner.submit(build_argv=_build, label="stub")

    return TestClient(app)


def test_health(client: TestClient) -> None:
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "ppiflow"
    assert "version" in health
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "ppiflow"
    assert detail["version"] == health["version"]


def test_manifest_lists_all_five_endpoints_in_extras(client: TestClient) -> None:
    body = client.get("/api/manifest").json()
    assert body["service"] == "ppiflow"
    extras = body["service_specific"]
    summary = extras["endpoints_summary"]
    for path in (
        "/api/sample/binder",
        "/api/sample/antibody",
        "/api/sample/nanobody",
        "/api/sample/monomer",
        "/api/sample/scaffolding",
    ):
        assert path in summary, f"{path} missing from endpoints_summary"
    # config_tips covers the three most common agent confusions.
    assert "cdr_length" in extras["config_tips"]
    assert "length_subset" in extras["config_tips"]
    assert "specified_hotspots" in extras["config_tips"]


def test_manifest_has_one_example_per_endpoint() -> None:
    """The real `server.app.app` registers all five endpoints; verify each has an
    example. Done against the real app rather than the test fixture so the
    adapter's endpoint_examples() dict keys align with the actual routes."""
    os.environ["PPIFLOW_JOBS_BASE_DIR"] = "/tmp/ppiflow_jobs_test"
    import importlib
    import sys
    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    body = TestClient(server_app.app).get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    for path in (
        "/api/sample/binder",
        "/api/sample/antibody",
        "/api/sample/nanobody",
        "/api/sample/monomer",
        "/api/sample/scaffolding",
    ):
        assert path in by_path, f"{path} not registered on real app"
        assert by_path[path]["examples"], f"{path} has no examples"


def test_stub_pipeline_round_trip(client: TestClient) -> None:
    import time
    job_id = client.post("/api/stub").json()["job_id"]
    for _ in range(50):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert body["status"] == "completed"

    files = client.get(f"/api/jobs/{job_id}/files").json()
    assert any(f.endswith("sample_0.pdb") for f in files["files"])


def test_openapi_registers_all_five_request_models() -> None:
    os.environ["PPIFLOW_JOBS_BASE_DIR"] = "/tmp/ppiflow_jobs_test"
    import importlib
    import sys
    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    schema = TestClient(server_app.app).get("/openapi.json").json()
    models = set(schema["components"]["schemas"].keys())
    for m in (
        "BinderRequest",
        "AntibodyRequest",
        "NanobodyRequest",
        "MonomerRequest",
        "ScaffoldingRequest",
        "JobInfo",
    ):
        assert m in models, f"{m} not in OpenAPI components"


# =====================================================================
# Task endpoint smoke tests (synchronous FC Async Task Mode-friendly)
# =====================================================================


@pytest.fixture
def real_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Load the real `server.app` with tmp-path env overrides so subprocess
    can be launched (subprocess_cwd = settings.root must exist)."""
    monkeypatch.setenv("PPIFLOW_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("PPIFLOW_ROOT", str(tmp_path / "ppiflow"))
    monkeypatch.setenv("PPIFLOW_CKPT_DIR", str(tmp_path / "ppiflow" / "checkpoint"))
    monkeypatch.setenv("PPIFLOW_CONFIG_DIR", str(tmp_path / "ppiflow" / "configs"))
    (tmp_path / "ppiflow").mkdir(parents=True, exist_ok=True)

    import importlib
    import sys
    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


def test_binder_task_endpoint_returns_terminal_status(real_client: TestClient) -> None:
    """POST /api/tasks/sample/binder blocks until subprocess exits."""
    pdb_bytes = b"REMARK fake pdb\nATOM\nEND\n"
    resp = real_client.post(
        "/api/tasks/sample/binder",
        data={"target_chain": "B"},
        files={"target": ("target.pdb", pdb_bytes, "chemical/x-pdb")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_monomer_task_endpoint_returns_terminal_status(real_client: TestClient) -> None:
    """POST /api/tasks/sample/monomer blocks; no uploads needed."""
    resp = real_client.post(
        "/api/tasks/sample/monomer",
        data={},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_monomer_task_endpoint_honors_job_id_header(real_client: TestClient) -> None:
    resp = real_client.post(
        "/api/tasks/sample/monomer",
        data={},
        headers={"X-Bioagent-Job-Id": "ppiflow-task-001"},
    )
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "ppiflow-task-001"

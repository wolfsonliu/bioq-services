"""End-to-end: an EchoAdapter cycled through the full HTTP surface.

Validates the contract that a new service can be built with ~25 lines of
service-side code and inherit submission, polling, log, download, and the
OpenAPI schema for free.
"""

from __future__ import annotations

import time
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from bioagent_service import JobAdapter, ServiceSettings, create_app
from bioagent_service.models import JobStatus


class _EchoRequest(BaseModel):
    message: str
    fail: bool = False
    produce_output: bool = True


class _EchoSettings(ServiceSettings):
    model_config = SettingsConfigDict(env_file=None, env_prefix="ECHO_TEST_", extra="ignore")


class _EchoAdapter(JobAdapter):
    name = "echo"


def _echo_argv(req: _EchoRequest, job_dir: Path) -> list[str]:
    """Endpoint-side argv builder. Lives outside the adapter on purpose: in the
    new framework shape, request → argv is a per-endpoint concern."""
    out = job_dir / "output"
    out.mkdir(parents=True, exist_ok=True)
    if req.fail:
        # exits 1 and writes a fake traceback so error_summary extraction can fire.
        script = (
            "echo 'Traceback (most recent call last):' >&2; "
            "echo \"ValueError: boom: $1\" >&2; "
            "exit 1"
        )
    elif not req.produce_output:
        # exits 0 but writes nothing — should be flagged as NO_OUTPUTS.
        script = "echo 'pretend success but no output' >&1"
    else:
        script = f"echo $1 > {out / 'msg.txt'}"
    return ["bash", "-c", script, "_", req.message]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = _EchoSettings(jobs_base_dir=tmp_path / "jobs", max_concurrent_jobs=2)
    adapter = _EchoAdapter(settings=settings)
    app = create_app(adapter, settings, title="Echo Test")

    @app.post("/api/echo")
    def echo(req: _EchoRequest):
        return app.state.runner.submit(
            build_argv=lambda _job_id, job_dir: _echo_argv(req, job_dir),
            label="echo",
        )

    return TestClient(app)


def _wait_for_terminal(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        r.raise_for_status()
        body = r.json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id!r} did not finish within {timeout}s")


def test_health(client: TestClient) -> None:
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "echo"
    assert health["version"] == "0.1.0"
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "echo"
    assert detail["version"] == "0.1.0"
    assert "disk_usage_mb" in detail


def test_openapi_includes_request_model(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    # The service-specific request model is registered.
    assert "_EchoRequest" in schema["components"]["schemas"] or any(
        name.endswith("EchoRequest") for name in schema["components"]["schemas"]
    )
    # The framework's JobInfo response model is registered.
    assert "JobInfo" in schema["components"]["schemas"]


def test_full_success_lifecycle(client: TestClient) -> None:
    submit = client.post("/api/echo", json={"message": "hello"})
    submit.raise_for_status()
    job_id = submit.json()["job_id"]

    final = _wait_for_terminal(client, job_id)
    assert final["status"] == JobStatus.COMPLETED.value
    assert final["failure_kind"] is None

    files = client.get(f"/api/jobs/{job_id}/files").json()
    assert files["files"] == ["msg.txt"]

    single = client.get(f"/api/jobs/{job_id}/file/msg.txt")
    assert single.status_code == 200
    assert single.content.strip() == b"hello"

    download = client.get(f"/api/jobs/{job_id}/download")
    assert download.status_code == 200
    with zipfile.ZipFile(BytesIO(download.content)) as zf:
        assert zf.read("msg.txt").strip() == b"hello"

    log = client.get(f"/api/jobs/{job_id}/log").json()
    assert log["job_id"] == job_id


def test_subprocess_failure_attaches_summary(client: TestClient) -> None:
    submit = client.post("/api/echo", json={"message": "oops", "fail": True})
    job_id = submit.json()["job_id"]
    final = _wait_for_terminal(client, job_id)
    assert final["status"] == JobStatus.FAILED.value
    assert final["failure_kind"] == "subprocess_error"
    assert final["error_summary"] is not None
    assert "ValueError" in final["error_summary"]
    assert final["error_tail"] is not None


def test_zero_rc_but_no_outputs_is_distinct_failure(client: TestClient) -> None:
    submit = client.post("/api/echo", json={"message": "x", "produce_output": False})
    job_id = submit.json()["job_id"]
    final = _wait_for_terminal(client, job_id)
    assert final["status"] == JobStatus.FAILED.value
    assert final["failure_kind"] == "no_outputs"


def test_download_before_completion_returns_400(client: TestClient) -> None:
    # Use a job that doesn't exist in a "completed" state — easier: create then
    # check download on a still-pending one. Submit + immediately try download.
    submit = client.post("/api/echo", json={"message": "x"})
    job_id = submit.json()["job_id"]
    r = client.get(f"/api/jobs/{job_id}/download")
    # Either 400 (still pending) or 200 (already done in <1ms — unlikely but possible).
    # We accept either; the goal is "never 5xx".
    assert r.status_code in (200, 400)
    _wait_for_terminal(client, job_id)


def test_path_traversal_blocked(client: TestClient) -> None:
    submit = client.post("/api/echo", json={"message": "x"})
    job_id = submit.json()["job_id"]
    _wait_for_terminal(client, job_id)
    r = client.get(f"/api/jobs/{job_id}/file/../../etc/passwd")
    # FastAPI may collapse the path in the URL; either 400 (our check) or 404 (file
    # didn't exist) — never a successful disclosure.
    assert r.status_code in (400, 404)


def test_delete_clears_store_and_dir(client: TestClient, tmp_path: Path) -> None:
    submit = client.post("/api/echo", json={"message": "x"})
    job_id = submit.json()["job_id"]
    _wait_for_terminal(client, job_id)

    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200

    r2 = client.get(f"/api/jobs/{job_id}")
    assert r2.status_code == 404


def test_build_argv_exception_cleans_up_partial_job(client: TestClient, tmp_path: Path) -> None:
    """If the endpoint's build_argv callback raises, the job dir + store entry
    must be torn down — otherwise an upstream validation error (bad zip, missing
    input file) leaves orphaned PENDING jobs around."""
    app = client.app  # type: ignore[attr-defined]

    @app.post("/api/explode")
    def explode():
        from fastapi import HTTPException

        def _build(_job_id: str, _job_dir: Path):
            raise HTTPException(status_code=422, detail="bad input")

        return app.state.runner.submit(build_argv=_build, label="explode")

    before = len(list(app.state.settings.jobs_base_dir.glob("*")))
    r = client.post("/api/explode")
    assert r.status_code == 422
    assert r.json()["detail"] == "bad input"
    after = len(list(app.state.settings.jobs_base_dir.glob("*")))
    # No new job dirs lingering on disk.
    assert after == before
    # Nothing in the store either.
    assert app.state.job_store.all_jobs() == []


def test_two_instances_sharing_nas_observe_each_others_jobs(tmp_path: Path) -> None:
    """Simulates two FC instances mounted on the same NAS.

    Instance A submits a job; instance B (a separate FastAPI app with the same
    `jobs_base_dir`) must be able to answer `GET /api/jobs/{id}` for it via the
    read-through cache. This is the consistency model FC actually exposes: a
    poll-after-submit may land on a different instance from the submit.
    """
    shared_dir = tmp_path / "shared_jobs"

    settings_a = _EchoSettings(jobs_base_dir=shared_dir, max_concurrent_jobs=2)
    adapter_a = _EchoAdapter(settings=settings_a)
    app_a = create_app(adapter_a, settings_a, title="Echo A")

    @app_a.post("/api/echo")
    def echo_a(req: _EchoRequest):
        return app_a.state.runner.submit(
            build_argv=lambda _id, jd: _echo_argv(req, jd), label="echo"
        )

    settings_b = _EchoSettings(jobs_base_dir=shared_dir, max_concurrent_jobs=2)
    adapter_b = _EchoAdapter(settings=settings_b)
    app_b = create_app(adapter_b, settings_b, title="Echo B")

    with TestClient(app_a) as ca, TestClient(app_b) as cb:
        # Submit on A.
        submit = ca.post("/api/echo", json={"message": "cross-instance"})
        job_id = submit.json()["job_id"]
        final = _wait_for_terminal(ca, job_id)
        assert final["status"] == JobStatus.COMPLETED.value

        # Bump the sidecar mtime past whatever B might have cached during creation
        # so B's read-through is guaranteed to see the COMPLETED state.
        import os
        sidecar = shared_dir / job_id / "job.json"
        future = sidecar.stat().st_mtime + 1.0
        os.utime(sidecar, (future, future))

        # Poll on B — must find it.
        r = cb.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        assert r.json()["status"] == JobStatus.COMPLETED.value

        # Outputs are on the shared NAS → B can download even though A produced them.
        dl = cb.get(f"/api/jobs/{job_id}/download")
        assert dl.status_code == 200

        # A deletes the job — B's next GET observes the deletion via stat eviction.
        assert ca.delete(f"/api/jobs/{job_id}").status_code == 200
        assert cb.get(f"/api/jobs/{job_id}").status_code == 404


def test_simulated_restart_recovers_jobs(tmp_path: Path) -> None:
    """Submit a job, tear the app down, build a fresh one pointed at the same dir.

    The recovered job should be query-able and downloadable — i.e., a client that
    polled across an FC container restart can keep going without losing state.
    """
    jobs_dir = tmp_path / "jobs"
    settings = _EchoSettings(jobs_base_dir=jobs_dir, max_concurrent_jobs=2)
    adapter = _EchoAdapter(settings=settings)

    app1 = create_app(adapter, settings, title="Echo Restart")

    @app1.post("/api/echo")
    def _submit(req: _EchoRequest):
        return app1.state.runner.submit(
            build_argv=lambda _id, jd: _echo_argv(req, jd), label="echo"
        )

    with TestClient(app1) as c1:
        resp = c1.post("/api/echo", json={"message": "persist-me"})
        job_id = resp.json()["job_id"]
        final = _wait_for_terminal(c1, job_id)
        assert final["status"] == JobStatus.COMPLETED.value

    # Same jobs_base_dir, brand-new app instance — simulates an FC cold start.
    settings2 = _EchoSettings(jobs_base_dir=jobs_dir, max_concurrent_jobs=2)
    adapter2 = _EchoAdapter(settings=settings2)
    app2 = create_app(adapter2, settings2, title="Echo Restart Round 2")

    with TestClient(app2) as c2:
        # The previously-completed job is back, with status preserved.
        r = c2.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        assert r.json()["status"] == JobStatus.COMPLETED.value
        # Outputs are still on disk → /download still works.
        dl = c2.get(f"/api/jobs/{job_id}/download")
        assert dl.status_code == 200

"""Task endpoint e2e tests using an EchoAdapter.

Mirrors test_app_e2e.py but exercises register_task_endpoint instead of the
submit/poll runner.  Validates that:
  - the endpoint blocks until subprocess completion
  - response carries the final JobInfo (status=completed)
  - client-supplied job_id (X-Bioagent-Job-Id) is honored
  - duplicate submissions return the existing job
  - subprocess failure → status=failed with error_summary populated
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from bioagent_service import JobAdapter, ServiceSettings, create_app, register_task_endpoint
from bioagent_service.models import JobStatus


class _EchoRequest(BaseModel):
    message: str = "hello"
    fail: bool = False


class _EchoSettings(ServiceSettings):
    model_config = SettingsConfigDict(env_file=None, env_prefix="ECHO_TASK_", extra="ignore")


class _EchoAdapter(JobAdapter):
    name = "echo"


def _echo_argv(req: _EchoRequest, job_id: str, job_dir: Path) -> list[str]:
    out = job_dir / "output"
    out.mkdir(parents=True, exist_ok=True)
    if req.fail:
        script = "echo 'Traceback (most recent call last):' >&2; echo 'ValueError: boom' >&2; exit 1"
    else:
        script = f"echo {req.message!r} > {out / 'msg.txt'}"
    return ["bash", "-c", script]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = _EchoSettings(jobs_base_dir=tmp_path / "jobs", max_concurrent_jobs=2, keepalive_interval_s=0)
    adapter = _EchoAdapter(settings=settings)
    app = create_app(adapter, settings, title="Echo Task Test")
    register_task_endpoint(
        app,
        path="/api/tasks/echo",
        label="echo",
        request_model=_EchoRequest,
        build_argv=_echo_argv,
    )
    return TestClient(app)


def test_task_endpoint_blocks_until_completion(client: TestClient) -> None:
    r = client.post("/api/tasks/echo", data={"message": "world"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == JobStatus.COMPLETED.value
    assert body["output_count"] == 1
    # Follow-up GET should see the persisted COMPLETED state (not a
    # transient cached value), confirming the JobStore wrote the
    # terminal lifecycle through.
    g = client.get(f"/api/jobs/{body['job_id']}")
    assert g.status_code == 200
    persisted = g.json()
    assert persisted["status"] == JobStatus.COMPLETED.value
    assert persisted["output_count"] == 1
    assert persisted["completed_at"] is not None


def test_task_endpoint_reads_job_id_header(client: TestClient) -> None:
    r = client.post(
        "/api/tasks/echo",
        data={"message": "hi"},
        headers={"X-Bioagent-Job-Id": "my-task-001"},
    )
    assert r.status_code == 200
    assert r.json()["job_id"] == "my-task-001"


def test_task_endpoint_reads_fc_async_task_id_fallback(client: TestClient) -> None:
    r = client.post(
        "/api/tasks/echo",
        data={"message": "hi"},
        headers={"X-Fc-Async-Task-Id": "fc-task-001"},
    )
    assert r.json()["job_id"] == "fc-task-001"


def test_task_endpoint_duplicate_returns_existing(client: TestClient) -> None:
    hdrs = {"X-Bioagent-Job-Id": "dup-001"}
    r1 = client.post("/api/tasks/echo", data={"message": "a"}, headers=hdrs)
    r2 = client.post("/api/tasks/echo", data={"message": "b"}, headers=hdrs)
    assert r1.json()["job_id"] == r2.json()["job_id"]
    # First job's message should win; the second one should NOT have re-run.
    assert r2.json()["input_params"]["message"] == "a"
    # created_at is set in store.create and would change on re-run.
    assert r1.json()["created_at"] == r2.json()["created_at"]


def test_task_endpoint_subprocess_failure(client: TestClient) -> None:
    r = client.post("/api/tasks/echo", data={"message": "hi", "fail": True})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == JobStatus.FAILED.value
    assert "boom" in (body.get("error_summary") or "")


def test_task_endpoint_disabled_when_setting_false(tmp_path: Path) -> None:
    settings = _EchoSettings(jobs_base_dir=tmp_path / "jobs", task_endpoints_enabled=False, keepalive_interval_s=0)
    adapter = _EchoAdapter(settings=settings)
    app = create_app(adapter, settings, title="Echo Disabled")
    register_task_endpoint(
        app,
        path="/api/tasks/echo",
        label="echo",
        request_model=_EchoRequest,
        build_argv=_echo_argv,
    )
    client = TestClient(app)
    # Route not registered → 404
    assert client.post("/api/tasks/echo", data={"message": "x"}).status_code == 404


# ---------------------------------------------------------------------------
# Custom-signature endpoint pattern (file uploads) — what services with
# multipart inputs (boltzgen-server, dockq-server, etc.) will follow.
# ---------------------------------------------------------------------------
from fastapi import Depends, File, Header, Request, UploadFile  # noqa: E402

from bioagent_service import execute_task  # noqa: E402
from bioagent_service.forms import model_form_depends  # noqa: E402
from bioagent_service.models import JobInfo  # noqa: E402
from typing import Optional  # noqa: E402


def test_execute_task_with_file_upload(tmp_path: Path) -> None:
    """Verify execute_task works when caller defines a custom endpoint with UploadFile.

    This is the pattern services like boltzgen-server use: their endpoint
    accepts multipart form fields including UploadFile, persists them in
    a save_inputs callback, then delegates to execute_task for the actual
    pipeline run.
    """
    settings = _EchoSettings(jobs_base_dir=tmp_path / "jobs", keepalive_interval_s=0)
    adapter = _EchoAdapter(settings=settings)
    app = create_app(adapter, settings, title="Echo Upload")

    @app.post("/api/tasks/echo-upload", response_model=JobInfo)
    def post_upload(
        request: Request,
        params: _EchoRequest = Depends(model_form_depends(_EchoRequest)),
        attachment: UploadFile = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
    ) -> JobInfo:
        job_id = x_bioagent_job_id or "upload-001"

        def _save(_req, input_dir: Path) -> None:
            if attachment and attachment.filename:
                (input_dir / attachment.filename).write_bytes(attachment.file.read())

        return execute_task(
            request,
            job_id=job_id,
            label="echo-upload",
            params=params,
            build_argv=_echo_argv,
            save_inputs=_save,
        )

    client = TestClient(app)
    payload = b"hello attachment"
    r = client.post(
        "/api/tasks/echo-upload",
        data={"message": "hi"},
        files={"attachment": ("a.txt", payload, "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["status"] == JobStatus.COMPLETED.value
    # The saved file should land under the job dir's input/.
    saved = (tmp_path / "jobs" / "upload-001" / "input" / "a.txt").read_bytes()
    assert saved == payload

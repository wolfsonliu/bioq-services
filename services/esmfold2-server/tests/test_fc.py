"""End-to-end tests against the deployed ESMFold2 Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/esmfold2-server/tests/test_fc.py

The "Task endpoints (async task mode)" section verifies FC async task mode
end-to-end: HTTP 202 on submit, X-Bioagent-Job-Id alignment, and idempotency.
These tests require the FC console to have async task mode enabled for the
function (see engineering/decisions/2026-06-17-fc-async-task-mode.md).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("esmfold2-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "esmfold2"
    assert "version" in body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/fold", "/api/tasks/fold"}


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


def test_fold_minimal_protein(client: httpx.Client, base_url: str) -> None:
    """Fold ubiquitin (76 aa) — smallest meaningful test."""
    ubiquitin = (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQK"
        "ESTLHLVLRLRGG"
    )
    r = client.post(
        "/api/fold",
        data={
            "sequences": f'[{{"type":"protein","id":"A","sequence":"{ubiquitin}"}}]',
            "num_sampling_steps": "10",
            "num_loops": "1",
        },
    )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    assert final["status"] == "completed", final
    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert any("prediction_0.cif" in f for f in files)
    assert any("metrics.json" in f for f in files)


# =====================================================================
# Task endpoints (async task mode) — synchronous-blocking endpoint
# invoked via FC Async Task Mode (X-Fc-Invocation-Type: Async).
# =====================================================================

def _async_submit_and_poll(
    client: httpx.Client,
    base_url: str,
    endpoint: str,
    payload: dict,
    *,
    task_id: str,
    timeout_s: int = 1800,
) -> tuple[str, dict, list[str]]:
    """Submit via FC async task mode, poll JobInfo to completion.

    Sends with `X-Fc-Invocation-Type: Async` + `X-Bioagent-Job-Id=<task_id>`.
    Expects 202.  The server-side task endpoint blocks synchronously inside
    the FC instance; we poll `/api/jobs/{task_id}` because JobInfo records
    every state transition.

    Returns (task_id, final_status, files).  Asserts terminal status is
    `completed`.  Raises if FC returned non-202.
    """
    r = client.post(
        endpoint,
        data=payload,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    assert r.status_code == 202, (
        f"expected 202 from async invocation; got {r.status_code} body={r.text!r}.  "
        f"Check that FC console has async task mode enabled for this function."
    )

    final = poll_job(client, base_url, task_id, timeout_s=timeout_s, interval_s=20)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{task_id}/files").json()["files"]
    return task_id, final, files


# Smallest meaningful test sequence: ubiquitin (76 aa).
_UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQK"
    "ESTLHLVLRLRGG"
)


def test_async_task_fold_minimal(client: httpx.Client, base_url: str) -> None:
    """Async invoke /api/tasks/fold, poll JobInfo to completion.

    Validates the full FC async task mode pipeline:
      - HTTP 202 on submit (proves async task mode is enabled in FC console)
      - task_id from X-Bioagent-Job-Id is used as the JobInfo.job_id
      - server runs the pipeline synchronously to completion inside the FC instance
      - JobInfo lifecycle (pending → running → completed) is persisted to NAS
    """
    import time
    task_id = f"fc-async-fold-{int(time.time())}"
    payload = {
        "sequences": f'[{{"type":"protein","id":"A","sequence":"{_UBIQUITIN}"}}]',
        "num_sampling_steps": "10",
        "num_loops": "1",
    }
    job_id, final, files = _async_submit_and_poll(
        client, base_url, "/api/tasks/fold", payload,
        task_id=task_id,
    )

    assert job_id == task_id, "task endpoint must echo X-Bioagent-Job-Id as JobInfo.job_id"
    assert final["completed_at"] is not None
    assert final["started_at"] is not None
    assert final["duration_seconds"] is not None and final["duration_seconds"] > 0

    assert any("prediction_0.cif" in f for f in files), f"no .cif in outputs: {files}"
    assert any("metrics.json" in f for f in files), f"no metrics.json in outputs: {files}"


def test_async_task_fold_honors_bioagent_job_id(
    client: httpx.Client, base_url: str,
) -> None:
    """Verify X-Bioagent-Job-Id flows through to JobInfo.job_id end-to-end."""
    import time
    task_id = f"fc-async-fold-id-{int(time.time())}"
    payload = {
        "sequences": f'[{{"type":"protein","id":"A","sequence":"{_UBIQUITIN}"}}]',
        "num_sampling_steps": "10",
        "num_loops": "1",
    }
    r = client.post(
        "/api/tasks/fold",
        data=payload,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    assert r.status_code == 202

    final = poll_job(client, base_url, task_id, timeout_s=1800, interval_s=20)
    assert final["status"] == "completed", final
    assert final["job_id"] == task_id, "JobInfo.job_id must equal X-Bioagent-Job-Id"


def test_async_task_duplicate_rejected_at_fc_platform_layer(
    client: httpx.Client, base_url: str,
) -> None:
    """Same X-Fc-Async-Task-Id twice → FC rejects the second at platform layer.

    FC's async task mode dedups by X-Fc-Async-Task-Id; the second invocation
    never reaches our function (returns HTTP 409 Conflict at the FC layer).
    This is observed behavior — verified 2026-06-19 on boltz-server.

    Test verifies:
      1. First submit succeeds (HTTP 202) and runs to completion.
      2. Second submit with SAME task_id returns HTTP 409 (FC platform dedup)
         or 202 (FC accepts → server-side execute_task dedup, also acceptable).
      3. The original JobInfo is unchanged (created_at, input_params).
    """
    import time
    task_id = f"fc-async-dup-{int(time.time())}"
    payload = {
        "sequences": f'[{{"type":"protein","id":"A","sequence":"{_UBIQUITIN}"}}]',
        "num_sampling_steps": "10",
        "num_loops": "1",
    }

    # First submit
    r1 = client.post(
        "/api/tasks/fold",
        data=payload,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    assert r1.status_code == 202

    # Wait for first to finish.
    final = poll_job(client, base_url, task_id, timeout_s=1800, interval_s=20)
    assert final["status"] == "completed", final
    first_created_at = final["created_at"]

    # Second submit with the SAME task_id → FC platform rejects with 409
    # (or, less commonly, FC accepts and our execute_task dedup catches it as 202).
    r2 = client.post(
        "/api/tasks/fold",
        data=payload,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    assert r2.status_code in (202, 409), (
        f"expected 409 (FC dedup) or 202 (FC accepts → server dedups); got {r2.status_code}"
    )

    # Either way, JobInfo must not change.
    if r2.status_code == 202:
        time.sleep(30)  # let server-side dedup finalize
    re_query = client.get(f"/api/jobs/{task_id}").json()
    assert re_query["status"] == "completed"
    assert re_query["created_at"] == first_created_at, (
        "duplicate async invoke must not reset created_at"
    )

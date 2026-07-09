"""FC async task mode tests for reinvent-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/reinvent-server/tests/test_fc_task.py -v

Validates the FC async task-mode endpoints (``X-Fc-Invocation-Type: Async``,
``X-Fc-Async-Task-Id``) end-to-end.  Async task mode pins the FC instance for
the whole job (no 30 s HTTP-gateway recycle risk) and dedups by
``X-Fc-Async-Task-Id`` at the platform layer.  Polling uses the same
``poll_job`` helper the sibling turbohopp test uses (backed by
``GET /api/jobs/<id>``).

Payload sizing — 128 KiB async cap
----------------------------------
FC's async invocation gateway caps the inbound event payload at ~128 KiB
(``EntityTooLarge`` 400 otherwise).  Large prior/model files therefore **cannot**
be uploaded over async multipart.  These tests deliberately use small/no-file
payloads:

  * ``/api/tasks/sampling`` — generator=reinvent, num_smiles=20, NO file →
    tiny body, no upload; the prior is loaded from NAS on the instance.
  * ``/api/tasks/scoring`` — a few inline SMILES in a small ``.smi`` plus a
    JSON-encoded ``scoring`` field.

202-before-validation pitfall
-----------------------------
FC async mode returns 202 *before* the function body validates the request.
A malformed / non-JSON complex form field (``scoring`` here) makes submit APPEAR
to succeed (202) but the job is never created → ``GET /api/jobs/<id>`` 404s
forever (a phantom 404).  We JSON-encode ``scoring`` exactly as the sync path
expects, and assert the job reaches a *real* terminal state — not a phantom 404.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "reinvent-server"

DATA_DIR = Path(__file__).resolve().parent / "data"

# Tiny inline SMILES — keeps the async scoring payload well under 128 KiB.
SMALL_SMI = "CCO\nc1ccccc1\nCC(=O)O\n"

# JSON-encoded [scoring] section. This exact encoding is what prevents the
# 202-before-validation phantom-404: the sync path decodes it via
# model_form_depends before the job is created.
SCORING_SPEC = {
    "type": "geometric_mean",
    "component": [{"QED": {"endpoint": [{"name": "QED", "weight": 1.0}]}}],
}

POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 15

TIMEOUT = httpx.Timeout(connect=30, read=300, write=600, pool=30)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture(scope="module")
def sampling_task_id() -> str:
    return f"fc-async-sampling-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def scoring_task_id() -> str:
    return f"fc-async-scoring-{int(time.time())}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Helpers (mirror turbohopp-server/tests/test_fc_task.py)
# ---------------------------------------------------------------------------


def _async_headers(task_id: str) -> dict[str, str]:
    return {
        "X-Fc-Invocation-Type": "Async",
        "X-Bioagent-Job-Id": task_id,
        "X-Fc-Async-Task-Id": task_id,
    }


def _get_with_retry(
    client: httpx.Client,
    path: str,
    *,
    max_attempts: int = 20,
    backoff_s: int = 30,
) -> httpx.Response:
    """GET that retries on FC HTTP-gateway 429 throttling."""
    last: httpx.Response | None = None
    for _attempt in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


def _poll_to_terminal(client: httpx.Client, task_id: str) -> dict:
    # max_concurrent_jobs may be 1 here → GET /api/jobs/<id> can 429 for
    # minutes at a stretch.  Bump max_transient_errors well above the framework
    # default (10) so poll_job rides out throttle windows.
    return poll_job(
        client, "", task_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        max_transient_errors=60,
    )


# ---------------------------------------------------------------------------
# submit + poll fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sampling_submit_response(
    client: httpx.Client,
    sampling_task_id: str,
) -> httpx.Response:
    # No file, tiny body → safe under the 128 KiB async cap.
    return client.post(
        "/api/tasks/sampling",
        data={"generator": "reinvent", "num_smiles": "20"},
        headers=_async_headers(sampling_task_id),
    )


@pytest.fixture(scope="module")
def sampling_task(
    client: httpx.Client,
    sampling_task_id: str,
    sampling_submit_response: httpx.Response,
) -> dict:
    assert sampling_submit_response.status_code == 202, (
        f"async sampling submit returned "
        f"{sampling_submit_response.status_code}: "
        f"{sampling_submit_response.text!r}"
    )
    final = _poll_to_terminal(client, sampling_task_id)
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


@pytest.fixture(scope="module")
def scoring_submit_response(
    client: httpx.Client,
    scoring_task_id: str,
) -> httpx.Response:
    # `scoring` MUST be JSON-encoded (see module docstring: 202-before-validation).
    return client.post(
        "/api/tasks/scoring",
        data={"scoring": json.dumps(SCORING_SPEC)},
        files={"smiles_file": ("compounds.smi", SMALL_SMI.encode(), "text/plain")},
        headers=_async_headers(scoring_task_id),
    )


@pytest.fixture(scope="module")
def scoring_task(
    client: httpx.Client,
    scoring_task_id: str,
    scoring_submit_response: httpx.Response,
) -> dict:
    assert scoring_submit_response.status_code == 202, (
        f"async scoring submit returned "
        f"{scoring_submit_response.status_code}: "
        f"{scoring_submit_response.text!r}"
    )
    return _poll_to_terminal(client, scoring_task_id)


# ===================================================================
# Section 1: Submit semantics + OpenAPI registration
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_sampling_returns_202(self, sampling_submit_response):
        assert sampling_submit_response.status_code == 202, (
            f"expected 202; got {sampling_submit_response.status_code} "
            f"body={sampling_submit_response.text!r}"
        )

    def test_task_endpoints_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200, (
            f"openapi.json fetch failed: {r.status_code} {r.text!r}"
        )
        spec = r.json()
        assert "/api/tasks/sampling" in spec["paths"], (
            "task endpoint missing from OpenAPI; "
            "settings.task_endpoints_enabled may be False"
        )
        assert "/api/tasks/scoring" in spec["paths"]


# ===================================================================
# Section 2: sampling — small payload, no file
# ===================================================================


@pytest.mark.fc
class TestAsyncSampling:
    def test_completed(self, sampling_task, sampling_task_id):
        assert sampling_task["status"] == "completed"
        assert sampling_task["job_id"] == sampling_task_id
        assert sampling_task.get("started_at") is not None
        assert sampling_task.get("completed_at") is not None
        assert sampling_task.get("duration_seconds") is not None
        assert sampling_task["duration_seconds"] > 0
        assert sampling_task.get("output_count", 0) > 0
        assert sampling_task.get("output_total_bytes", 0) > 0

    def test_input_params_echoed(self, sampling_task):
        params = sampling_task.get("input_params") or {}
        assert params.get("generator") == "reinvent"
        assert params.get("num_smiles") == 20

    def test_sampling_csv_produced(self, client, sampling_task_id, sampling_task):
        files = _get_with_retry(
            client, f"/api/jobs/{sampling_task_id}/files",
        ).json()["files"]
        assert any(f.endswith("sampling.csv") for f in files), (
            f"no sampling.csv in outputs: {files}"
        )


# ===================================================================
# Section 3: scoring — JSON-encoded `scoring` field guards against phantom-404
# ===================================================================


@pytest.mark.fc
class TestAsyncScoring:
    def test_reaches_real_terminal_state(self, scoring_task, scoring_task_id):
        # The whole point: a well-formed (JSON-encoded `scoring`) async submit
        # must reach a REAL terminal state — completed, or a genuine failed with
        # an error_summary — NOT a phantom 404 (which is what a malformed
        # complex field would produce; see the 202-before-validation pitfall in
        # the module docstring). poll_job would raise TimeoutError on a phantom
        # 404, so simply reaching here proves the job was actually created.
        assert scoring_task["job_id"] == scoring_task_id
        assert scoring_task["status"] in ("completed", "failed")
        if scoring_task["status"] == "failed":
            assert scoring_task.get("error_summary"), (
                "a real FAILED must carry an error_summary, not be a phantom 404"
            )

    def test_score_results_when_completed(self, client, scoring_task_id, scoring_task):
        if scoring_task["status"] != "completed":
            pytest.skip(f"scoring task not completed: {scoring_task['status']}")
        files = _get_with_retry(
            client, f"/api/jobs/{scoring_task_id}/files",
        ).json()["files"]
        assert any(f.endswith("score_results.csv") for f in files), (
            f"no score_results.csv in outputs: {files}"
        )


# ===================================================================
# Section 4: Job lifecycle on the async sampling task
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, sampling_task_id, sampling_task):
        r = _get_with_retry(client, f"/api/jobs/{sampling_task_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == sampling_task_id
        assert body["status"] == "completed"

    def test_log_endpoint(self, client, sampling_task_id, sampling_task):
        r = _get_with_retry(client, f"/api/jobs/{sampling_task_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)

"""FC async task mode tests for semlaflow-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/semlaflow-server/tests/test_fc_task.py -v

Validates ``POST /api/tasks/generate`` end-to-end in FC async task mode
(``X-Fc-Invocation-Type: Async``).  Async task mode pins one FC instance
for the entire pipeline (no 30 s HTTP gateway recycle) and dedups by
``X-Fc-Async-Task-Id`` at the platform layer.

SemlaFlow generation is unconditional — no file uploads — so all payloads
sit comfortably under FC's 128 KiB async event cap.
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "semlaflow-server"

# qm9 n_molecules=10 / integration_steps=50 incl. novelty ref build + cold
# start + NAS load. Generous ceiling (see design §12.1).
POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 15

TIMEOUT = httpx.Timeout(connect=30, read=120, write=60, pool=30)


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
def generate_task_id() -> str:
    return f"fc-async-gen-{int(time.time())}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Helpers
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
    max_attempts: int = 10,
    backoff_s: int = 20,
) -> httpx.Response:
    """GET that retries on FC HTTP-gateway 429 throttling.

    See project memory ``project_fc_http_polling_unreliable_at_concurrency``.
    """
    last: httpx.Response | None = None
    for _ in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


def _poll_to_completion(client: httpx.Client, task_id: str) -> dict:
    final = poll_job(
        client, "", task_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


def _count_sdf_mols(sdf_bytes: bytes) -> int:
    return sdf_bytes.decode("utf-8", errors="replace").count("$$$$")


# ---------------------------------------------------------------------------
# Module-scoped submit + task fixtures — one inference for the whole suite.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generate_submit_response(
    client: httpx.Client, generate_task_id: str,
) -> httpx.Response:
    return client.post(
        "/api/tasks/generate",
        data={
            "model_name": "qm9",
            "n_molecules": "10",
            "integration_steps": "50",
            "seed": "42",
        },
        headers=_async_headers(generate_task_id),
    )


@pytest.fixture(scope="module")
def generate_task(
    client: httpx.Client,
    generate_task_id: str,
    generate_submit_response: httpx.Response,
) -> dict:
    assert generate_submit_response.status_code == 202, (
        f"async generate submit returned {generate_submit_response.status_code}: "
        f"{generate_submit_response.text!r}"
    )
    return _poll_to_completion(client, generate_task_id)


# ===================================================================
# Section 1: submit semantics + OpenAPI
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_generate_returns_202(self, generate_submit_response):
        assert generate_submit_response.status_code == 202, (
            f"expected 202; got {generate_submit_response.status_code} "
            f"body={generate_submit_response.text!r}"
        )

    def test_task_endpoints_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert "/api/tasks/generate" in spec.get("paths", {}), (
            "task endpoint missing from OpenAPI; "
            "settings.task_endpoints_enabled may be False"
        )


# ===================================================================
# Section 2: completion + outputs
# ===================================================================


@pytest.mark.fc
class TestAsyncGenerate:
    def test_completed(self, generate_task, generate_task_id, client):
        assert generate_task["status"] == "completed"
        assert generate_task["job_id"] == generate_task_id
        assert generate_task.get("started_at") is not None
        assert generate_task.get("completed_at") is not None
        d = generate_task.get("duration_seconds")
        assert d is not None and d > 3.0, (
            f"duration {d}s too short — subprocess may not have really run"
        )
        assert generate_task.get("output_count", 0) > 0

        r = _get_with_retry(client, f"/api/jobs/{generate_task_id}/files")
        assert r.status_code == 200
        files = r.json()["files"]
        assert "predictions.smol.sdf" in files
        assert "metrics.json" in files
        assert "generation_stats.json" in files

    def test_input_params_echoed(self, generate_task):
        params = generate_task.get("input_params") or {}
        assert params.get("model_name") == "qm9"
        assert params.get("n_molecules") == 10
        assert params.get("integration_steps") == 50
        assert params.get("seed") == 42

    def test_sdf_downloadable(self, client, generate_task_id, generate_task):
        r = _get_with_retry(
            client, f"/api/jobs/{generate_task_id}/file/predictions.smol.sdf"
        )
        assert r.status_code == 200
        assert len(r.content) > 100, "SDF too small — sampling likely failed"
        n_mols = _count_sdf_mols(r.content)
        assert 1 <= n_mols <= 10, f"expected 1-10 mols in SDF, got {n_mols}"

    def test_generation_stats_json(self, client, generate_task_id, generate_task):
        r = _get_with_retry(
            client, f"/api/jobs/{generate_task_id}/file/generation_stats.json"
        )
        assert r.status_code == 200
        stats = r.json()
        assert stats["n_requested"] == 10
        assert stats["n_valid"] >= 1
        assert stats["seed"] == 42

    def test_metrics_json(self, client, generate_task_id, generate_task):
        r = _get_with_retry(
            client, f"/api/jobs/{generate_task_id}/file/metrics.json"
        )
        assert r.status_code == 200
        metrics = r.json()
        assert "validity" in metrics


# ===================================================================
# Section 3: identity — X-Bioagent-Job-Id propagates
# ===================================================================


@pytest.mark.fc
class TestAsyncTaskIdentity:
    def test_job_id_matches_task_id(self, generate_task, generate_task_id):
        assert generate_task["job_id"] == generate_task_id, (
            "task endpoint must use X-Bioagent-Job-Id as JobInfo.job_id"
        )


# ===================================================================
# Section 4: job lifecycle on the shared task
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, generate_task_id, generate_task):
        r = _get_with_retry(client, f"/api/jobs/{generate_task_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == generate_task_id
        assert body["status"] == "completed"

    def test_log_endpoint(self, client, generate_task_id, generate_task):
        r = _get_with_retry(client, f"/api/jobs/{generate_task_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text") or ""
        assert isinstance(log_text, str)

    def test_download_zip(self, client, generate_task_id, generate_task):
        r = _get_with_retry(client, f"/api/jobs/{generate_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any("predictions.smol.sdf" in n for n in names), (
            f"predictions.smol.sdf missing from zip: {names}"
        )

    def test_missing_file_returns_404(
        self, client, generate_task_id, generate_task,
    ):
        r = _get_with_retry(
            client, f"/api/jobs/{generate_task_id}/file/nonexistent.xyz"
        )
        assert r.status_code == 404


# ===================================================================
# Section 5: platform-layer dedup — same X-Fc-Async-Task-Id must not rerun
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmit same X-Fc-Async-Task-Id after completion.

    Per FC async task mode contract (see engineering/decisions/
    2026-06-17-fc-async-task-mode.md and project memory
    ``project_fc_async_dedup_at_platform_layer.md``), a duplicate should
    return 409 without invoking the function; if FC forwards, the framework
    layer (``execute_task``) returns the existing JobInfo without rerun.
    """

    def test_duplicate_does_not_rerun(
        self,
        client: httpx.Client,
        generate_task_id: str,
        generate_task: dict,
    ):
        first_created_at = generate_task["created_at"]
        first_completed_at = generate_task["completed_at"]
        first_seed = (generate_task.get("input_params") or {}).get("seed")

        # Resubmit with a different seed — must not stick.
        r2 = client.post(
            "/api/tasks/generate",
            data={"model_name": "qm9", "n_molecules": "10",
                  "integration_steps": "50", "seed": "999"},
            headers=_async_headers(generate_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (framework dedups); "
            f"got {r2.status_code} body={r2.text!r}"
        )

        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(client, f"/api/jobs/{generate_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("seed") == first_seed, (
            "duplicate async submit must not overwrite input_params"
        )

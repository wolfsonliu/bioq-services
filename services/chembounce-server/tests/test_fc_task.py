"""FC async task mode tests for chembounce-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/chembounce-server/tests/test_fc_task.py -v

Validates ``/api/tasks/scaffold_hop`` end-to-end against the deployed FC
function in async task mode (``X-Fc-Invocation-Type: Async``).

Async task mode pins the FC instance for the whole scaffold-hop run (no
30 s HTTP-gateway recycle risk) and dedups by ``X-Fc-Async-Task-Id`` at
the platform layer.  ChemBounce on the smallest useful input still takes
several minutes; the long-tail "complex molecule" cases can take 30+ min,
so we stick to one small inference (losartan + ``frag_max_n=10`` + 250mw
DB) and keep the module-scoped fixture pattern to reuse that single run
across all assertions.

Input is form-only (SMILES + numeric thresholds) — well under FC's
128 KiB async payload cap; no file staging needed.

After long polling runs FC's HTTP gateway sometimes returns 429 on
follow-up GETs (see project memory
``project_fc_http_polling_unreliable_at_concurrency``), so auxiliary
status/files/download requests go through ``_get_with_retry``.
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "chembounce-server"

# Losartan — same example SMILES used in test_fc.py and upstream README.
LOSARTAN = "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"

# ChemBounce single-fragment path with frag_max_n=10 + 250mw DB takes
# ~5-15 min after cold start + weight load.  Give it 30 min to absorb
# the NAS-mount latency + fingerprint file mmap.
POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 20

TIMEOUT = httpx.Timeout(connect=30, read=300, write=60, pool=30)


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
def scaffold_hop_task_id() -> str:
    return f"fc-async-hop-{int(time.time())}-{uuid.uuid4().hex[:6]}"


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

    After a long-running async task the FC HTTP gateway can rate-limit
    subsequent GETs to ``/api/jobs/...``.  This is a platform-layer
    artifact — see project memory
    ``project_fc_http_polling_unreliable_at_concurrency.md``.
    """
    last: httpx.Response | None = None
    for _attempt in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


def _poll_to_completion(client: httpx.Client, task_id: str) -> dict:
    final = poll_job(
        client,
        "",
        task_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


# ---------------------------------------------------------------------------
# Submit + task fixtures — one inference reused across assertions.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scaffold_hop_submit_response(
    client: httpx.Client, scaffold_hop_task_id: str
) -> httpx.Response:
    """Async scaffold_hop submit — smallest useful call (250mw + frag_max_n=10)."""
    return client.post(
        "/api/tasks/scaffold_hop",
        data={
            "input_smiles": LOSARTAN,
            "frag_max_n": "10",
            "tanimoto_threshold": "0.5",
            "database": "250mw",
        },
        headers=_async_headers(scaffold_hop_task_id),
    )


@pytest.fixture(scope="module")
def scaffold_hop_task(
    client: httpx.Client,
    scaffold_hop_task_id: str,
    scaffold_hop_submit_response: httpx.Response,
) -> dict:
    assert scaffold_hop_submit_response.status_code == 202, (
        f"async scaffold_hop submit returned "
        f"{scaffold_hop_submit_response.status_code}: "
        f"{scaffold_hop_submit_response.text!r}"
    )
    return _poll_to_completion(client, scaffold_hop_task_id)


# ===================================================================
# Section 1: Submit semantics + OpenAPI registration
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_scaffold_hop_returns_202(self, scaffold_hop_submit_response):
        assert scaffold_hop_submit_response.status_code == 202, (
            f"expected 202; got {scaffold_hop_submit_response.status_code} "
            f"body={scaffold_hop_submit_response.text!r}"
        )

    def test_task_endpoint_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200, (
            f"openapi.json fetch failed: {r.status_code} {r.text!r}"
        )
        spec = r.json()
        assert "/api/tasks/scaffold_hop" in spec.get("paths", {}), (
            "task endpoint missing from OpenAPI; "
            "settings.task_endpoints_enabled may be False"
        )


# ===================================================================
# Section 2: Completion + outputs.
# ===================================================================


@pytest.mark.fc
class TestAsyncScaffoldHop:
    def test_completed(self, scaffold_hop_task, scaffold_hop_task_id, client):
        task = scaffold_hop_task
        assert task["status"] == "completed"
        assert task["job_id"] == scaffold_hop_task_id
        assert task.get("started_at") is not None
        assert task.get("completed_at") is not None
        d = task.get("duration_seconds")
        # ChemBounce fragmentation + fingerprint search on losartan even
        # with frag_max_n=10 takes at least ~30s cold; anything under 5s
        # means the subprocess likely exited early on a config error.
        assert d is not None and d > 5, (
            f"duration {d}s too short — subprocess may have short-circuited"
        )
        assert task.get("output_count", 0) > 0
        assert task.get("output_total_bytes", 0) > 0

    def test_overall_result_present(
        self, client, scaffold_hop_task_id, scaffold_hop_task
    ):
        r = _get_with_retry(client, f"/api/jobs/{scaffold_hop_task_id}/files")
        assert r.status_code == 200
        files = r.json()["files"]
        assert any("overall_result.txt" in f for f in files), (
            f"overall_result.txt missing from outputs: {files}"
        )

    def test_input_params_echoed(self, scaffold_hop_task):
        params = scaffold_hop_task.get("input_params") or {}
        assert params.get("input_smiles") == LOSARTAN
        assert params.get("frag_max_n") == 10
        assert params.get("tanimoto_threshold") == 0.5
        assert params.get("database") == "250mw"

    def test_overall_result_downloadable(
        self, client, scaffold_hop_task_id, scaffold_hop_task
    ):
        files = _get_with_retry(
            client, f"/api/jobs/{scaffold_hop_task_id}/files"
        ).json()["files"]
        target = next(f for f in files if "overall_result.txt" in f)
        r = _get_with_retry(
            client, f"/api/jobs/{scaffold_hop_task_id}/file/{target}"
        )
        assert r.status_code == 200
        # Upstream writes at least a TSV header row.
        text = r.content.decode("utf-8", errors="replace")
        assert len(text) > 0, "overall_result.txt is empty"
        # First column of upstream's TSV is 'Fragment_no'.
        assert "Fragment_no" in text.splitlines()[0], (
            f"overall_result.txt header unexpected: {text.splitlines()[:2]!r}"
        )


# ===================================================================
# Section 3: Identity — X-Bioagent-Job-Id propagates to JobInfo.job_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncTaskIdentity:
    def test_job_id_matches_task_id(
        self, scaffold_hop_task, scaffold_hop_task_id
    ):
        assert scaffold_hop_task["job_id"] == scaffold_hop_task_id, (
            "task endpoint must use X-Bioagent-Job-Id as JobInfo.job_id"
        )


# ===================================================================
# Section 4: Job lifecycle endpoints on the single shared task.
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(
        self, client, scaffold_hop_task_id, scaffold_hop_task
    ):
        r = _get_with_retry(client, f"/api/jobs/{scaffold_hop_task_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == scaffold_hop_task_id
        assert body["status"] == "completed"

    def test_log_endpoint(
        self, client, scaffold_hop_task_id, scaffold_hop_task
    ):
        r = _get_with_retry(client, f"/api/jobs/{scaffold_hop_task_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0, "log should be non-empty for a completed job"

    def test_download_zip(
        self, client, scaffold_hop_task_id, scaffold_hop_task
    ):
        r = _get_with_retry(
            client, f"/api/jobs/{scaffold_hop_task_id}/download"
        )
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any("overall_result.txt" in n for n in names), (
            f"overall_result.txt missing from zip: {names}"
        )

    def test_single_file_download_missing_returns_404(
        self, client, scaffold_hop_task_id, scaffold_hop_task
    ):
        r = _get_with_retry(
            client, f"/api/jobs/{scaffold_hop_task_id}/file/nonexistent.xyz"
        )
        assert r.status_code == 404


# ===================================================================
# Section 5: Duplicate dedup — FC platform rejects repeat task_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmit same X-Fc-Async-Task-Id after completion.

    Per the FC async task mode contract (engineering/decisions/
    2026-06-17-fc-async-task-mode.md and project memory
    ``project_fc_async_dedup_at_platform_layer.md``), FC dedups by
    ``X-Fc-Async-Task-Id`` at the platform layer — a duplicate returns 409
    without invoking the function.  If FC forwards anyway, the framework
    layer (``execute_task``) returns the existing JobInfo without re-running.
    """

    def test_duplicate_does_not_rerun(
        self,
        client: httpx.Client,
        scaffold_hop_task_id: str,
        scaffold_hop_task: dict,
    ):
        first_created_at = scaffold_hop_task["created_at"]
        first_completed_at = scaffold_hop_task["completed_at"]
        first_frag_max_n = (scaffold_hop_task.get("input_params") or {}).get(
            "frag_max_n"
        )

        # Resubmit same task_id with a different frag_max_n.  Neither the
        # new body nor a fresh completed_at should stick.
        r2 = client.post(
            "/api/tasks/scaffold_hop",
            data={
                "input_smiles": LOSARTAN,
                "frag_max_n": "42",  # different from first run's 10
                "tanimoto_threshold": "0.5",
                "database": "250mw",
            },
            headers=_async_headers(scaffold_hop_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → framework "
            f"dedups); got {r2.status_code} body={r2.text!r}"
        )

        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(
            client, f"/api/jobs/{scaffold_hop_task_id}"
        ).json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("frag_max_n") == (
            first_frag_max_n
        ), "duplicate async submit must not overwrite input_params"

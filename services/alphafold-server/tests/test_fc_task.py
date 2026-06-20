"""FC async task mode tests for alphafold-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/alphafold-server/tests/test_fc_task.py -v

Validates the `/api/tasks/fold` endpoint end-to-end against the deployed FC
function in async task mode (`X-Fc-Invocation-Type: Async`).

Async task mode means the HTTP request to FC returns HTTP 202 immediately;
the server-side `execute_task` blocks the function instance for the entire
pipeline lifetime so FC won't recycle mid-run.  The framework persists
JobInfo to NAS at each state transition, so we observe progress via
`GET /api/jobs/{task_id}`.

AlphaFold takes ~30-60 min per call (MSA reduced_dbs ~10-30 min + inference
~15-30 min), so we use a single short monomer_ptm fixture shared across
assertions.  The duplicate-dedup test reuses the same task_id, so it
piggybacks on the already-completed first run.
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

SERVICE = "alphafold-server"

# Smallest viable AlphaFold input: 76aa ubiquitin monomer_ptm + reduced_dbs.
# Multimer pipeline would double the cost; we only need to prove the task
# endpoint and async mode wiring work end-to-end.
EXAMPLE_FASTA = """\
>test_ubiquitin
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
"""

# AlphaFold MSA + inference can take a long time inside FC; give it 90 min.
POLL_TIMEOUT_S = 5400
POLL_INTERVAL_S = 30

# httpx timeouts: short connect, long read for the 202 request (FC sometimes
# takes 10-20s to enqueue), long enough write for the FASTA multipart.
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
def task_id() -> str:
    """One task_id shared by all assertions in this module."""
    return f"fc-async-fold-{int(time.time())}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_submit(
    client: httpx.Client,
    endpoint: str,
    *,
    task_id: str,
    fasta: str = EXAMPLE_FASTA,
    **form_fields: str,
) -> httpx.Response:
    """POST to a task endpoint with FC async task headers."""
    files = {"input_fasta": ("test.fasta", fasta.encode(), "text/plain")}
    return client.post(
        endpoint,
        data=form_fields,
        files=files,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )


def _get_with_retry(
    client: httpx.Client,
    path: str,
    *,
    max_attempts: int = 8,
    backoff_s: int = 15,
) -> httpx.Response:
    """GET that retries on FC's HTTP-gateway 429 throttling.

    After a long-running async task, the FC HTTP gateway can rate-limit
    subsequent GETs to `/api/jobs/...`.  This is a platform-layer
    artifact, not an alphafold-server bug — see project memory
    `project_fc_http_polling_unreliable_at_concurrency.md`.  Production
    code should use FCDispatcher.get_status (the FC GetAsyncTask API);
    here we just retry the HTTP call so the test still observes the
    underlying state.
    """
    last: httpx.Response | None = None
    for attempt in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


# ---------------------------------------------------------------------------
# Module-scoped run: submit once via async, poll JobInfo to completion.
# Other tests reuse the resulting JobInfo via the `fold_task` fixture.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def submit_response(client: httpx.Client, task_id: str) -> httpx.Response:
    """The raw HTTP response from the FC async invocation.

    Cached so multiple assertions (status_code, body shape) can inspect it
    without re-submitting.
    """
    return _async_submit(
        client,
        "/api/tasks/fold",
        task_id=task_id,
        model_preset="monomer_ptm",
        db_preset="reduced_dbs",
        models_to_relax="best",
    )


@pytest.fixture(scope="module")
def fold_task(client: httpx.Client, base_url: str, task_id: str, submit_response: httpx.Response) -> dict:
    """Final JobInfo after async submit + poll to terminal status."""
    assert submit_response.status_code == 202, (
        f"async submit returned {submit_response.status_code}: "
        f"{submit_response.text!r}.  Async task mode must be enabled in the "
        f"FC console for this function."
    )
    final = poll_job(
        client,
        "",  # base_url is already baked into the httpx client
        task_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


# ===================================================================
# Section 1: Submit semantics — proves FC async task mode is wired up.
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_returns_202(self, submit_response):
        """FC's async invocation must return HTTP 202 Accepted.

        A 200 here means the server treated it as a synchronous call
        (FC async task mode not enabled in the console).  A 409 means
        FC has a stale duplicate task pinned to this task_id.
        """
        assert submit_response.status_code == 202, (
            f"expected 202; got {submit_response.status_code} "
            f"body={submit_response.text!r}"
        )

    def test_task_endpoint_registered_in_openapi(self, client):
        spec = client.get("/openapi.json").json()
        assert "/api/tasks/fold" in spec["paths"], (
            "task endpoint missing from OpenAPI; settings.task_endpoints_enabled "
            "may be False on the deployed function"
        )


# ===================================================================
# Section 2: Job lifecycle — task ran inside the FC instance to completion.
# ===================================================================


@pytest.mark.fc
class TestAsyncRunsToCompletion:
    def test_completed_status(self, fold_task):
        assert fold_task["status"] == "completed"

    def test_started_and_completed_timestamps(self, fold_task):
        assert fold_task.get("started_at") is not None
        assert fold_task.get("completed_at") is not None

    def test_duration_recorded(self, fold_task):
        d = fold_task.get("duration_seconds")
        assert d is not None and d > 0
        # AlphaFold monomer_ptm + reduced_dbs is at least a few minutes;
        # if it returned in <30s the task didn't actually run.
        assert d > 30, f"duration {d}s too short for real AlphaFold work"

    def test_outputs_present(self, fold_task):
        assert fold_task.get("output_count", 0) > 0
        assert fold_task.get("output_total_bytes", 0) > 0

    def test_input_params_echoed(self, fold_task):
        params = fold_task.get("input_params") or {}
        assert params.get("model_preset") == "monomer_ptm"
        assert params.get("db_preset") == "reduced_dbs"


# ===================================================================
# Section 3: Identity — X-Bioagent-Job-Id propagates to JobInfo.job_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncTaskIdentity:
    def test_job_id_matches_task_id(self, fold_task, task_id):
        assert fold_task["job_id"] == task_id, (
            "task endpoint must use X-Bioagent-Job-Id as the JobInfo.job_id"
        )

    def test_job_visible_via_status_endpoint(self, client, task_id, fold_task):
        # fold_task fixture already polled to completion; confirm a fresh
        # GET sees the same record (rules out polling-only artifacts).
        r = _get_with_retry(client, f"/api/jobs/{task_id}")
        assert r.status_code == 200, f"status GET failed: {r.status_code} {r.text!r}"
        body = r.json()
        assert body["status"] == "completed"
        assert body["job_id"] == task_id


# ===================================================================
# Section 4: Outputs — the subprocess actually produced ranked PDB(s).
# ===================================================================


@pytest.mark.fc
class TestAsyncOutputs:
    def test_files_listing_includes_ranked_pdb(self, client, task_id, fold_task):
        r = _get_with_retry(client, f"/api/jobs/{task_id}/files")
        assert r.status_code == 200, f"files GET failed: {r.status_code} {r.text!r}"
        files = r.json()["files"]
        assert any("ranked_0.pdb" in n for n in files), (
            f"ranked_0.pdb missing from outputs: {files}"
        )

    def test_ranked_pdb_downloadable(self, client, task_id, fold_task):
        r = _get_with_retry(client, f"/api/jobs/{task_id}/files")
        assert r.status_code == 200, f"files GET failed: {r.status_code} {r.text!r}"
        files = r.json()["files"]
        ranked = [f for f in files if "ranked_0.pdb" in f]
        assert ranked, f"ranked_0.pdb missing from outputs: {files}"
        download = _get_with_retry(client, f"/api/jobs/{task_id}/file/{ranked[0]}")
        assert download.status_code == 200
        text = download.content.decode("utf-8", errors="replace")
        assert "ATOM" in text, "PDB should contain ATOM records"

    def test_download_zip(self, client, task_id, fold_task):
        r = _get_with_retry(client, f"/api/jobs/{task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any("ranked_0.pdb" in n for n in names), (
            f"ranked_0.pdb missing from zip: {names}"
        )


# ===================================================================
# Section 5: Duplicate dedup — FC platform layer rejects repeat task_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmitting the same X-Fc-Async-Task-Id after completion.

    Per the FC async task mode contract (engineering/decisions/
    2026-06-17-fc-async-task-mode.md and project memory
    `project_fc_async_dedup_at_platform_layer.md`), FC dedups by
    X-Fc-Async-Task-Id at the platform layer — a duplicate returns 409
    without ever invoking the function.

    If FC's behavior ever changes to forward the duplicate, the framework
    layer (`execute_task`) checks the JobInfo store and returns the existing
    record without re-running.  Either path is acceptable; what must NOT
    happen is a second subprocess run that overwrites the first's outputs.
    """

    def test_duplicate_does_not_rerun(
        self, client: httpx.Client, task_id: str, fold_task: dict
    ):
        # Capture the first run's invariants.
        first_created_at = fold_task["created_at"]
        first_completed_at = fold_task["completed_at"]
        first_model_preset = (fold_task.get("input_params") or {}).get("model_preset")

        # Resubmit with the SAME task_id but a different model_preset to
        # prove the second body wasn't applied.
        r2 = _async_submit(
            client,
            "/api/tasks/fold",
            task_id=task_id,
            model_preset="monomer",  # different from first run's monomer_ptm
            db_preset="reduced_dbs",
            models_to_relax="none",
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → framework "
            f"dedups); got {r2.status_code} body={r2.text!r}"
        )

        # If FC forwarded the call, give the framework dedup check a moment.
        if r2.status_code == 202:
            time.sleep(30)

        re_query_resp = _get_with_retry(client, f"/api/jobs/{task_id}")
        assert re_query_resp.status_code == 200, (
            f"status GET failed: {re_query_resp.status_code} {re_query_resp.text!r}"
        )
        re_query = re_query_resp.json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("model_preset") == first_model_preset, (
            "duplicate async submit must not overwrite input_params with second body"
        )

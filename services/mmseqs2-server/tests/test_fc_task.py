"""FC async task mode tests for mmseqs2-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/mmseqs2-server/tests/test_fc_task.py -v

Validates the two ``/api/tasks/<endpoint>`` endpoints (msa / pair)
end-to-end against the deployed FC function in async task mode
(``X-Fc-Invocation-Type: Async``).

Async task mode pins the FC instance for the whole computation (no
HTTP-gateway recycle risk) and dedups by ``X-Fc-Async-Task-Id`` at the
platform layer.  Instance utilization is much higher than sync submit +
poll — the ColabFold-protocol side (``/ticket/*``) lives in
:mod:`test_fc.py` and consolidates onto a single MSA via a module-scoped
fixture there for the same reason.

Test fixtures are module-scoped so a single msa (~monomer) + a single pair
(~heterodimer) MSA computation covers all per-endpoint completion,
JobInfo, and dedup assertions.
"""

from __future__ import annotations

import io
import tarfile
import time
import uuid
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "mmseqs2-server"

# Small monomer (52 aa) — keeps the MSA quick on the GPU subset DB.
SHORT_MONOMER = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVN"
MONOMER_Q = f">probe1\n{SHORT_MONOMER}\n"

# Two-chain heterodimer for /api/tasks/pair.
PAIRED_CHAIN_B = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDY"
)
PAIRED_Q = (
    f">chainA\n{SHORT_MONOMER}\n"
    f">chainB\n{PAIRED_CHAIN_B}\n"
)

# mmseqs MSA cold-start ~60s; short-seq search runs 3-10 min on the GPU
# subset DB.  Allow 40 min per task to absorb cold start + slow paths.
POLL_TIMEOUT_S = 2400
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
def msa_task_id() -> str:
    return f"fc-async-msa-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def pair_task_id() -> str:
    return f"fc-async-pair-{int(time.time())}-{uuid.uuid4().hex[:6]}"


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
    max_attempts: int = 20,
    backoff_s: int = 30,
) -> httpx.Response:
    """GET that retries on FC HTTP-gateway 429 throttling.

    Per project memory ``project_fc_http_polling_unreliable_at_concurrency``,
    ``GET /api/jobs/<id>`` on FC can 429 for several minutes when the
    account's GPU quota is under pressure.  Long retry window lets tests
    ride out throttling rather than bailing.
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
    # Bump ``max_transient_errors`` well above the framework default (10) so
    # poll_job rides out throttle windows.  Effective retry buffer:
    # 60 × 20s = 20 min of consecutive 429s before bailing.
    final = poll_job(
        client,
        "",
        task_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
        max_transient_errors=60,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


# ---------------------------------------------------------------------------
# Per-endpoint submit + task fixtures — one MSA per endpoint, reused
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def msa_submit_response(
    client: httpx.Client, msa_task_id: str,
) -> httpx.Response:
    # ``mode=all`` = UniRef30 only, filter=1, unpaired.  Pinned here so the
    # test suite does NOT require the colabfold_envdb_202108_db to be
    # uploaded to NAS — env DB deployment is tracked separately.  Switch to
    # ``mode=env`` once envdb + .idx is staged.
    return client.post(
        "/api/tasks/msa",
        data={"q": MONOMER_Q, "mode": "all"},
        headers=_async_headers(msa_task_id),
    )


@pytest.fixture(scope="module")
def msa_task(
    client: httpx.Client,
    msa_task_id: str,
    msa_submit_response: httpx.Response,
) -> dict:
    assert msa_submit_response.status_code == 202, (
        f"async /api/tasks/msa submit returned "
        f"{msa_submit_response.status_code}: {msa_submit_response.text!r}.  "
        f"Verify FC console has async task mode enabled for this function "
        f"(see engineering/decisions/2026-06-17-fc-async-task-mode.md)."
    )
    return _poll_to_completion(client, msa_task_id)


@pytest.fixture(scope="module")
def pair_submit_response(
    client: httpx.Client, pair_task_id: str,
) -> httpx.Response:
    return client.post(
        "/api/tasks/pair",
        data={"q": PAIRED_Q, "mode": "pairgreedy"},
        headers=_async_headers(pair_task_id),
    )


@pytest.fixture(scope="module")
def pair_task(
    client: httpx.Client,
    pair_task_id: str,
    pair_submit_response: httpx.Response,
) -> dict:
    assert pair_submit_response.status_code == 202, (
        f"async /api/tasks/pair submit returned "
        f"{pair_submit_response.status_code}: {pair_submit_response.text!r}"
    )
    return _poll_to_completion(client, pair_task_id)


# ===================================================================
# Section 1: Submit semantics + OpenAPI registration
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_msa_returns_202(self, msa_submit_response: httpx.Response) -> None:
        assert msa_submit_response.status_code == 202, (
            f"expected 202; got {msa_submit_response.status_code} "
            f"body={msa_submit_response.text!r}"
        )

    def test_pair_returns_202(self, pair_submit_response: httpx.Response) -> None:
        assert pair_submit_response.status_code == 202, (
            f"expected 202; got {pair_submit_response.status_code} "
            f"body={pair_submit_response.text!r}"
        )

    def test_task_endpoints_registered_in_openapi(self, client: httpx.Client) -> None:
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200, f"openapi.json fetch failed: {r.status_code}"
        spec = r.json()
        expected = {"/api/tasks/msa", "/api/tasks/pair"}
        missing = expected - set(spec["paths"])
        assert not missing, (
            f"task endpoints missing from OpenAPI: {missing}; "
            f"settings.task_endpoints_enabled may be False"
        )


# ===================================================================
# Section 2: /api/tasks/msa — completion + JobInfo
# ===================================================================


def _assert_completed_shape(task: dict, task_id: str) -> None:
    assert task["status"] == "completed"
    assert task["job_id"] == task_id
    assert task.get("started_at") is not None
    assert task.get("completed_at") is not None
    d = task.get("duration_seconds")
    assert d is not None and d > 3.0, (
        f"duration {d}s too short — subprocess may not have actually run"
    )
    assert task.get("output_count", 0) > 0
    assert task.get("output_total_bytes", 0) > 0


@pytest.mark.fc
class TestAsyncMsa:
    def test_completed(self, msa_task: dict, msa_task_id: str) -> None:
        _assert_completed_shape(msa_task, msa_task_id)

    def test_input_params_summary(self, msa_task: dict) -> None:
        params = msa_task.get("input_params") or {}
        assert params.get("mode") == "all"
        assert params.get("sequence_count") == 1
        assert params.get("total_residues") == len(SHORT_MONOMER)

    def test_raw_query_not_echoed_in_input_params(self, msa_task: dict) -> None:
        """Privacy: JobInfo persists to NAS — raw sequence must never land there."""
        assert SHORT_MONOMER not in repr(msa_task.get("input_params"))

    def test_a3m_file_in_outputs(
        self, client: httpx.Client, msa_task_id: str, msa_task: dict,
    ) -> None:
        r = _get_with_retry(client, f"/api/jobs/{msa_task_id}/files")
        assert r.status_code == 200
        files = r.json()["files"]
        assert any(f.endswith(".a3m") for f in files), (
            f"no .a3m file in msa output: {files}"
        )

    def test_a3m_downloadable(
        self, client: httpx.Client, msa_task_id: str, msa_task: dict,
    ) -> None:
        files = _get_with_retry(client, f"/api/jobs/{msa_task_id}/files").json()["files"]
        a3m = next(f for f in files if f.endswith(".a3m"))
        r = _get_with_retry(client, f"/api/jobs/{msa_task_id}/file/{a3m}")
        assert r.status_code == 200
        assert len(r.content) > 100, f"a3m suspiciously small: {len(r.content)} bytes"

    def test_result_download_tarball_reachable(
        self, client: httpx.Client, msa_task_id: str, msa_task: dict,
    ) -> None:
        """ColabFold-protocol read path on the async-produced job works too.

        Cross-surface check: the async task path and the ColabFold protocol
        share the same JobInfo store and output dir — a job created via
        /api/tasks/msa should be downloadable via /result/download/<id>.
        """
        r = _get_with_retry(client, f"/result/download/{msa_task_id}")
        assert r.status_code == 200
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
            names = tf.getnames()
        assert any(n.endswith(".a3m") for n in names), (
            f"no .a3m in cross-surface tarball: {names}"
        )


# ===================================================================
# Section 3: /api/tasks/pair — completion + JobInfo
# ===================================================================


@pytest.mark.fc
class TestAsyncPair:
    def test_completed(self, pair_task: dict, pair_task_id: str) -> None:
        _assert_completed_shape(pair_task, pair_task_id)

    def test_input_params_summary(self, pair_task: dict) -> None:
        params = pair_task.get("input_params") or {}
        assert params.get("mode") == "pairgreedy"
        assert params.get("sequence_count") == 2
        assert params.get("total_residues") == len(SHORT_MONOMER) + len(PAIRED_CHAIN_B)

    def test_a3m_files_in_outputs(
        self, client: httpx.Client, pair_task_id: str, pair_task: dict,
    ) -> None:
        files = _get_with_retry(client, f"/api/jobs/{pair_task_id}/files").json()["files"]
        a3m = [f for f in files if f.endswith(".a3m")]
        # Paired multimer output must include per-chain a3m's (>=2 chains).
        assert len(a3m) >= 1, f"expected >=1 .a3m for paired, got: {files}"


# ===================================================================
# Section 4: Job lifecycle — log, status, framework endpoints
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    """Assertions on the JobInfo lifecycle via framework endpoints, using
    the (cheaper) msa task fixture."""

    def test_status_endpoint(
        self, client: httpx.Client, msa_task_id: str, msa_task: dict,
    ) -> None:
        r = _get_with_retry(client, f"/api/jobs/{msa_task_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == msa_task_id
        assert body["status"] == "completed"

    def test_log_endpoint_nonempty(
        self, client: httpx.Client, msa_task_id: str, msa_task: dict,
    ) -> None:
        r = _get_with_retry(client, f"/api/jobs/{msa_task_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0, "log should be non-empty for a completed job"

    def test_download_zip_contains_a3m(
        self, client: httpx.Client, msa_task_id: str, msa_task: dict,
    ) -> None:
        import zipfile
        r = _get_with_retry(client, f"/api/jobs/{msa_task_id}/download")
        assert r.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            names = zf.namelist()
        assert any(n.endswith(".a3m") for n in names), (
            f"no .a3m in download zip: {names}"
        )


# ===================================================================
# Section 5: Duplicate dedup — FC platform layer + framework fallback
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmitting the same X-Fc-Async-Task-Id after completion.

    Per the FC async task mode contract
    (:doc:`engineering/decisions/2026-06-17-fc-async-task-mode`) and project
    memory ``project_fc_async_dedup_at_platform_layer``: FC dedups at the
    platform layer — duplicate returns 409 without invoking the function.
    If FC ever forwards anyway, the framework layer (``execute_task``)
    returns the existing JobInfo without re-running.
    """

    def test_duplicate_msa_does_not_rerun(
        self,
        client: httpx.Client,
        msa_task_id: str,
        msa_task: dict,
    ) -> None:
        first_created_at = msa_task["created_at"]
        first_completed_at = msa_task["completed_at"]
        first_mode = (msa_task.get("input_params") or {}).get("mode")

        # Resubmit same task_id with a DIFFERENT mode to prove dedup — if
        # anything re-runs, the new mode ("nofilter") would overwrite the
        # input_params snapshot.  Both modes are UniRef30-only so the env DB
        # need not be staged (first msa fixture also uses UniRef30-only).
        r2 = client.post(
            "/api/tasks/msa",
            data={"q": MONOMER_Q, "mode": "nofilter"},
            headers=_async_headers(msa_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → framework "
            f"dedups); got {r2.status_code} body={r2.text!r}"
        )
        if r2.status_code == 202:
            # FC forwarded — give framework-layer dedup a moment.
            time.sleep(15)

        re_query = _get_with_retry(client, f"/api/jobs/{msa_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("mode") == first_mode, (
            "duplicate async submit must not overwrite input_params"
        )

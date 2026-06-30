"""FC async task mode tests for deeprank-ab-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/deeprank-ab-server/tests/test_fc_task.py -v

Validates the ``/api/tasks/score`` endpoint end-to-end against the deployed
FC function in async task mode (``X-Fc-Invocation-Type: Async``).

Async task mode keeps the function instance alive for the whole pipeline
(no 30s HTTP-gateway recycle risk) and dedups by ``X-Fc-Async-Task-Id`` at
the FC platform layer.  Compared to the submit/poll path in ``test_fc.py``,
this file uses a SINGLE shared task per fixture (antibody + nanobody) — so
the test session spawns at most two function instances even when every test
class runs.  See project memory
``project_fc_http_polling_unreliable_at_concurrency.md`` for why this matters.

PDB source — sync bootstrap, then ``file://``
---------------------------------------------
FC's async invocation gateway caps the inbound event payload at 128 KiB
(``EntityTooLarge`` 400 otherwise), so we cannot ``files={"input_pdb": ...}``
the 313 KB ``tests/data/test.pdb`` on the async path.  But the SYNC HTTP
path has no such cap and writes the upload to
``/data/deeprank_ab_jobs/<job_id>/input/input.pdb`` on the NAS mount as a
side effect of ``framework.JobRunner.submit``.  So a session-scoped
``staged_pdb_uri`` fixture does one sync POST to ``/api/score`` at the
start of the session and returns ``file://<that NAS path>``; every async
submit then passes ``input_pdb_uri=<staged_pdb_uri>`` as a form field.
Net cost: 1 extra inference (the bootstrap sync call) on top of the 2
async inferences, comparable to ``alphafold-server`` / ``promera-server``.

Override the staged URI with ``DEEPRANK_AB_TEST_PDB_NAS_PATH`` if you've
pre-staged it elsewhere on NAS and want to skip the bootstrap call.
"""

from __future__ import annotations

import csv
import io
import os
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "deeprank-ab-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_PDB = DATA_DIR / "test.pdb"

# Optional pre-staged NAS path.  If set, the bootstrap sync upload is
# skipped and async submits reference this path directly.
PRESTAGED_PDB_NAS_PATH = os.environ.get("DEEPRANK_AB_TEST_PDB_NAS_PATH")

# JobsBaseDir on the FC instance — must match settings.jobs_base_dir
# (see Dockerfile ``DEEPRANK_AB_JOBS_BASE_DIR``).  We assemble
# ``file://<jobs_base_dir>/<bootstrap_job_id>/input/input.pdb`` after the
# sync bootstrap so async submits can read the PDB from NAS without
# multipart-uploading it again.
JOBS_BASE_DIR_ON_FC = "/data/deeprank_ab_jobs"

# DeepRank-Ab on the example complex is fast (~3-10 min including ESM-2
# embedding + ANARCI + EGNN), but the FC instance can spend a minute on
# cold-start weight-load.  Give it 30 min total.
POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 15

# httpx timeouts: short connect, long read for the 202 enqueue (FC may take
# 10-20s), enough write for the multipart PDB upload (~MB scale).
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
def staged_pdb_uri(client: httpx.Client) -> str:
    """One-time sync upload that lands the PDB on the FC NAS.

    Returns the ``file://`` URI the async submits should pass in
    ``input_pdb_uri``.  If ``DEEPRANK_AB_TEST_PDB_NAS_PATH`` is set we
    trust it points at a pre-staged file and skip the upload.

    The sync POST does run a real inference in the background — but the
    PDB is written to ``input/input.pdb`` synchronously *before*
    ``submit`` returns, so async tests can reference it immediately.
    """
    if PRESTAGED_PDB_NAS_PATH:
        return f"file://{PRESTAGED_PDB_NAS_PATH}"

    with open(TEST_PDB, "rb") as fh:
        r = client.post(
            "/api/score",
            files={"input_pdb": (TEST_PDB.name, fh.read(), "chemical/x-pdb")},
            data={
                "heavy_chain_id": "H",
                "light_chain_id": "L",
                "antigen_chain_id": "A",
            },
        )
    assert r.status_code == 200, (
        f"bootstrap sync upload failed: {r.status_code} {r.text!r}"
    )
    job_id = r.json()["job_id"]
    # JobRunner.submit saves the upload synchronously before returning, so
    # the file already exists on NAS by the time we read the response.
    return f"file://{JOBS_BASE_DIR_ON_FC}/{job_id}/input/input.pdb"


@pytest.fixture(scope="module")
def task_id() -> str:
    """One task_id shared by every antibody (H/L/A) assertion in this module."""
    return f"fc-async-score-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def nanobody_task_id() -> str:
    """Distinct task_id for the nanobody fixture so FC dedup doesn't collide."""
    return f"fc-async-score-nb-{int(time.time())}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_submit(
    client: httpx.Client,
    endpoint: str,
    *,
    task_id: str,
    pdb_uri: str,
    **form_fields: str,
) -> httpx.Response:
    """POST to a task endpoint with FC async task headers.

    The PDB is referenced by ``input_pdb_uri`` (NAS ``file://`` from the
    bootstrap sync upload) so the request body stays under FC's 128 KiB
    async-event cap.
    """
    return client.post(
        endpoint,
        data={"input_pdb_uri": pdb_uri, **form_fields},
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
    subsequent GETs to ``/api/jobs/...``.  This is a platform-layer artifact,
    not a deeprank-ab-server bug — see project memory
    ``project_fc_http_polling_unreliable_at_concurrency.md``.  Production
    code should use ``FCDispatcher.get_status`` (the FC GetAsyncTask API);
    here we just retry the HTTP call so the test still observes the
    underlying state.
    """
    last: httpx.Response | None = None
    for _attempt in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


# ---------------------------------------------------------------------------
# Module-scoped runs: submit once via async, poll JobInfo to completion.
# All assertion classes reuse the resulting JobInfo via the fixtures below.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def submit_response(
    client: httpx.Client, task_id: str, staged_pdb_uri: str
) -> httpx.Response:
    """Raw 202 from the antibody (H/L/A) async invocation."""
    return _async_submit(
        client,
        "/api/tasks/score",
        task_id=task_id,
        pdb_uri=staged_pdb_uri,
        heavy_chain_id="H",
        light_chain_id="L",
        antigen_chain_id="A",
    )


@pytest.fixture(scope="module")
def score_task(
    client: httpx.Client,
    base_url: str,
    task_id: str,
    submit_response: httpx.Response,
) -> dict:
    """Final JobInfo after antibody async submit + poll to terminal status."""
    assert submit_response.status_code == 202, (
        f"async submit returned {submit_response.status_code}: "
        f"{submit_response.text!r}.  Async task mode must be enabled in the "
        f"FC console for this function."
    )
    final = poll_job(
        client,
        "",  # base_url already baked into the httpx client
        task_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


@pytest.fixture(scope="module")
def nanobody_submit_response(
    client: httpx.Client, nanobody_task_id: str, staged_pdb_uri: str
) -> httpx.Response:
    """Raw 202 from the nanobody (light_chain='-') async invocation."""
    return _async_submit(
        client,
        "/api/tasks/score",
        task_id=nanobody_task_id,
        pdb_uri=staged_pdb_uri,
        heavy_chain_id="H",
        light_chain_id="-",
        antigen_chain_id="A",
    )


@pytest.fixture(scope="module")
def nanobody_score_task(
    client: httpx.Client,
    base_url: str,
    nanobody_task_id: str,
    nanobody_submit_response: httpx.Response,
) -> dict:
    """Final JobInfo after nanobody async submit + poll to terminal status."""
    assert nanobody_submit_response.status_code == 202, (
        f"async submit returned {nanobody_submit_response.status_code}: "
        f"{nanobody_submit_response.text!r}"
    )
    final = poll_job(
        client,
        "",
        nanobody_task_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
    )
    assert final["status"] == "completed", f"nanobody task did not complete: {final}"
    return final


# ===================================================================
# Section 1: Submit semantics — proves FC async task mode is wired up.
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_returns_202(self, submit_response):
        """FC's async invocation must return HTTP 202 Accepted.

        A 200 here means the server treated it as a synchronous call (FC
        async task mode not enabled in the console).  A 409 means FC has a
        stale duplicate task pinned to this task_id — re-run with a fresh
        session.
        """
        assert submit_response.status_code == 202, (
            f"expected 202; got {submit_response.status_code} "
            f"body={submit_response.text!r}"
        )

    def test_task_endpoint_registered_in_openapi(self, client):
        spec = client.get("/openapi.json").json()
        assert "/api/tasks/score" in spec["paths"], (
            "task endpoint missing from OpenAPI; settings.task_endpoints_enabled "
            "may be False on the deployed function"
        )


# ===================================================================
# Section 2: Job lifecycle — task ran inside the FC instance to completion.
# ===================================================================


@pytest.mark.fc
class TestAsyncRunsToCompletion:
    def test_completed_status(self, score_task):
        assert score_task["status"] == "completed"

    def test_started_and_completed_timestamps(self, score_task):
        assert score_task.get("started_at") is not None
        assert score_task.get("completed_at") is not None

    def test_duration_recorded(self, score_task):
        d = score_task.get("duration_seconds")
        assert d is not None and d > 0
        # ESM-2 embedding + EGNN inference always takes >10s on real input;
        # a sub-10s duration usually means the subprocess crashed early.
        assert d > 10, f"duration {d}s too short for real DeepRank-Ab work"

    def test_outputs_present(self, score_task):
        assert score_task.get("output_count", 0) > 0
        assert score_task.get("output_total_bytes", 0) > 0

    def test_input_params_echoed(self, score_task):
        params = score_task.get("input_params") or {}
        assert params.get("heavy_chain_id") == "H"
        assert params.get("light_chain_id") == "L"
        assert params.get("antigen_chain_id") == "A"


# ===================================================================
# Section 3: Identity — X-Bioagent-Job-Id propagates to JobInfo.job_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncTaskIdentity:
    def test_job_id_matches_task_id(self, score_task, task_id):
        assert score_task["job_id"] == task_id, (
            "task endpoint must use X-Bioagent-Job-Id as the JobInfo.job_id"
        )

    def test_job_visible_via_status_endpoint(self, client, task_id, score_task):
        # score_task fixture already polled to completion; confirm a fresh
        # GET sees the same record (rules out polling-only artifacts).
        r = _get_with_retry(client, f"/api/jobs/{task_id}")
        assert r.status_code == 200, f"status GET failed: {r.status_code} {r.text!r}"
        body = r.json()
        assert body["status"] == "completed"
        assert body["job_id"] == task_id


# ===================================================================
# Section 4: Outputs — the subprocess actually produced predictions CSV.
# ===================================================================


@pytest.mark.fc
class TestAsyncOutputs:
    def test_files_listing_includes_predictions_csv(self, client, task_id, score_task):
        r = _get_with_retry(client, f"/api/jobs/{task_id}/files")
        assert r.status_code == 200, f"files GET failed: {r.status_code} {r.text!r}"
        files = r.json()["files"]
        assert any(n.endswith("_predictions.csv") for n in files), (
            f"_predictions.csv missing from outputs: {files}"
        )

    def test_files_listing_includes_hdf5(self, client, task_id, score_task):
        r = _get_with_retry(client, f"/api/jobs/{task_id}/files")
        files = r.json()["files"]
        assert any(n.endswith(".hdf5") for n in files), (
            f"HDF5 files (graph / predictions) missing from outputs: {files}"
        )

    def test_predictions_csv_schema_and_values(self, client, task_id, score_task):
        """CSV must declare predicted_dockq + quality_flag, all rows in [0,1]."""
        r = _get_with_retry(client, f"/api/jobs/{task_id}/files")
        files = r.json()["files"]
        csv_files = [f for f in files if f.endswith("_predictions.csv")]
        assert csv_files, f"no predictions CSV: {files}"

        download = _get_with_retry(client, f"/api/jobs/{task_id}/file/{csv_files[0]}")
        assert download.status_code == 200
        reader = csv.DictReader(io.StringIO(download.text))
        rows = list(reader)

        assert rows, "predictions CSV should have at least one row"
        assert "predicted_dockq" in reader.fieldnames, (
            f"missing predicted_dockq column: {reader.fieldnames}"
        )
        assert "quality_flag" in reader.fieldnames, (
            f"missing quality_flag column: {reader.fieldnames}"
        )
        for row in rows:
            dockq = float(row["predicted_dockq"])
            assert 0.0 <= dockq <= 1.0, f"predicted_dockq={dockq} out of [0,1]"
            assert row["quality_flag"] in ("ok", "low_HL_contacts", "not_applicable"), (
                f"unexpected quality_flag: {row['quality_flag']!r}"
            )

    def test_download_zip(self, client, task_id, score_task):
        r = _get_with_retry(client, f"/api/jobs/{task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any(n.endswith("_predictions.csv") for n in names), (
            f"predictions CSV missing from zip: {names}"
        )


# ===================================================================
# Section 4b: Nanobody (VHH) pipeline — light_chain_id='-' end-to-end.
# ===================================================================


@pytest.mark.fc
class TestAsyncNanobody:
    """Async task mode with light_chain_id='-' (VHH / single-chain antibody).

    Exercises the nanobody code path: skip light-chain annotation, no H/L
    contact gate; quality_flag should be 'ok' or 'not_applicable'.
    """

    def test_submit_returns_202(self, nanobody_submit_response):
        assert nanobody_submit_response.status_code == 202, (
            f"expected 202; got {nanobody_submit_response.status_code} "
            f"body={nanobody_submit_response.text!r}"
        )

    def test_completed_status(self, nanobody_score_task):
        assert nanobody_score_task["status"] == "completed"

    def test_input_params_echoed(self, nanobody_score_task):
        params = nanobody_score_task.get("input_params") or {}
        assert params.get("heavy_chain_id") == "H"
        assert params.get("light_chain_id") == "-"
        assert params.get("antigen_chain_id") == "A"

    def test_job_id_matches_task_id(self, nanobody_score_task, nanobody_task_id):
        assert nanobody_score_task["job_id"] == nanobody_task_id

    def test_quality_flag_skips_low_hl_contacts(
        self, client, nanobody_task_id, nanobody_score_task
    ):
        """For nanobodies the H/L contact check must not flag 'low_HL_contacts'."""
        r = _get_with_retry(client, f"/api/jobs/{nanobody_task_id}/files")
        files = r.json()["files"]
        csv_files = [f for f in files if f.endswith("_predictions.csv")]
        assert csv_files, f"no predictions CSV in nanobody output: {files}"

        download = _get_with_retry(
            client, f"/api/jobs/{nanobody_task_id}/file/{csv_files[0]}"
        )
        assert download.status_code == 200
        reader = csv.DictReader(io.StringIO(download.text))
        for row in reader:
            assert row["quality_flag"] in ("ok", "not_applicable"), (
                f"nanobody quality_flag must be 'ok' or 'not_applicable', "
                f"got {row['quality_flag']!r}"
            )


# ===================================================================
# Section 5: Duplicate dedup — FC platform layer rejects repeat task_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmitting the same X-Fc-Async-Task-Id after completion.

    Per the FC async task mode contract (engineering/decisions/
    2026-06-17-fc-async-task-mode.md and project memory
    ``project_fc_async_dedup_at_platform_layer.md``), FC dedups by
    ``X-Fc-Async-Task-Id`` at the platform layer — a duplicate returns 409
    without ever invoking the function.

    If FC's behavior ever changes to forward the duplicate, the framework
    layer (``execute_task``) checks the JobInfo store and returns the existing
    record without re-running.  Either path is acceptable; what must NOT
    happen is a second subprocess run that overwrites the first's outputs.
    """

    def test_duplicate_does_not_rerun(
        self,
        client: httpx.Client,
        task_id: str,
        score_task: dict,
        staged_pdb_uri: str,
    ):
        first_created_at = score_task["created_at"]
        first_completed_at = score_task["completed_at"]
        first_heavy = (score_task.get("input_params") or {}).get("heavy_chain_id")

        # Resubmit with the SAME task_id but different chain IDs so we can
        # prove the second body wasn't applied.
        r2 = _async_submit(
            client,
            "/api/tasks/score",
            task_id=task_id,
            pdb_uri=staged_pdb_uri,
            heavy_chain_id="B",  # different from first run's "H"
            light_chain_id="L",
            antigen_chain_id="A",
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → framework "
            f"dedups); got {r2.status_code} body={r2.text!r}"
        )

        if r2.status_code == 202:
            time.sleep(15)

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
        assert (re_query.get("input_params") or {}).get("heavy_chain_id") == first_heavy, (
            "duplicate async submit must not overwrite input_params with second body"
        )

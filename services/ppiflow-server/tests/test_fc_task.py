"""FC async task mode tests for ppiflow-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/ppiflow-server/tests/test_fc_task.py -v

Validates the 4 ``/api/tasks/sample/<mode>`` endpoints (binder / antibody /
nanobody / monomer) end-to-end against the deployed FC function in async
task mode (``X-Fc-Invocation-Type: Async``).  Scaffolding is covered only at
the OpenAPI / dedup layer because its motif CSV references PDB files that
must be pre-staged on NAS.

Async task mode pins the FC instance for the whole pipeline (no 30s
HTTP-gateway recycle risk) and dedups by ``X-Fc-Async-Task-Id`` at the
platform layer.  Each inference has a SINGLE shared task fixture so the
test session spawns one instance per endpoint, not per assertion — total ~5
instances (4 async + 1 sync bootstrap).

PDB source — sync bootstrap, then ``file://``
---------------------------------------------
FC's async invocation gateway caps the inbound event payload at 128 KiB
(``EntityTooLarge`` 400 otherwise).  The antigen PDB ``1IJZ_IL13.pdb`` is
143 KB so multipart upload would always fail on the async path.  But the
SYNC HTTP path has no cap and writes the upload to
``/data/ppiflow_jobs/<job_id>/input/target.pdb`` on NAS as a side effect of
``framework.JobRunner.submit``.  The ``staged_antigen_uri`` fixture does
one sync POST to ``/api/sample/binder`` and returns ``file://<that path>``;
every subsequent async submit passes ``antigen_uri=<staged>``.  The two
framework PDBs (57 KB nanobody, 111 KB scFv) fit under the cap and are
sent as multipart on the async path.

Override the bootstrap with ``PPIFLOW_TEST_ANTIGEN_NAS_PATH=/data/...`` if
you have the antigen pre-staged elsewhere on NAS.
"""

from __future__ import annotations

import io
import os
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "ppiflow-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
ANTIGEN_PDB = DATA_DIR / "1IJZ_IL13.pdb"
SCFV_FRAMEWORK_PDB = DATA_DIR / "6nou_scfv_framework.pdb"
NANOBODY_FRAMEWORK_PDB = DATA_DIR / "7eow_nanobody_framework.pdb"

# Optional pre-staged NAS path for the antigen PDB.  If set, the bootstrap
# sync upload is skipped.
PRESTAGED_ANTIGEN_NAS_PATH = os.environ.get("PPIFLOW_TEST_ANTIGEN_NAS_PATH")

# Jobs base dir on the FC instance — must match Dockerfile's
# ``PPIFLOW_JOBS_BASE_DIR`` (settings default).
JOBS_BASE_DIR_ON_FC = "/data/ppiflow_jobs"

# PPIFlow inference is ~5-10 min per call (1 sample, tight CDR specs).
# Allow generous slack for cold start + NAS read of the 280 MB checkpoint.
POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 20

# httpx timeouts: short connect, long read for the 202 enqueue (FC can take
# 10-30s), long write for multipart frameworks (~100 KB).
TIMEOUT = httpx.Timeout(connect=30, read=120, write=120, pool=30)


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
def staged_antigen_uri(client: httpx.Client) -> str:
    """One-time sync upload that lands the antigen PDB on the FC NAS.

    Returns ``file://<NAS path>`` for use as ``antigen_uri`` in async
    submits.  If ``PPIFLOW_TEST_ANTIGEN_NAS_PATH`` is set we trust it points
    at a pre-staged file and skip the upload.

    The sync POST runs a real binder inference in the background — but the
    PDB is saved to ``input/target.pdb`` synchronously *before* ``submit``
    returns, so async tests can reference it immediately.
    """
    if PRESTAGED_ANTIGEN_NAS_PATH:
        return f"file://{PRESTAGED_ANTIGEN_NAS_PATH}"

    with open(ANTIGEN_PDB, "rb") as fh:
        r = client.post(
            "/api/sample/binder",
            files={"target": (ANTIGEN_PDB.name, fh.read(), "chemical/x-pdb")},
            data={
                "target_chain": "C",
                "binder_chain": "A",
                "specified_hotspots": "C11,C14,C15",
                "samples_min_length": "60",
                "samples_max_length": "70",
                "samples_per_target": "1",
                "name": "bootstrap_stage",
            },
        )
    assert r.status_code == 200, (
        f"bootstrap sync upload failed: {r.status_code} {r.text!r}"
    )
    job_id = r.json()["job_id"]
    # JobRunner.submit saves the upload synchronously before returning, so
    # the file already exists on NAS by the time we read the response.
    return f"file://{JOBS_BASE_DIR_ON_FC}/{job_id}/input/target.pdb"


# Task-id fixtures — one per endpoint so dedup tests can target a known
# completed task without colliding with other fixtures.
@pytest.fixture(scope="module")
def binder_task_id() -> str:
    return f"fc-async-binder-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def antibody_task_id() -> str:
    return f"fc-async-antibody-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def nanobody_task_id() -> str:
    return f"fc-async-nanobody-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def monomer_task_id() -> str:
    return f"fc-async-monomer-{int(time.time())}-{uuid.uuid4().hex[:6]}"


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
    max_attempts: int = 8,
    backoff_s: int = 15,
) -> httpx.Response:
    """GET that retries on FC HTTP-gateway 429 throttling.

    Long-running async tasks can trip FC's request-rate limiter on the
    poll path — a platform artifact, not a ppiflow-server bug.  See
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
# Per-endpoint submit fixtures (module-scoped — one inference per endpoint)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binder_submit_response(
    client: httpx.Client, binder_task_id: str, staged_antigen_uri: str
) -> httpx.Response:
    """Async binder submit: antigen via file://, smallest viable config."""
    return client.post(
        "/api/tasks/sample/binder",
        data={
            "target_uri": staged_antigen_uri,
            "target_chain": "C",
            "binder_chain": "A",
            "specified_hotspots": "C11,C14,C15",
            "samples_min_length": "60",
            "samples_max_length": "70",
            "samples_per_target": "1",
            "name": "async_binder",
        },
        headers=_async_headers(binder_task_id),
    )


@pytest.fixture(scope="module")
def binder_task(
    client: httpx.Client,
    binder_task_id: str,
    binder_submit_response: httpx.Response,
) -> dict:
    assert binder_submit_response.status_code == 202, (
        f"async binder submit returned {binder_submit_response.status_code}: "
        f"{binder_submit_response.text!r}"
    )
    return _poll_to_completion(client, binder_task_id)


@pytest.fixture(scope="module")
def antibody_submit_response(
    client: httpx.Client, antibody_task_id: str, staged_antigen_uri: str
) -> httpx.Response:
    """Async antibody submit: antigen via file://, scFv framework multipart (~111 KB)."""
    with open(SCFV_FRAMEWORK_PDB, "rb") as fw:
        return client.post(
            "/api/tasks/sample/antibody",
            data={
                "antigen_uri": staged_antigen_uri,
                "antigen_chain": "C",
                "heavy_chain": "A",
                "light_chain": "B",
                "specified_hotspots": "C11,C14,C15",
                "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,10-10,CDRL1,7-7,CDRL2,3-3,CDRL3,9-9",
                "samples_per_target": "1",
                "name": "async_antibody",
            },
            files={
                "framework": (SCFV_FRAMEWORK_PDB.name, fw.read(), "chemical/x-pdb"),
            },
            headers=_async_headers(antibody_task_id),
        )


@pytest.fixture(scope="module")
def antibody_task(
    client: httpx.Client,
    antibody_task_id: str,
    antibody_submit_response: httpx.Response,
) -> dict:
    assert antibody_submit_response.status_code == 202, (
        f"async antibody submit returned {antibody_submit_response.status_code}: "
        f"{antibody_submit_response.text!r}"
    )
    return _poll_to_completion(client, antibody_task_id)


@pytest.fixture(scope="module")
def nanobody_submit_response(
    client: httpx.Client, nanobody_task_id: str, staged_antigen_uri: str
) -> httpx.Response:
    """Async nanobody submit: antigen via file://, VHH framework multipart (~57 KB)."""
    with open(NANOBODY_FRAMEWORK_PDB, "rb") as fw:
        return client.post(
            "/api/tasks/sample/nanobody",
            data={
                "antigen_uri": staged_antigen_uri,
                "antigen_chain": "C",
                "heavy_chain": "A",
                "specified_hotspots": "C11,C14,C15",
                "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,10-10",
                "samples_per_target": "1",
                "name": "async_nanobody",
            },
            files={
                "framework": (NANOBODY_FRAMEWORK_PDB.name, fw.read(), "chemical/x-pdb"),
            },
            headers=_async_headers(nanobody_task_id),
        )


@pytest.fixture(scope="module")
def nanobody_task(
    client: httpx.Client,
    nanobody_task_id: str,
    nanobody_submit_response: httpx.Response,
) -> dict:
    assert nanobody_submit_response.status_code == 202, (
        f"async nanobody submit returned {nanobody_submit_response.status_code}: "
        f"{nanobody_submit_response.text!r}"
    )
    return _poll_to_completion(client, nanobody_task_id)


@pytest.fixture(scope="module")
def monomer_submit_response(
    client: httpx.Client, monomer_task_id: str
) -> httpx.Response:
    """Async monomer submit: no uploads, tiny form-only payload."""
    return client.post(
        "/api/tasks/sample/monomer",
        data={
            "length_subset": "[40]",
            "samples_per_target": "1",
            "name": "async_monomer",
        },
        headers=_async_headers(monomer_task_id),
    )


@pytest.fixture(scope="module")
def monomer_task(
    client: httpx.Client,
    monomer_task_id: str,
    monomer_submit_response: httpx.Response,
) -> dict:
    assert monomer_submit_response.status_code == 202, (
        f"async monomer submit returned {monomer_submit_response.status_code}: "
        f"{monomer_submit_response.text!r}"
    )
    return _poll_to_completion(client, monomer_task_id)


# ===================================================================
# Section 1: Submit semantics + OpenAPI registration
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_binder_returns_202(self, binder_submit_response):
        assert binder_submit_response.status_code == 202, (
            f"expected 202; got {binder_submit_response.status_code} "
            f"body={binder_submit_response.text!r}"
        )

    def test_antibody_returns_202(self, antibody_submit_response):
        assert antibody_submit_response.status_code == 202, (
            f"expected 202; got {antibody_submit_response.status_code} "
            f"body={antibody_submit_response.text!r}"
        )

    def test_nanobody_returns_202(self, nanobody_submit_response):
        assert nanobody_submit_response.status_code == 202, (
            f"expected 202; got {nanobody_submit_response.status_code} "
            f"body={nanobody_submit_response.text!r}"
        )

    def test_monomer_returns_202(self, monomer_submit_response):
        assert monomer_submit_response.status_code == 202, (
            f"expected 202; got {monomer_submit_response.status_code} "
            f"body={monomer_submit_response.text!r}"
        )

    def test_task_endpoints_registered_in_openapi(self, client):
        spec = client.get("/openapi.json").json()
        expected = {
            "/api/tasks/sample/binder",
            "/api/tasks/sample/antibody",
            "/api/tasks/sample/nanobody",
            "/api/tasks/sample/monomer",
            "/api/tasks/sample/scaffolding",
        }
        missing = expected - set(spec["paths"])
        assert not missing, (
            f"task endpoints missing from OpenAPI: {missing}; "
            f"settings.task_endpoints_enabled may be False"
        )


# ===================================================================
# Section 2: Per-endpoint completion + outputs
# ===================================================================


def _assert_completed_with_pdb_outputs(task: dict, task_id: str, client: httpx.Client):
    assert task["status"] == "completed"
    assert task["job_id"] == task_id
    assert task.get("started_at") is not None
    assert task.get("completed_at") is not None
    d = task.get("duration_seconds")
    assert d is not None and d > 30, (
        f"duration {d}s too short for real PPIFlow work"
    )
    assert task.get("output_count", 0) > 0
    assert task.get("output_total_bytes", 0) > 0

    r = _get_with_retry(client, f"/api/jobs/{task_id}/files")
    assert r.status_code == 200
    files = r.json()["files"]
    assert any(f.endswith(".pdb") for f in files), (
        f"no .pdb in outputs: {files}"
    )


@pytest.mark.fc
class TestAsyncBinder:
    def test_completed(self, binder_task, binder_task_id, client):
        _assert_completed_with_pdb_outputs(binder_task, binder_task_id, client)

    def test_input_params_echoed(self, binder_task):
        params = binder_task.get("input_params") or {}
        assert params.get("name") == "async_binder"
        assert params.get("samples_per_target") == 1

    def test_output_pdb_uses_name_prefix(self, client, binder_task_id, binder_task):
        """PPIFlow's binder writes ``output/<name>_<idx>.pdb`` (flat, prefixed).

        NOTE: the adapter's ``tool_outputs`` docstring still says
        ``output/<name>/*.pdb`` (nested subdir) — that's correct for
        antibody/nanobody/monomer/scaffolding but stale for binder.  This
        assertion documents the real binder layout.
        """
        files = _get_with_retry(client, f"/api/jobs/{binder_task_id}/files").json()["files"]
        pdbs = [f for f in files if f.endswith(".pdb")]
        assert pdbs, f"no .pdb outputs: {files}"
        assert any(f.startswith("async_binder") for f in pdbs), (
            f"binder PDB outputs should be prefixed with the request name: {pdbs}"
        )


@pytest.mark.fc
class TestAsyncAntibody:
    def test_completed(self, antibody_task, antibody_task_id, client):
        _assert_completed_with_pdb_outputs(antibody_task, antibody_task_id, client)

    def test_input_params_echoed(self, antibody_task):
        params = antibody_task.get("input_params") or {}
        assert params.get("name") == "async_antibody"
        assert "CDRL1" in params.get("cdr_length", ""), (
            "antibody must include light-chain CDRs in cdr_length"
        )


@pytest.mark.fc
class TestAsyncNanobody:
    def test_completed(self, nanobody_task, nanobody_task_id, client):
        _assert_completed_with_pdb_outputs(nanobody_task, nanobody_task_id, client)

    def test_input_params_echoed(self, nanobody_task):
        params = nanobody_task.get("input_params") or {}
        assert params.get("name") == "async_nanobody"
        # Nanobody spec must NOT include CDRL* (heavy-only).
        assert "CDRL" not in params.get("cdr_length", ""), (
            "nanobody cdr_length should not include light-chain CDRs"
        )


@pytest.mark.fc
class TestAsyncMonomer:
    def test_completed(self, monomer_task, monomer_task_id, client):
        _assert_completed_with_pdb_outputs(monomer_task, monomer_task_id, client)

    def test_input_params_echoed(self, monomer_task):
        params = monomer_task.get("input_params") or {}
        assert params.get("name") == "async_monomer"
        assert params.get("length_subset") == [40]


# ===================================================================
# Section 3: Job lifecycle on the binder task (single shared inference)
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_job_visible_via_status_endpoint(self, client, binder_task_id, binder_task):
        r = _get_with_retry(client, f"/api/jobs/{binder_task_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == binder_task_id
        assert body["status"] == "completed"

    def test_job_log_endpoint(self, client, binder_task_id, binder_task):
        r = _get_with_retry(client, f"/api/jobs/{binder_task_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0, "log should be non-empty for a completed job"

    def test_job_download_zip(self, client, binder_task_id, binder_task):
        r = _get_with_retry(client, f"/api/jobs/{binder_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any(n.endswith(".pdb") for n in names), (
            f"PDB outputs missing from zip: {names}"
        )

    def test_single_file_download_pdb(self, client, binder_task_id, binder_task):
        files = _get_with_retry(
            client, f"/api/jobs/{binder_task_id}/files"
        ).json()["files"]
        pdb_path = next(f for f in files if f.endswith(".pdb"))
        r = _get_with_retry(client, f"/api/jobs/{binder_task_id}/file/{pdb_path}")
        assert r.status_code == 200
        text = r.content.decode("utf-8", errors="replace")
        assert "ATOM" in text, "PDB should contain ATOM records"


# ===================================================================
# Section 4: Duplicate dedup — FC platform layer rejects repeat task_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmitting the same X-Fc-Async-Task-Id after completion.

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
        monomer_task_id: str,
        monomer_task: dict,
    ):
        first_created_at = monomer_task["created_at"]
        first_completed_at = monomer_task["completed_at"]
        first_name = (monomer_task.get("input_params") or {}).get("name")

        # Resubmit same task_id with a different name to prove dedup.
        r2 = client.post(
            "/api/tasks/sample/monomer",
            data={
                "length_subset": "[50]",  # different
                "samples_per_target": "1",
                "name": "should_not_apply",
            },
            headers=_async_headers(monomer_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → framework "
            f"dedups); got {r2.status_code} body={r2.text!r}"
        )

        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(client, f"/api/jobs/{monomer_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("name") == first_name, (
            "duplicate async submit must not overwrite input_params"
        )

"""FC async task mode tests for rfantibody-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/rfantibody-server/tests/test_fc_task.py -v

Validates the 3 ``/api/tasks/<step>`` endpoints (rfdiffusion / proteinmpnn / rf2)
end-to-end against the deployed FC function in async task mode
(``X-Fc-Invocation-Type: Async``).

Async task mode pins the FC instance for the whole pipeline (no 30s
HTTP-gateway recycle risk) and dedups by ``X-Fc-Async-Task-Id`` at the
platform layer.

PDB source — sync bootstrap, then ``file://`` / ``job://``
----------------------------------------------------------
FC's async invocation gateway caps the inbound event payload at 128 KiB
(``EntityTooLarge`` 400 otherwise).  The two test PDBs are both too big to
multipart-upload on the async path:

  * ``rsv_site3.pdb`` (target)   — 461 KB
  * ``hu-4D5-8_Fv.pdb`` (frame)  — 131 KB

So we use a sync-bootstrap pattern: one sync POST to ``/api/rfdiffusion``
uploads both PDBs and lands them on NAS at
``/data/rfantibody_jobs/<bootstrap_id>/input/{target,framework}.pdb`` as a
side effect of ``framework.JobRunner.submit``.  The async rfdiffusion test
then references both via ``file://`` URIs (NEW in v0.0.18 — earlier versions
only accepted UploadFile).  ProteinMPNN and RF2 chain off prior outputs
via ``job://<prev_job_id>/<file>`` URIs, which already worked.

Override the bootstrap with ``RFANTIBODY_TEST_TARGET_NAS_PATH=`` /
``RFANTIBODY_TEST_FRAMEWORK_NAS_PATH=`` env vars if you have the PDBs
pre-staged elsewhere on NAS.

This file requires deployed version >= v0.0.18.
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

SERVICE = "rfantibody-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET_PDB = DATA_DIR / "rsv_site3.pdb"
FRAMEWORK_PDB = DATA_DIR / "hu-4D5-8_Fv.pdb"

PRESTAGED_TARGET = os.environ.get("RFANTIBODY_TEST_TARGET_NAS_PATH")
PRESTAGED_FRAMEWORK = os.environ.get("RFANTIBODY_TEST_FRAMEWORK_NAS_PATH")

# Jobs base dir on the FC instance — must match Dockerfile's
# ``RFANTIBODY_JOBS_BASE_DIR`` (settings default).
JOBS_BASE_DIR_ON_FC = "/data/rfantibody_jobs"

# RFdiffusion is fast at diffuser_t=25 + num_designs=1 (~3-5 min). RF2 with
# num_recycles=2 takes ~5-10 min.  Allow 30 min per stage to absorb cold
# start + weight load.
POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 20

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
def staged_pdb_uris(client: httpx.Client) -> tuple[str, str]:
    """One-time sync upload that lands target + framework on the FC NAS.

    Returns ``(target_uri, framework_uri)`` for use as ``target_uri`` /
    ``framework_uri`` in subsequent async submits.

    The sync POST runs a real RFdiffusion inference in the background —
    we don't need its output, only the side-effect of saving both PDBs to
    NAS before ``submit`` returns.  Net cost: 1 extra inference run.

    If both ``RFANTIBODY_TEST_*_NAS_PATH`` env vars are set, the bootstrap
    upload is skipped.
    """
    if PRESTAGED_TARGET and PRESTAGED_FRAMEWORK:
        return f"file://{PRESTAGED_TARGET}", f"file://{PRESTAGED_FRAMEWORK}"

    with open(TARGET_PDB, "rb") as t, open(FRAMEWORK_PDB, "rb") as f:
        r = client.post(
            "/api/rfdiffusion",
            files={
                "target": (TARGET_PDB.name, t.read(), "chemical/x-pdb"),
                "framework": (FRAMEWORK_PDB.name, f.read(), "chemical/x-pdb"),
            },
            data={
                "num_designs": "1",
                "diffuser_t": "25",
                "design_loops": "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13",
                "hotspots": "T305,T456",
                "deterministic": "true",
            },
        )
    assert r.status_code == 200, (
        f"bootstrap sync upload failed: {r.status_code} {r.text!r}"
    )
    job_id = r.json()["job_id"]
    base = f"file://{JOBS_BASE_DIR_ON_FC}/{job_id}/input"
    return f"{base}/target.pdb", f"{base}/framework.pdb"


# Task-id fixtures — one per endpoint.  Pipeline chained via job:// URIs:
# rfdiffusion -> proteinmpnn -> rf2.
@pytest.fixture(scope="module")
def rfdiffusion_task_id() -> str:
    return f"fc-async-rfd-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def proteinmpnn_task_id() -> str:
    return f"fc-async-mpnn-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def rf2_task_id() -> str:
    return f"fc-async-rf2-{int(time.time())}-{uuid.uuid4().hex[:6]}"


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

    See ``project_fc_http_polling_unreliable_at_concurrency.md``.
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
    # rfantibody's FC concurrent-request budget is very tight, so
    # ``GET /api/jobs/<id>`` polls can hit 429 for 4-7 min at a time.
    # Bump ``max_transient_errors`` well above the framework default (10)
    # so poll_job rides out throttle windows instead of bailing.  Effective
    # retry buffer: 60 × 20s = 20 min of consecutive 429s.
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
# Per-endpoint submit fixtures — pipeline chained
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rfdiffusion_submit_response(
    client: httpx.Client,
    rfdiffusion_task_id: str,
    staged_pdb_uris: tuple[str, str],
) -> httpx.Response:
    """Async rfdiffusion submit — both PDBs via file:// URIs."""
    target_uri, framework_uri = staged_pdb_uris
    return client.post(
        "/api/tasks/rfdiffusion",
        data={
            "target_uri": target_uri,
            "framework_uri": framework_uri,
            "num_designs": "1",
            "diffuser_t": "25",
            "design_loops": "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13",
            "hotspots": "T305,T456",
            "deterministic": "true",
        },
        headers=_async_headers(rfdiffusion_task_id),
    )


@pytest.fixture(scope="module")
def rfdiffusion_task(
    client: httpx.Client,
    rfdiffusion_task_id: str,
    rfdiffusion_submit_response: httpx.Response,
) -> dict:
    assert rfdiffusion_submit_response.status_code == 202, (
        f"async rfdiffusion submit returned {rfdiffusion_submit_response.status_code}: "
        f"{rfdiffusion_submit_response.text!r}.  This file requires v0.0.18+ "
        f"(URI fallback for target / framework).  Verify the deployed version."
    )
    return _poll_to_completion(client, rfdiffusion_task_id)


@pytest.fixture(scope="module")
def proteinmpnn_submit_response(
    client: httpx.Client,
    proteinmpnn_task_id: str,
    rfdiffusion_task: dict,
) -> httpx.Response:
    """Async proteinmpnn submit — chained via job:// from rfdiffusion output."""
    rfd_id = rfdiffusion_task["job_id"]
    return client.post(
        "/api/tasks/proteinmpnn",
        data={
            "input_quiver_uri": f"job://{rfd_id}/1_rfdiffusion.qv",
            "seqs_per_struct": "1",
            "deterministic": "true",
        },
        headers=_async_headers(proteinmpnn_task_id),
    )


@pytest.fixture(scope="module")
def proteinmpnn_task(
    client: httpx.Client,
    proteinmpnn_task_id: str,
    proteinmpnn_submit_response: httpx.Response,
) -> dict:
    assert proteinmpnn_submit_response.status_code == 202, (
        f"async proteinmpnn submit returned {proteinmpnn_submit_response.status_code}: "
        f"{proteinmpnn_submit_response.text!r}"
    )
    return _poll_to_completion(client, proteinmpnn_task_id)


@pytest.fixture(scope="module")
def rf2_submit_response(
    client: httpx.Client,
    rf2_task_id: str,
    proteinmpnn_task: dict,
) -> httpx.Response:
    """Async rf2 submit — chained via job:// from proteinmpnn output."""
    mpnn_id = proteinmpnn_task["job_id"]
    return client.post(
        "/api/tasks/rf2",
        data={
            "input_quiver_uri": f"job://{mpnn_id}/2_proteinmpnn.qv",
            "num_recycles": "2",
        },
        headers=_async_headers(rf2_task_id),
    )


@pytest.fixture(scope="module")
def rf2_task(
    client: httpx.Client,
    rf2_task_id: str,
    rf2_submit_response: httpx.Response,
) -> dict:
    assert rf2_submit_response.status_code == 202, (
        f"async rf2 submit returned {rf2_submit_response.status_code}: "
        f"{rf2_submit_response.text!r}"
    )
    return _poll_to_completion(client, rf2_task_id)


# ===================================================================
# Section 1: Submit semantics + OpenAPI registration
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_rfdiffusion_returns_202(self, rfdiffusion_submit_response):
        assert rfdiffusion_submit_response.status_code == 202, (
            f"expected 202; got {rfdiffusion_submit_response.status_code} "
            f"body={rfdiffusion_submit_response.text!r}"
        )

    def test_proteinmpnn_returns_202(self, proteinmpnn_submit_response):
        assert proteinmpnn_submit_response.status_code == 202, (
            f"expected 202; got {proteinmpnn_submit_response.status_code} "
            f"body={proteinmpnn_submit_response.text!r}"
        )

    def test_rf2_returns_202(self, rf2_submit_response):
        assert rf2_submit_response.status_code == 202, (
            f"expected 202; got {rf2_submit_response.status_code} "
            f"body={rf2_submit_response.text!r}"
        )

    def test_task_endpoints_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200, (
            f"openapi.json fetch failed: {r.status_code} {r.text!r}"
        )
        spec = r.json()
        expected = {
            "/api/tasks/rfdiffusion",
            "/api/tasks/proteinmpnn",
            "/api/tasks/rf2",
        }
        missing = expected - set(spec["paths"])
        assert not missing, (
            f"task endpoints missing from OpenAPI: {missing}; "
            f"settings.task_endpoints_enabled may be False"
        )


# ===================================================================
# Section 2: Per-stage completion + outputs
# ===================================================================


def _assert_completed_with_qv(task: dict, task_id: str, client: httpx.Client,
                              expected_qv_name: str, *, min_duration_s: float = 3.0):
    """Assert a completed task produced ``expected_qv_name``.

    ``min_duration_s`` is a low-bar sanity check that the subprocess actually
    ran (defaults to 3 s so proteinmpnn's fast single-seq path passes;
    rfdiffusion + rf2 callers pass a larger value).
    """
    assert task["status"] == "completed"
    assert task["job_id"] == task_id
    assert task.get("started_at") is not None
    assert task.get("completed_at") is not None
    d = task.get("duration_seconds")
    assert d is not None and d > min_duration_s, (
        f"duration {d}s too short (min {min_duration_s}s) — subprocess may not have run"
    )
    assert task.get("output_count", 0) > 0
    assert task.get("output_total_bytes", 0) > 0

    r = _get_with_retry(client, f"/api/jobs/{task_id}/files")
    assert r.status_code == 200
    files = r.json()["files"]
    assert any(expected_qv_name in f for f in files), (
        f"{expected_qv_name} missing from outputs: {files}"
    )


@pytest.mark.fc
class TestAsyncRfdiffusion:
    def test_completed(self, rfdiffusion_task, rfdiffusion_task_id, client):
        # rfdiffusion at diffuser_t=25 num_designs=1 takes ~3-5 min minimum.
        _assert_completed_with_qv(
            rfdiffusion_task, rfdiffusion_task_id, client, "1_rfdiffusion.qv",
            min_duration_s=60,
        )

    def test_input_params_echoed(self, rfdiffusion_task):
        params = rfdiffusion_task.get("input_params") or {}
        assert params.get("num_designs") == 1
        assert params.get("diffuser_t") == 25
        assert params.get("hotspots") == "T305,T456"
        assert params.get("deterministic") is True

    def test_qv_downloadable(self, client, rfdiffusion_task_id, rfdiffusion_task):
        files = _get_with_retry(
            client, f"/api/jobs/{rfdiffusion_task_id}/files"
        ).json()["files"]
        qv = next(f for f in files if "1_rfdiffusion.qv" in f)
        r = _get_with_retry(client, f"/api/jobs/{rfdiffusion_task_id}/file/{qv}")
        assert r.status_code == 200
        assert len(r.content) > 100, (
            f"1_rfdiffusion.qv unexpectedly small: {len(r.content)} bytes"
        )


@pytest.mark.fc
class TestAsyncProteinMPNN:
    def test_completed(self, proteinmpnn_task, proteinmpnn_task_id, client):
        _assert_completed_with_qv(
            proteinmpnn_task, proteinmpnn_task_id, client, "2_proteinmpnn.qv"
        )

    def test_input_params_echoed(self, proteinmpnn_task):
        params = proteinmpnn_task.get("input_params") or {}
        assert params.get("seqs_per_struct") == 1
        assert params.get("deterministic") is True
        assert params.get("loops") == "H1,H2,H3"

    def test_qv_downloadable(self, client, proteinmpnn_task_id, proteinmpnn_task):
        files = _get_with_retry(
            client, f"/api/jobs/{proteinmpnn_task_id}/files"
        ).json()["files"]
        qv = next(f for f in files if "2_proteinmpnn.qv" in f)
        r = _get_with_retry(client, f"/api/jobs/{proteinmpnn_task_id}/file/{qv}")
        assert r.status_code == 200
        assert len(r.content) > 100


@pytest.mark.fc
class TestAsyncRF2:
    def test_completed(self, rf2_task, rf2_task_id, client):
        # RF2 at num_recycles=2 takes ~5-10 min minimum.
        _assert_completed_with_qv(
            rf2_task, rf2_task_id, client, "3_rf2.qv", min_duration_s=60,
        )

    def test_input_params_echoed(self, rf2_task):
        params = rf2_task.get("input_params") or {}
        assert params.get("num_recycles") == 2
        assert params.get("hotspot_show_prop") == 0.1

    def test_qv_downloadable(self, client, rf2_task_id, rf2_task):
        files = _get_with_retry(
            client, f"/api/jobs/{rf2_task_id}/files"
        ).json()["files"]
        qv = next(f for f in files if "3_rf2.qv" in f)
        r = _get_with_retry(client, f"/api/jobs/{rf2_task_id}/file/{qv}")
        assert r.status_code == 200
        assert len(r.content) > 100


# ===================================================================
# Section 3: Job lifecycle on the rfdiffusion task (cheapest fixture)
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, rfdiffusion_task_id, rfdiffusion_task):
        r = _get_with_retry(client, f"/api/jobs/{rfdiffusion_task_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == rfdiffusion_task_id
        assert body["status"] == "completed"

    def test_log_endpoint(self, client, rfdiffusion_task_id, rfdiffusion_task):
        r = _get_with_retry(client, f"/api/jobs/{rfdiffusion_task_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0, "log should be non-empty for a completed job"

    def test_download_zip(self, client, rfdiffusion_task_id, rfdiffusion_task):
        r = _get_with_retry(client, f"/api/jobs/{rfdiffusion_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any("1_rfdiffusion.qv" in n for n in names), (
            f"1_rfdiffusion.qv missing from zip: {names}"
        )


# ===================================================================
# Section 4: Duplicate dedup — FC platform layer rejects repeat task_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmitting the same X-Fc-Async-Task-Id after completion.

    Per the FC async task mode contract (engineering/decisions/
    2026-06-17-fc-async-task-mode.md and project memory
    ``project_fc_async_dedup_at_platform_layer.md``), FC dedups by
    ``X-Fc-Async-Task-Id`` at the platform layer — duplicate returns 409
    without invoking the function.  If FC forwards anyway, the framework
    layer (``execute_task``) returns the existing JobInfo without re-running.
    """

    def test_duplicate_does_not_rerun(
        self,
        client: httpx.Client,
        proteinmpnn_task_id: str,
        proteinmpnn_task: dict,
        rfdiffusion_task: dict,
    ):
        first_created_at = proteinmpnn_task["created_at"]
        first_completed_at = proteinmpnn_task["completed_at"]
        first_seqs = (proteinmpnn_task.get("input_params") or {}).get("seqs_per_struct")

        # Resubmit same task_id with different seqs_per_struct to prove dedup.
        rfd_id = rfdiffusion_task["job_id"]
        r2 = client.post(
            "/api/tasks/proteinmpnn",
            data={
                "input_quiver_uri": f"job://{rfd_id}/1_rfdiffusion.qv",
                "seqs_per_struct": "3",  # different from first run's 1
                "deterministic": "true",
            },
            headers=_async_headers(proteinmpnn_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → framework "
            f"dedups); got {r2.status_code} body={r2.text!r}"
        )

        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(client, f"/api/jobs/{proteinmpnn_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("seqs_per_struct") == first_seqs, (
            "duplicate async submit must not overwrite input_params"
        )

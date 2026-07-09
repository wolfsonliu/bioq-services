"""FC async task mode tests for openbpmd-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/openbpmd-server/tests/test_fc_task.py -v

Validates /api/tasks/score under FC async task mode (``X-Fc-Invocation-Type:
Async``). Async task mode pins the FC instance for the whole job (no HTTP
gateway recycle risk) and dedups by ``X-Fc-Async-Task-Id`` at the platform
layer.

Input staging — sync bootstrap, then ``file://`` URIs
-----------------------------------------------------
FC async invocation caps the event payload at 128 KiB, but the Amber system
is ~10 MB.  So we do a sync bootstrap: one sync POST to /api/score lands
structure + parameters on NAS at
``/data/openbpmd_jobs/<id>/input/{solvated.rst7,solvated.prm7}`` (a
side-effect of _save_inputs running before the subprocess).  Async submits
then reference those via ``file://`` URIs.

As with test_fc.py the run uses the short-trajectory knobs (sim_ns=0.02,
nreps=1) so the regression finishes in minutes.
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

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "openbpmd-server"

_DEFAULT_CLONE = (
    Path(__file__).resolve().parents[3]
    / "opensource" / "OpenBPMD" / "tests" / "files"
)
TEST_STRUCTURE = Path(
    os.environ.get("OPENBPMD_TEST_STRUCTURE", _DEFAULT_CLONE / "solvated.rst7")
)
TEST_PARAMETERS = Path(
    os.environ.get("OPENBPMD_TEST_PARAMETERS", _DEFAULT_CLONE / "solvated.prm7")
)
LIG_RESNAME = os.environ.get("OPENBPMD_TEST_LIG_RESNAME", "UNK")

# NAS layout on the deployed FC service — must match settings.jobs_base_dir.
JOBS_BASE_DIR_ON_FC = "/data/openbpmd_jobs"

# Optional pre-staged NAS paths to skip the sync bootstrap on reruns.
PRESTAGED_STRUCTURE = os.environ.get("OPENBPMD_TEST_STRUCTURE_NAS_PATH")
PRESTAGED_PARAMETERS = os.environ.get("OPENBPMD_TEST_PARAMETERS_NAS_PATH")

POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 20
TIMEOUT = httpx.Timeout(connect=30, read=300, write=600, pool=30)

_fixtures_present = TEST_STRUCTURE.exists() and TEST_PARAMETERS.exists()
_needs_fixtures = pytest.mark.skipif(
    not (_fixtures_present or (PRESTAGED_STRUCTURE and PRESTAGED_PARAMETERS)),
    reason=f"fixtures missing: {TEST_STRUCTURE} / {TEST_PARAMETERS}",
)


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
def staged_uris(client: httpx.Client) -> tuple[str, str]:
    """Sync bootstrap that lands both inputs on FC NAS; returns file:// URIs."""
    if PRESTAGED_STRUCTURE and PRESTAGED_PARAMETERS:
        return f"file://{PRESTAGED_STRUCTURE}", f"file://{PRESTAGED_PARAMETERS}"

    with open(TEST_STRUCTURE, "rb") as fh_s, open(TEST_PARAMETERS, "rb") as fh_p:
        r = client.post(
            "/api/score",
            files={
                "structure": ("solvated.rst7", fh_s.read(), "application/octet-stream"),
                "parameters": ("solvated.prm7", fh_p.read(), "application/octet-stream"),
            },
            data={
                "lig_resname": LIG_RESNAME,
                "nreps": "1",
                "sim_ns": "0.02",
                "equil_steps": "500",
            },
        )
    assert r.status_code == 200, f"bootstrap failed: {r.status_code} {r.text!r}"
    job_id = r.json()["job_id"]
    base = f"file://{JOBS_BASE_DIR_ON_FC}/{job_id}/input"
    return f"{base}/solvated.rst7", f"{base}/solvated.prm7"


@pytest.fixture(scope="module")
def score_task_id() -> str:
    return f"fc-async-score-{int(time.time())}-{uuid.uuid4().hex[:6]}"


def _async_headers(task_id: str) -> dict[str, str]:
    return {
        "X-Fc-Invocation-Type": "Async",
        "X-Bioagent-Job-Id": task_id,
        "X-Fc-Async-Task-Id": task_id,
    }


def _get_with_retry(client, path, *, max_attempts=20, backoff_s=30):
    last = None
    for _ in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    return last


def _poll_to_completion(client, task_id: str) -> dict:
    final = poll_job(
        client, "", task_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        max_transient_errors=60,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


@pytest.fixture(scope="module")
def score_submit_response(client, score_task_id, staged_uris: tuple[str, str]):
    structure_uri, parameters_uri = staged_uris
    return client.post(
        "/api/tasks/score",
        data={
            "structure_uri": structure_uri,
            "parameters_uri": parameters_uri,
            "lig_resname": LIG_RESNAME,
            "nreps": "1",
            "sim_ns": "0.02",
            "equil_steps": "500",
        },
        headers=_async_headers(score_task_id),
    )


@pytest.fixture(scope="module")
def score_task(client, score_task_id, score_submit_response) -> dict:
    assert score_submit_response.status_code == 202, (
        f"async submit returned {score_submit_response.status_code}: "
        f"{score_submit_response.text!r}"
    )
    return _poll_to_completion(client, score_task_id)


# ===================================================================
# Section 1: submit semantics + OpenAPI
# ===================================================================


@pytest.mark.fc
@_needs_fixtures
class TestAsyncSubmit:
    def test_score_returns_202(self, score_submit_response):
        assert score_submit_response.status_code == 202

    def test_task_endpoints_registered(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200
        assert "/api/tasks/score" in r.json()["paths"]


# ===================================================================
# Section 2: completion + output
# ===================================================================


@pytest.mark.fc
@_needs_fixtures
class TestAsyncScore:
    def test_completed(self, score_task, score_task_id, client):
        assert score_task["status"] == "completed"
        assert score_task["job_id"] == score_task_id
        assert score_task.get("output_count", 0) > 0
        files = _get_with_retry(
            client, f"/api/jobs/{score_task_id}/files"
        ).json()["files"]
        assert any(f == "results.csv" for f in files)
        assert any(f == "scoring_stats.json" for f in files)

    def test_input_params_echoed(self, score_task):
        params = score_task.get("input_params") or {}
        assert params.get("nreps") == 1
        assert params.get("lig_resname") == LIG_RESNAME

    def test_results_downloadable(self, client, score_task_id, score_task):
        r = _get_with_retry(client, f"/api/jobs/{score_task_id}/file/results.csv")
        assert r.status_code == 200
        assert b"CompScore" in r.content


# ===================================================================
# Section 3: lifecycle
# ===================================================================


@pytest.mark.fc
@_needs_fixtures
class TestJobLifecycle:
    def test_status_endpoint(self, client, score_task_id, score_task):
        body = _get_with_retry(client, f"/api/jobs/{score_task_id}").json()
        assert body["status"] == "completed"

    def test_download_zip(self, client, score_task_id, score_task):
        r = _get_with_retry(client, f"/api/jobs/{score_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any("results.csv" in n for n in zf.namelist())


# ===================================================================
# Section 4: platform-layer dedup — repeat submit
# ===================================================================


@pytest.mark.fc
@_needs_fixtures
class TestAsyncDuplicateDedup:
    def test_duplicate_does_not_rerun(self, client, score_task_id, score_task, staged_uris):
        first_created = score_task["created_at"]
        first_completed = score_task["completed_at"]

        structure_uri, parameters_uri = staged_uris
        r2 = client.post(
            "/api/tasks/score",
            data={
                "structure_uri": structure_uri,
                "parameters_uri": parameters_uri,
                "lig_resname": LIG_RESNAME,
                "nreps": "1",
                "sim_ns": "0.02",
                "equil_steps": "500",
            },
            headers=_async_headers(score_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 202 or 409; got {r2.status_code} {r2.text!r}"
        )
        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(client, f"/api/jobs/{score_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created, "created_at was reset"
        assert re_query["completed_at"] == first_completed, "task was rerun"

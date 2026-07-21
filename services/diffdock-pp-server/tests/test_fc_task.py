"""FC async task mode tests for diffdock-pp-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/diffdock-pp-server/tests/test_fc_task.py -v

Validates the /api/tasks/dock endpoint end-to-end under FC async task mode
(``X-Fc-Invocation-Type: Async``). Async task mode pins the FC instance
for the whole job (no 30s HTTP-gateway recycle risk) and dedups by
``X-Fc-Async-Task-Id`` at the platform layer.

PDB source — sync bootstrap, then ``file://`` URIs
--------------------------------------------------
FC async invocation caps the event payload at **128 KiB**. Our 1a2k
fixtures are 161 KB (receptor) + 129 KB (ligand) = ~290 KB in multipart,
so uploading them directly in the async submit gets rejected with
``EntityTooLarge`` at the FC gateway.

So we use a sync-bootstrap pattern: one sync POST to ``/api/dock`` (with
smallest valid params) lands both PDBs at
``/data/diffdock_pp_jobs/<bootstrap_id>/input/{receptor,ligand}.pdb`` as a
side-effect of framework's ``_save_inputs`` running BEFORE the subprocess
starts.  Subsequent async submits reference the staged files via
``file://`` URIs.

Override the bootstrap with ``DIFFDOCK_PP_TEST_RECEPTOR_NAS_PATH=`` /
``DIFFDOCK_PP_TEST_LIGAND_NAS_PATH=`` env vars pointing at PDB files
pre-staged elsewhere on NAS — skips the bootstrap inference cost on reruns.
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

SERVICE = "diffdock-pp-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_RECEPTOR = DATA_DIR / "1a2k_receptor.pdb"
TEST_LIGAND = DATA_DIR / "1a2k_ligand.pdb"

# NAS layout on the deployed FC service — must match settings.jobs_base_dir.
JOBS_BASE_DIR_ON_FC = "/data/diffdock_pp_jobs"

# Optional pre-staged NAS paths to skip the sync-bootstrap step on reruns.
PRESTAGED_RECEPTOR = os.environ.get("DIFFDOCK_PP_TEST_RECEPTOR_NAS_PATH")
PRESTAGED_LIGAND = os.environ.get("DIFFDOCK_PP_TEST_LIGAND_NAS_PATH")

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
    """One-time sync upload that lands receptor + ligand on FC NAS.

    Returns ``(receptor_uri, ligand_uri)`` for use as ``receptor_uri`` /
    ``ligand_uri`` in subsequent async submits.

    The sync POST runs a real DiffDock-PP inference in the background —
    we don't need its output, only the side-effect of ``_save_inputs``
    persisting both PDBs to NAS before submit returns.  Net cost: 1
    extra minimal (num_samples=1) inference run.

    If both ``DIFFDOCK_PP_TEST_*_NAS_PATH`` env vars are set, the
    bootstrap is skipped.
    """
    if PRESTAGED_RECEPTOR and PRESTAGED_LIGAND:
        return f"file://{PRESTAGED_RECEPTOR}", f"file://{PRESTAGED_LIGAND}"

    with open(TEST_RECEPTOR, "rb") as fh_r, open(TEST_LIGAND, "rb") as fh_l:
        r = client.post(
            "/api/dock",
            # Force stable filenames — the framework saves upload.filename,
            # so we control the resulting NAS path.
            files={
                "receptor": ("receptor.pdb", fh_r.read(), "chemical/x-pdb"),
                "ligand": ("ligand.pdb", fh_l.read(), "chemical/x-pdb"),
            },
            data={
                "num_samples": "1",
                "top_k": "1",
                "use_confidence_model": "false",  # skip confidence pass to cut cost
                "seed": "0",
            },
        )
    assert r.status_code == 200, (
        f"bootstrap sync upload failed: {r.status_code} {r.text!r}"
    )
    job_id = r.json()["job_id"]
    base = f"file://{JOBS_BASE_DIR_ON_FC}/{job_id}/input"
    return f"{base}/receptor.pdb", f"{base}/ligand.pdb"


@pytest.fixture(scope="module")
def dock_task_id() -> str:
    return f"fc-async-dock-{int(time.time())}-{uuid.uuid4().hex[:6]}"


def _async_headers(task_id: str) -> dict[str, str]:
    return {
        "X-Fc-Invocation-Type": "Async",
        "X-Bioagent-Job-Id": task_id,
        "X-Fc-Async-Task-Id": task_id,
    }


def _get_with_retry(client, path, *, max_attempts=20, backoff_s=30):
    """FC gateway 429 retry. Essential for max_concurrent_jobs=1 services."""
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
    assert final["status"] == "completed", (
        f"task did not complete: {final}"
    )
    return final


@pytest.fixture(scope="module")
def dock_submit_response(
    client, dock_task_id, staged_pdb_uris: tuple[str, str],
):
    receptor_uri, ligand_uri = staged_pdb_uris
    return client.post(
        "/api/tasks/dock",
        data={
            "receptor_uri": receptor_uri,
            "ligand_uri": ligand_uri,
            "num_samples": "4",
            "top_k": "2",
            "use_confidence_model": "true",
            "seed": "42",
        },
        headers=_async_headers(dock_task_id),
    )


@pytest.fixture(scope="module")
def dock_task(client, dock_task_id, dock_submit_response) -> dict:
    assert dock_submit_response.status_code == 202, (
        f"async dock submit returned {dock_submit_response.status_code}: "
        f"{dock_submit_response.text!r}"
    )
    return _poll_to_completion(client, dock_task_id)


# ===================================================================
# Section 1: submit semantics + OpenAPI
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_dock_returns_202(self, dock_submit_response):
        assert dock_submit_response.status_code == 202

    def test_task_endpoints_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200
        expected = {"/api/tasks/dock"}
        missing = expected - set(r.json()["paths"])
        assert not missing, (
            f"task endpoints missing: {missing}; "
            f"settings.task_endpoints_enabled may be False"
        )


# ===================================================================
# Section 2: completion + output
# ===================================================================


@pytest.mark.fc
class TestAsyncDock:
    def test_completed(self, dock_task, dock_task_id, client):
        assert dock_task["status"] == "completed"
        assert dock_task["job_id"] == dock_task_id
        d = dock_task.get("duration_seconds")
        assert d is not None and d > 30.0, (
            f"duration {d}s too short — subprocess may not have run "
            f"real diffusion sampling"
        )
        assert dock_task.get("output_count", 0) > 0
        r = _get_with_retry(client, f"/api/jobs/{dock_task_id}/files")
        assert r.status_code == 200
        files = r.json()["files"]
        assert any(f == "dock_pose_1.pdb" for f in files)
        assert any(f == "confidence_scores.json" for f in files)

    def test_input_params_echoed(self, dock_task):
        params = dock_task.get("input_params") or {}
        assert params.get("num_samples") == 4
        assert params.get("top_k") == 2

    def test_dock_pose_downloadable(self, client, dock_task_id, dock_task):
        files = _get_with_retry(
            client, f"/api/jobs/{dock_task_id}/files"
        ).json()["files"]
        pdb = next(f for f in files if f == "dock_pose_1.pdb")
        r = _get_with_retry(client, f"/api/jobs/{dock_task_id}/file/{pdb}")
        assert r.status_code == 200
        assert len(r.content) > 100


# ===================================================================
# Section 3: lifecycle
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, dock_task_id, dock_task):
        body = _get_with_retry(client, f"/api/jobs/{dock_task_id}").json()
        assert body["status"] == "completed"

    def test_log_endpoint(self, client, dock_task_id, dock_task):
        r = _get_with_retry(client, f"/api/jobs/{dock_task_id}/log")
        assert r.status_code == 200
        assert len((r.json().get("log") or r.json().get("text") or "")) > 0

    def test_download_zip(self, client, dock_task_id, dock_task):
        r = _get_with_retry(client, f"/api/jobs/{dock_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any("dock_pose_" in n for n in zf.namelist())


# ===================================================================
# Section 4: platform-layer dedup — repeat submit
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Repeat submits of the same X-Fc-Async-Task-Id must not rerun."""

    def test_duplicate_does_not_rerun(
        self, client, dock_task_id, dock_task, staged_pdb_uris,
    ):
        first_created = dock_task["created_at"]
        first_completed = dock_task["completed_at"]

        receptor_uri, ligand_uri = staged_pdb_uris
        r2 = client.post(
            "/api/tasks/dock",
            data={
                "receptor_uri": receptor_uri,
                "ligand_uri": ligand_uri,
                # Deliberately different params — must not take effect if dedup works.
                "num_samples": "8",
                "top_k": "4",
                "seed": "999",
            },
            headers=_async_headers(dock_task_id),
        )
        # FC platform dedup returns 409; framework-layer dedup returns 202 + existing JobInfo.
        assert r2.status_code in (202, 409), (
            f"expected 202 or 409; got {r2.status_code} {r2.text!r}"
        )
        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(
            client, f"/api/jobs/{dock_task_id}"
        ).json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created, "created_at was reset"
        assert re_query["completed_at"] == first_completed, "task was rerun"

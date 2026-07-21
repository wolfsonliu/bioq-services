"""FC async task mode tests for diffusion-hopping-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/diffusion-hopping-server/tests/test_fc_task.py -v

Validates ``/api/tasks/generate`` end-to-end under FC async task mode
(``X-Fc-Invocation-Type: Async``).  Async task mode pins the FC instance
for the whole job (no 30 s HTTP-gateway recycle risk) and dedups by
``X-Fc-Async-Task-Id`` at the platform layer.

Payload sizing — sync bootstrap
-------------------------------
FC's async invocation gateway caps the inbound event payload at 128 KiB
(``EntityTooLarge`` 400 otherwise).  Our test protein is ~680 KB, which
blows through the cap.  So we use a sync-bootstrap pattern: one sync POST
to ``/api/generate`` uploads both PDB + SDF and lands them on NAS at
``/data/diffusion_hopping_jobs/<bootstrap_id>/input/{...}`` as a side
effect of ``JobRunner.submit``.  Async tests then reference both via
``file://`` URIs.  Net cost: 1 extra inference run.

Override the bootstrap with ``DIFFHOPP_TEST_PROTEIN_NAS_PATH=`` /
``DIFFHOPP_TEST_LIGAND_NAS_PATH=`` env vars if pre-staged elsewhere on NAS.
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

SERVICE = "diffusion-hopping-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_PROTEIN = DATA_DIR / "1a0q_protein.pdb"
TEST_LIGAND = DATA_DIR / "1a0q_ligand.sdf"

PRESTAGED_PROTEIN = os.environ.get("DIFFHOPP_TEST_PROTEIN_NAS_PATH")
PRESTAGED_LIGAND = os.environ.get("DIFFHOPP_TEST_LIGAND_NAS_PATH")

# Jobs base dir on the FC instance — must match Dockerfile's
# DIFFUSION_HOPPING_JOBS_BASE_DIR (settings default).
JOBS_BASE_DIR_ON_FC = "/data/diffusion_hopping_jobs"

# Diffusion sampling is much slower than turbohopp's consistency model —
# 3 samples * ~250 denoising steps.  Give it a generous window.
POLL_TIMEOUT_S = 1200
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
def staged_uris(client: httpx.Client) -> tuple[str, str]:
    """One-time sync upload that lands protein + ligand on the FC NAS.

    Returns ``(protein_uri, reference_ligand_uri)`` as ``file://`` URIs.
    The sync POST runs a real (short) generation — we only need its
    side-effect of saving both inputs to NAS before ``submit`` returns.
    """
    if PRESTAGED_PROTEIN and PRESTAGED_LIGAND:
        return f"file://{PRESTAGED_PROTEIN}", f"file://{PRESTAGED_LIGAND}"

    with open(TEST_PROTEIN, "rb") as fh_p, open(TEST_LIGAND, "rb") as fh_l:
        r = client.post(
            "/api/generate",
            files={
                "protein": ("1a0q_protein.pdb", fh_p.read(), "chemical/x-pdb"),
                "reference_ligand": (
                    "1a0q_ligand.sdf", fh_l.read(), "chemical/x-mdl-sdfile",
                ),
            },
            data={
                "num_samples": "1",
                "model_variant": "gvp_conditional",
            },
        )
    assert r.status_code == 200, (
        f"bootstrap sync upload failed: {r.status_code} {r.text!r}"
    )
    job_id = r.json()["job_id"]
    base = f"file://{JOBS_BASE_DIR_ON_FC}/{job_id}/input"
    return f"{base}/1a0q_protein.pdb", f"{base}/1a0q_ligand.sdf"


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


def _poll_to_completion(client: httpx.Client, task_id: str) -> dict:
    # ``max_concurrent_jobs=1`` here → GET /api/jobs/<id> can 429 for
    # 4-7 min at a stretch.  Bump ``max_transient_errors`` well above the
    # framework default (10) so poll_job rides out throttle windows.
    final = poll_job(
        client, "", task_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        max_transient_errors=60,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


# ---------------------------------------------------------------------------
# submit + poll fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generate_submit_response(
    client: httpx.Client,
    generate_task_id: str,
    staged_uris: tuple[str, str],
) -> httpx.Response:
    protein_uri, ligand_uri = staged_uris
    return client.post(
        "/api/tasks/generate",
        data={
            "protein_uri": protein_uri,
            "reference_ligand_uri": ligand_uri,
            "num_samples": "3",
            "model_variant": "gvp_conditional",
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
        f"async generate submit returned "
        f"{generate_submit_response.status_code}: "
        f"{generate_submit_response.text!r}"
    )
    return _poll_to_completion(client, generate_task_id)


# ===================================================================
# Section 1: Submit semantics + OpenAPI registration
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_generate_returns_202(self, generate_submit_response):
        assert generate_submit_response.status_code == 202, (
            f"expected 202; got {generate_submit_response.status_code} "
            f"body={generate_submit_response.text!r}"
        )

    def test_task_endpoint_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200, (
            f"openapi.json fetch failed: {r.status_code} {r.text!r}"
        )
        spec = r.json()
        assert "/api/tasks/generate" in spec["paths"], (
            "task endpoint missing from OpenAPI; "
            "settings.task_endpoints_enabled may be False"
        )


# ===================================================================
# Section 2: Per-stage completion + outputs
# ===================================================================


def _assert_completed_with_sdf(
    task: dict,
    task_id: str,
    client: httpx.Client,
    *,
    min_duration_s: float = 5.0,
) -> list[str]:
    assert task["status"] == "completed"
    assert task["job_id"] == task_id
    assert task.get("started_at") is not None
    assert task.get("completed_at") is not None
    d = task.get("duration_seconds")
    assert d is not None and d > min_duration_s, (
        f"duration {d}s too short (min {min_duration_s}s) — "
        f"subprocess may not have run"
    )
    assert task.get("output_count", 0) > 0
    assert task.get("output_total_bytes", 0) > 0

    r = _get_with_retry(client, f"/api/jobs/{task_id}/files")
    assert r.status_code == 200
    files = r.json()["files"]
    assert any(f.endswith(".sdf") for f in files), (
        f"no .sdf files in outputs: {files}"
    )
    return files


@pytest.mark.fc
class TestAsyncGenerate:
    def test_completed(self, generate_task, generate_task_id, client):
        _assert_completed_with_sdf(
            generate_task, generate_task_id, client, min_duration_s=5.0,
        )

    def test_input_params_echoed(self, generate_task):
        params = generate_task.get("input_params") or {}
        assert params.get("num_samples") == 3
        assert params.get("model_variant") == "gvp_conditional"

    def test_sdf_downloadable(self, client, generate_task_id, generate_task):
        files = _get_with_retry(
            client, f"/api/jobs/{generate_task_id}/files",
        ).json()["files"]
        sdf = next(f for f in files if f.endswith(".sdf"))
        r = _get_with_retry(client, f"/api/jobs/{generate_task_id}/file/{sdf}")
        assert r.status_code == 200
        assert len(r.content) > 20, (
            f"{sdf} unexpectedly small: {len(r.content)} bytes"
        )


# ===================================================================
# Section 3: Job lifecycle on the async task
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
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0, "log should be non-empty for a completed job"

    def test_download_zip(self, client, generate_task_id, generate_task):
        r = _get_with_retry(client, f"/api/jobs/{generate_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any(n.endswith(".sdf") for n in zf.namelist()), (
            f"no .sdf in zip: {zf.namelist()}"
        )


# ===================================================================
# Section 4: Platform-layer dedup
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmitting the same X-Fc-Async-Task-Id after completion.

    FC dedups at the platform layer (409); if it forwards anyway,
    ``execute_task`` in the framework returns the existing JobInfo.  Either
    way, the pipeline must not re-run.
    """

    def test_duplicate_does_not_rerun(
        self,
        client: httpx.Client,
        generate_task_id: str,
        generate_task: dict,
        staged_uris: tuple[str, str],
    ):
        first_created_at = generate_task["created_at"]
        first_completed_at = generate_task["completed_at"]
        first_num_samples = (generate_task.get("input_params") or {}).get("num_samples")

        protein_uri, ligand_uri = staged_uris
        r2 = client.post(
            "/api/tasks/generate",
            data={
                "protein_uri": protein_uri,
                "reference_ligand_uri": ligand_uri,
                "num_samples": "7",  # different from first run's 3
                "model_variant": "egnn_conditional",
            },
            headers=_async_headers(generate_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → "
            f"framework dedups); got {r2.status_code} body={r2.text!r}"
        )
        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(
            client, f"/api/jobs/{generate_task_id}",
        ).json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("num_samples") == first_num_samples, (
            "duplicate async submit must not overwrite input_params"
        )

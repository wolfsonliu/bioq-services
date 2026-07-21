"""FC integration tests for diffusion-hopping-server (opt-in, sync submit/poll).

Marked ``@pytest.mark.fc``, skipped by default.  Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/diffusion-hopping-server/tests/test_fc.py -v

Fixtures (1a0q_protein.pdb + 1a0q_ligand.sdf) ship in tests/data/ — same
files as turbohopp-server so the two services can be regressed against the
same input for a fair speed comparison.

diffusion-hopping-server ships with ``max_concurrent_jobs=1``, so 429s from
the FC HTTP gateway are expected under any parallel work; every HTTP call
goes through ``_http_with_retry`` to absorb them.

DiffHopp diffusion sampling with default steps (~250) takes minutes on
H20/A10.  Allow 20 min per inference stage to absorb cold-start weight load.
"""

from __future__ import annotations

import io
import json
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "diffusion-hopping-server"
SESSION_HEADER = "bioagent-session-id"

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_PROTEIN = DATA_DIR / "1a0q_protein.pdb"
TEST_LIGAND = DATA_DIR / "1a0q_ligand.sdf"

TIMEOUT = httpx.Timeout(connect=30, read=600, write=600, pool=30)

# DiffHopp diffusion is slow — 100+ denoising steps per sample.
POLL_TIMEOUT_S = 1200
POLL_INTERVAL_S = 20


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
def session_headers() -> dict[str, str]:
    """Session affinity header so all polls hit the same FC instance."""
    return {SESSION_HEADER: f"test-{uuid.uuid4().hex[:12]}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_with_retry(
    call: Callable[[], httpx.Response],
    *,
    max_attempts: int = 20,
    backoff_s: int = 30,
) -> httpx.Response:
    """Retry FC's 429 ResourceExhausted.

    diffusion-hopping-server runs with ``max_concurrent_jobs=1`` so even
    sequential GETs occasionally trip the 429 gateway throttle.
    """
    last: httpx.Response | None = None
    for _attempt in range(max_attempts):
        last = call()
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


def _retry_get(client: httpx.Client, path: str, **kw: Any) -> httpx.Response:
    return _http_with_retry(lambda: client.get(path, **kw))


def _retry_post(client: httpx.Client, path: str, **kw: Any) -> httpx.Response:
    return _http_with_retry(lambda: client.post(path, **kw))


def _assert_submitted(body: dict) -> str:
    assert "job_id" in body
    assert body["status"] in ("pending", "running")
    assert body.get("created_at") is not None
    assert isinstance(body.get("input_params"), dict)
    return body["job_id"]


def _assert_completed(body: dict) -> None:
    assert body["status"] == "completed", (
        f"failed: kind={body.get('failure_kind')} "
        f"summary={body.get('error_summary')!r}"
    )
    assert body.get("started_at") is not None
    assert body.get("completed_at") is not None
    assert body.get("duration_seconds") is not None
    assert body["duration_seconds"] > 0
    assert body.get("output_count", 0) > 0
    assert body.get("output_total_bytes", 0) > 0


def _save_job_outputs(
    client: httpx.Client, job_id: str, job_info: dict, dst_dir: Path,
) -> None:
    """Persist JobInfo / log / zip / extracted SDFs for post-mortem inspection."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "jobinfo.json").write_text(json.dumps(job_info, indent=2))
    try:
        r = _retry_get(client, f"/api/jobs/{job_id}/log")
        if r.status_code == 200:
            body = r.json()
            (dst_dir / "subprocess.log").write_text(
                body.get("log") or body.get("text") or ""
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[fc_outputs] log download failed: {exc!r}")
    try:
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        if r.status_code == 200 and r.content:
            (dst_dir / f"{job_id}.zip").write_bytes(r.content)
            extract_to = dst_dir / "extracted"
            extract_to.mkdir(exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                zf.extractall(extract_to)
    except Exception as exc:  # noqa: BLE001
        print(f"[fc_outputs] zip download failed: {exc!r}")


# ---------------------------------------------------------------------------
# Module-scoped inference fixtures — one gvp + one egnn run per session.
# ---------------------------------------------------------------------------


def _submit_and_wait(
    client: httpx.Client,
    session_headers: dict[str, str],
    local_output_dir: Path,
    variant: str,
    label: str,
) -> dict:
    with open(TEST_PROTEIN, "rb") as fh_p, open(TEST_LIGAND, "rb") as fh_l:
        r = _retry_post(
            client, "/api/generate",
            files={
                "protein": ("1a0q_protein.pdb", fh_p.read(), "chemical/x-pdb"),
                "reference_ligand": (
                    "1a0q_ligand.sdf", fh_l.read(), "chemical/x-mdl-sdfile",
                ),
            },
            data={"num_samples": "3", "model_variant": variant},
            headers=session_headers,
        )
    assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
    job_id = _assert_submitted(r.json())

    final = poll_job(
        client, "", job_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        max_transient_errors=60,
        extra_headers=session_headers,
    )
    _save_job_outputs(client, job_id, final, local_output_dir / label)
    _assert_completed(final)
    return final


@pytest.fixture(scope="module")
def gvp_job(
    client: httpx.Client,
    session_headers: dict[str, str],
    local_output_dir: Path,
) -> dict:
    """gvp_conditional (DiffHopp paper main variant) — 3 samples."""
    return _submit_and_wait(
        client, session_headers, local_output_dir, "gvp_conditional", "gvp",
    )


@pytest.fixture(scope="module")
def egnn_job(
    client: httpx.Client,
    session_headers: dict[str, str],
    local_output_dir: Path,
) -> dict:
    """egnn_conditional — verifies the EGNN checkpoint set loads too."""
    return _submit_and_wait(
        client, session_headers, local_output_dir, "egnn_conditional", "egnn",
    )


# ===================================================================
# Section 1: Smoke (no inference compute)
# ===================================================================


@pytest.mark.fc
class TestSmoke:
    def test_healthz(self, client):
        r = _retry_get(client, "/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "diffusion-hopping"
        assert "version" in body

    def test_healthz_detail(self, client):
        """/healthz/detail reports NAS weights probe."""
        r = _retry_get(client, "/healthz/detail")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "diffusion-hopping"
        assert body["weights_dir"] == "/data/models/diffusion-hopping/checkpoints"
        assert body["weights_loaded"] is True, (
            f"NAS weights missing: {body.get('weights_missing')}. "
            f"Rsync the 4 ckpts under {body['weights_dir']}/."
        )
        assert body.get("weights_missing") == {}
        assert body["max_concurrent_jobs"] >= 1

    def test_openapi_served(self, client):
        r = _retry_get(client, "/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert "/api/generate" in spec["paths"]


# ===================================================================
# Section 2: Manifest
# ===================================================================


@pytest.mark.fc
class TestManifest:
    def test_sync_endpoint_listed(self, client):
        body = _retry_get(client, "/api/manifest").json()
        paths = {e["path"] for e in body["endpoints"]}
        assert "/api/generate" in paths
        extras = paths - {"/api/generate"}
        assert extras <= {"/api/tasks/generate"}, (
            f"unexpected non-task endpoints: {extras - {'/api/tasks/generate'}}"
        )


# ===================================================================
# Section 3: Errors (fast, no GPU)
# ===================================================================


@pytest.mark.fc
class TestErrors:
    def test_unknown_job_status_404(self, client):
        assert _retry_get(client, "/api/jobs/missing-job-id").status_code == 404

    def test_unknown_job_files_404(self, client):
        assert _retry_get(client, "/api/jobs/missing-job-id/files").status_code == 404

    def test_unknown_job_log_404(self, client):
        assert _retry_get(client, "/api/jobs/missing-job-id/log").status_code == 404

    def test_unknown_job_download_404(self, client):
        assert _retry_get(client, "/api/jobs/missing-job-id/download").status_code == 404

    def test_generate_rejects_missing_inputs(self, client):
        """Neither upload nor URI → 400/422."""
        r = _retry_post(client, "/api/generate", data={"num_samples": "3"})
        assert r.status_code in (400, 422)

    def test_generate_rejects_num_samples_out_of_range(self, client):
        r = _retry_post(
            client, "/api/generate",
            data={
                "protein_uri": "file:///nonexistent.pdb",
                "reference_ligand_uri": "file:///nonexistent.sdf",
                "num_samples": "0",
            },
        )
        assert r.status_code == 422

    def test_generate_rejects_bad_model_variant(self, client):
        r = _retry_post(
            client, "/api/generate",
            data={
                "protein_uri": "file:///nonexistent.pdb",
                "reference_ligand_uri": "file:///nonexistent.sdf",
                "model_variant": "does_not_exist",
            },
        )
        assert r.status_code == 422


# ===================================================================
# Section 4: Sync inference — GVP variant (module-scoped fixture)
# ===================================================================


@pytest.mark.fc
class TestSyncGVP:
    def test_job_completed(self, gvp_job):
        assert gvp_job["status"] == "completed"

    def test_input_params_echo(self, gvp_job):
        params = gvp_job.get("input_params") or {}
        assert params.get("num_samples") == 3
        assert params.get("model_variant") == "gvp_conditional"

    def test_sdf_output_present(self, client, gvp_job):
        job_id = gvp_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        sdf_files = [f for f in files if f.endswith(".sdf")]
        assert sdf_files, f"no SDF files in output: {files}"


# ===================================================================
# Section 5: Sync inference — EGNN variant (verifies alt ckpts load)
# ===================================================================


@pytest.mark.fc
class TestSyncEGNN:
    def test_job_completed(self, egnn_job):
        assert egnn_job["status"] == "completed"

    def test_input_params_echo(self, egnn_job):
        params = egnn_job.get("input_params") or {}
        assert params.get("model_variant") == "egnn_conditional"


# ===================================================================
# Section 6: Job lifecycle on the shared GVP job
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, gvp_job):
        job_id = gvp_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == job_id
        assert body["status"] == "completed"

    def test_files_endpoint(self, client, gvp_job):
        job_id = gvp_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/files")
        assert r.status_code == 200
        assert any(f.endswith(".sdf") for f in r.json()["files"])

    def test_single_file_download_sdf(self, client, gvp_job):
        job_id = gvp_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        sdf = next(f for f in files if f.endswith(".sdf"))
        r = _retry_get(client, f"/api/jobs/{job_id}/file/{sdf}")
        assert r.status_code == 200
        assert len(r.content) > 20

    def test_job_log_endpoint(self, client, gvp_job):
        job_id = gvp_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0

    def test_job_download_zip(self, client, gvp_job):
        job_id = gvp_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any(n.endswith(".sdf") for n in zf.namelist())

    def test_job_file_not_found(self, client, gvp_job):
        job_id = gvp_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/file/nonexistent.xyz")
        assert r.status_code == 404

"""FC integration tests for turbohopp-server (opt-in, sync submit/poll path).

Run only when FC tests are enabled::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/turbohopp-server/tests/test_fc.py -v

Fixtures (1a0q_protein.pdb + 1a0q_ligand.sdf) ship in tests/data/ — same
files as diffusion-hopping-server so the two services can be regressed
against the same input for a fair speed comparison.

TurboHopp with 5-40 sampling steps runs in seconds-to-tens-of-seconds per
job on H20 / A10.  Allow 15 min per stage to absorb cold-start weight loads.

turbohopp-server ships with ``max_concurrent_jobs=1``, so 429s from the FC
HTTP gateway are expected under any parallel work; every HTTP call goes
through ``_http_with_retry`` to absorb them.
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "turbohopp-server"
SESSION_HEADER = "bioagent-session-id"

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_PROTEIN = DATA_DIR / "1a0q_protein.pdb"
TEST_LIGAND = DATA_DIR / "1a0q_ligand.sdf"

TIMEOUT = httpx.Timeout(connect=30, read=600, write=600, pool=30)

# Consistency sampling is fast; leave headroom for cold start + weight load.
POLL_TIMEOUT_S = 900
POLL_INTERVAL_S = 15


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
    """Run an HTTP call, retrying on FC's 429 ResourceExhausted.

    turbohopp-server runs with ``max_concurrent_jobs=1`` so even sequential
    GETs occasionally trip the 429 gateway throttle.
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


def _retry_delete(client: httpx.Client, path: str, **kw: Any) -> httpx.Response:
    return _http_with_retry(lambda: client.delete(path, **kw))


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


# ---------------------------------------------------------------------------
# Module-scoped inference fixture — one 5-step generation runs once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generate_job(client: httpx.Client, session_headers: dict[str, str]) -> dict:
    """Minimal scaffold hopping — 3 samples × 5 sampling steps."""
    with open(TEST_PROTEIN, "rb") as fh_p, open(TEST_LIGAND, "rb") as fh_l:
        r = _retry_post(
            client, "/api/generate",
            files={
                "protein": ("1a0q_protein.pdb", fh_p.read(), "chemical/x-pdb"),
                "reference_ligand": (
                    "1a0q_ligand.sdf", fh_l.read(), "chemical/x-mdl-sdfile",
                ),
            },
            data={
                "num_samples": "3",
                "num_sampling_steps": "5",
                "seed": "42",
            },
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
    _assert_completed(final)
    return final


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
        assert body["service"] == "turbohopp"
        assert "version" in body

    def test_healthz_detail(self, client):
        """/healthz/detail reports NAS weights probe."""
        r = _retry_get(client, "/healthz/detail")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "turbohopp"
        assert body["weights_dir"] == "/data/models/turbohopp/checkpoints/v1"
        # If this fails, either the FC NAS mount is missing OR the operator
        # hasn't rsync'd a consistency-model .ckpt yet — see README ## Weights.
        assert body["weights_loaded"] is True, (
            f"NAS weights missing at {body['weights_dir']}: "
            f"files_found={body.get('files_found')}. "
            f"See services/turbohopp-server/README.md ## Weights."
        )
        assert body["files_found"] >= 1
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

    def test_service_specific_model_info(self, client):
        extras = _retry_get(client, "/api/manifest").json()["service_specific"]
        assert extras["model"]["name"] == "TurboHopp"
        assert "consistency" in extras["model"]["task"].lower()

    def test_service_specific_config_tips(self, client):
        extras = _retry_get(client, "/api/manifest").json()["service_specific"]
        tips = extras["config_tips"]
        assert "num_sampling_steps" in tips
        assert "find_best" in tips

    def test_service_specific_uri_schemes(self, client):
        extras = _retry_get(client, "/api/manifest").json()["service_specific"]
        schemes = extras["input_uri_schemes"]
        assert "job://<job_id>/<filename>" in schemes


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
        """Neither upload nor URI → 422."""
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


# ===================================================================
# Section 4: Sync inference — module-scoped fixture
# ===================================================================


@pytest.mark.fc
class TestSyncGenerate:
    def test_job_completed(self, generate_job):
        assert generate_job["status"] == "completed"

    def test_input_params_echo(self, generate_job):
        params = generate_job.get("input_params") or {}
        assert params.get("num_samples") == 3
        assert params.get("num_sampling_steps") == 5
        assert params.get("seed") == 42

    def test_sdf_output_present(self, client, generate_job):
        job_id = generate_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        sdf_files = [f for f in files if f.endswith(".sdf")]
        assert sdf_files, f"no SDF files in output: {files}"


# ===================================================================
# Section 5: Job lifecycle on the shared job
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, generate_job):
        job_id = generate_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == job_id
        assert body["status"] == "completed"

    def test_files_endpoint(self, client, generate_job):
        job_id = generate_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/files")
        assert r.status_code == 200
        assert any(f.endswith(".sdf") for f in r.json()["files"])

    def test_single_file_download_sdf(self, client, generate_job):
        job_id = generate_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        sdf = next(f for f in files if f.endswith(".sdf"))
        r = _retry_get(client, f"/api/jobs/{job_id}/file/{sdf}")
        assert r.status_code == 200
        assert len(r.content) > 20

    def test_job_log_endpoint(self, client, generate_job):
        job_id = generate_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0

    def test_job_download_zip(self, client, generate_job):
        job_id = generate_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any(n.endswith(".sdf") for n in zf.namelist())

    def test_job_file_not_found(self, client, generate_job):
        job_id = generate_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/file/nonexistent.xyz")
        assert r.status_code == 404

"""FC integration tests for rfantibody-server (opt-in, submit/poll path).

Run only when FC tests are enabled::

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/rfantibody-server/tests/test_fc.py -v

RFantibody is a 3-step pipeline:
  * ``/api/rfdiffusion``  — antibody backbone design  (target + framework PDB)
  * ``/api/proteinmpnn``  — CDR sequence design       (1_rfdiffusion.qv input)
  * ``/api/rf2``          — structure prediction      (2_proteinmpnn.qv input)

This file focuses on the **sync** submit/poll path.  Module-scoped fixtures
chain the three stages via ``job://`` URIs so the expensive rfdiffusion
step runs ONCE.  For functional testing of the async path see
``test_fc_task.py`` (preferred — fewer FC instances, dedup, no
HTTP-gateway recycle risk).
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

SERVICE = "rfantibody-server"
SESSION_HEADER = "bioagent-session-id"

DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET_PDB = DATA_DIR / "rsv_site3.pdb"
FRAMEWORK_PDB = DATA_DIR / "hu-4D5-8_Fv.pdb"

# Long read timeout for the submit POST itself (cold start + multipart of
# ~600 KB combined PDBs).  poll_job uses its own short reads.
TIMEOUT = httpx.Timeout(connect=30, read=600, write=600, pool=30)

# RFdiffusion ~3-5 min, ProteinMPNN ~1-3 min, RF2 ~5-10 min.  Allow 30 min
# per stage to absorb cold-start NAS reads of the 4 weight files.
POLL_TIMEOUT_S = 1800
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


def _assert_submitted(body: dict) -> str:
    assert "job_id" in body
    assert body["status"] in ("pending", "running")
    assert body.get("created_at") is not None
    assert isinstance(body.get("input_params"), dict)
    return body["job_id"]


def _assert_completed(body: dict) -> None:
    assert body["status"] == "completed", (
        f"failed: kind={body.get('failure_kind')} summary={body.get('error_summary')!r}"
    )
    assert body.get("started_at") is not None
    assert body.get("completed_at") is not None
    assert body.get("duration_seconds") is not None
    assert body["duration_seconds"] > 0
    assert body.get("output_count", 0) > 0
    assert body.get("output_total_bytes", 0) > 0


def _download_bytes(client: httpx.Client, job_id: str, path: str) -> bytes:
    r = _http_with_retry(lambda: client.get(f"/api/jobs/{job_id}/file/{path}"))
    assert r.status_code == 200, f"download {path} failed: {r.status_code} {r.text!r}"
    return r.content


def _http_with_retry(
    call: Callable[[], httpx.Response],
    *,
    max_attempts: int = 20,
    backoff_s: int = 30,
) -> httpx.Response:
    """Run an HTTP call, retrying on FC's 429 ResourceExhausted.

    The deployed rfantibody-server function has a very tight FC
    concurrent-request budget — even sequential GETs occasionally trip the
    429 gateway throttle.  Project memory:
    ``project_fc_http_polling_unreliable_at_concurrency.md`` (the same
    pattern affects polls under high concurrency).

    On 429, sleep ``backoff_s`` and retry; non-429 responses are returned
    directly without sleep.
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


# ---------------------------------------------------------------------------
# Chained pipeline fixtures (module-scoped — rfdiffusion runs once).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rfdiffusion_job(client: httpx.Client, session_headers: dict[str, str]) -> dict:
    """Submit a minimal rfdiffusion job and poll to completion."""
    with open(TARGET_PDB, "rb") as t, open(FRAMEWORK_PDB, "rb") as f:
        r = _retry_post(
            client, "/api/rfdiffusion",
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
            headers=session_headers,
        )
    assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
    job_id = _assert_submitted(r.json())

    final = poll_job(
        client, "", job_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        extra_headers=session_headers,
    )
    _assert_completed(final)
    return final


@pytest.fixture(scope="module")
def proteinmpnn_job(
    client: httpx.Client,
    session_headers: dict[str, str],
    rfdiffusion_job: dict,
) -> dict:
    """Run proteinmpnn off the rfdiffusion output via job:// URI."""
    rfd_id = rfdiffusion_job["job_id"]
    r = _retry_post(
        client, "/api/proteinmpnn",
        data={
            "input_uri": f"job://{rfd_id}/1_rfdiffusion.qv",
            "seqs_per_struct": "1",
            "deterministic": "true",
        },
        headers=session_headers,
    )
    assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
    job_id = _assert_submitted(r.json())

    final = poll_job(
        client, "", job_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        extra_headers=session_headers,
    )
    _assert_completed(final)
    return final


@pytest.fixture(scope="module")
def rf2_job(
    client: httpx.Client,
    session_headers: dict[str, str],
    proteinmpnn_job: dict,
) -> dict:
    """Run rf2 off the proteinmpnn output via job:// URI."""
    mpnn_id = proteinmpnn_job["job_id"]
    r = _retry_post(
        client, "/api/rf2",
        data={
            "input_uri": f"job://{mpnn_id}/2_proteinmpnn.qv",
            "num_recycles": "2",
        },
        headers=session_headers,
    )
    assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
    job_id = _assert_submitted(r.json())

    final = poll_job(
        client, "", job_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
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
        assert body["service"] == "rfantibody"
        assert "version" in body

    def test_healthz_detail(self, client):
        """Custom /healthz/detail reports NAS-mounted weight presence."""
        r = _retry_get(client, "/healthz/detail")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "rfantibody"
        # Weights externalized to NAS — verify presence + file count.
        # See engineering/decisions/2026-06-26-service-weights-externalization.md.
        assert body["weights_dir"] == "/data/models/rfantibody/weights"
        assert body["weights_loaded"] is True, (
            f"NAS weights missing at {body['weights_dir']}: "
            f"files_found={body.get('files_found')}"
        )
        assert body["files_found"] >= 3, (
            "expected at least 3 weight files (rfdiffusion / proteinmpnn / rf2); "
            f"got {body.get('files_found')}"
        )
        assert body["max_concurrent_jobs"] >= 1
        assert isinstance(body.get("active_jobs"), int)

    def test_openapi_served(self, client):
        r = _retry_get(client, "/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        for p in ("/api/rfdiffusion", "/api/proteinmpnn", "/api/rf2"):
            assert p in spec["paths"], f"missing endpoint in OpenAPI: {p}"


# ===================================================================
# Section 2: Manifest
# ===================================================================


@pytest.mark.fc
class TestManifest:
    def test_sync_endpoints_listed(self, client):
        body = _retry_get(client, "/api/manifest").json()
        paths = {e["path"] for e in body["endpoints"]}
        sync_endpoints = {"/api/rfdiffusion", "/api/proteinmpnn", "/api/rf2"}
        assert sync_endpoints <= paths, (
            f"missing sync endpoints: {sync_endpoints - paths}"
        )
        extras = paths - sync_endpoints
        expected_task = {f"/api/tasks/{s}" for s in
                         ("rfdiffusion", "proteinmpnn", "rf2")}
        assert extras <= expected_task, (
            f"unexpected non-task endpoints: {extras - expected_task}"
        )

    def test_service_specific_weights(self, client):
        extras = _retry_get(client, "/api/manifest").json()["service_specific"]
        weights = extras["weights"]
        assert "rfdiffusion" in weights
        assert "proteinmpnn" in weights
        assert "rf2" in weights

    def test_service_specific_tool_outputs(self, client):
        extras = _retry_get(client, "/api/manifest").json()["service_specific"]
        outputs = extras["tool_outputs"]
        assert outputs["rfdiffusion"].endswith("1_rfdiffusion.qv")
        assert outputs["proteinmpnn"].endswith("2_proteinmpnn.qv")
        assert outputs["rf2"].endswith("3_rf2.qv")

    def test_service_specific_uri_schemes(self, client):
        extras = _retry_get(client, "/api/manifest").json()["service_specific"]
        schemes = extras["input_uri_schemes"]
        assert "job://<job_id>/<filename>" in schemes
        assert "oss://<bucket>/<key>" in schemes

    def test_chaining_tip_mentions_job_uri(self, client):
        extras = _retry_get(client, "/api/manifest").json()["service_specific"]
        assert "job://" in extras["chaining_tip"]


# ===================================================================
# Section 3: Error cases (no real job, quick)
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

    def test_unknown_job_file_404(self, client):
        assert _retry_get(client, "/api/jobs/missing-job-id/file/foo.qv").status_code == 404

    def test_rfdiffusion_rejects_missing_target(self, client):
        """rfdiffusion with neither target upload nor target_uri → 422."""
        with open(FRAMEWORK_PDB, "rb") as f:
            r = _retry_post(
                client, "/api/rfdiffusion",
                files={"framework": (FRAMEWORK_PDB.name, f.read(), "chemical/x-pdb")},
                data={"num_designs": "1"},
            )
        assert r.status_code == 422

    def test_rfdiffusion_rejects_missing_framework(self, client):
        """rfdiffusion with neither framework upload nor framework_uri → 422."""
        with open(TARGET_PDB, "rb") as t:
            r = _retry_post(
                client, "/api/rfdiffusion",
                files={"target": (TARGET_PDB.name, t.read(), "chemical/x-pdb")},
                data={"num_designs": "1"},
            )
        assert r.status_code == 422

    def test_proteinmpnn_rejects_missing_input(self, client):
        r = _retry_post(client, "/api/proteinmpnn", data={"seqs_per_struct": "1"})
        assert r.status_code == 422

    def test_rf2_rejects_missing_input(self, client):
        r = _retry_post(client, "/api/rf2", data={"num_recycles": "2"})
        assert r.status_code == 422

    def test_rfdiffusion_diffuser_t_out_of_range(self, client):
        """diffuser_t=0 < ge=1 → 422 (no need to upload anything)."""
        r = _retry_post(
            client, "/api/rfdiffusion",
            data={
                "target_uri": "file:///nonexistent_target.pdb",
                "framework_uri": "file:///nonexistent_framework.pdb",
                "num_designs": "1", "diffuser_t": "0",
            },
        )
        assert r.status_code == 422


# ===================================================================
# Section 4: Sync pipeline — rfdiffusion → proteinmpnn → rf2
# ===================================================================


@pytest.mark.fc
class TestSyncRfdiffusion:
    def test_job_completed(self, rfdiffusion_job):
        assert rfdiffusion_job["status"] == "completed"

    def test_input_params_echo(self, rfdiffusion_job):
        params = rfdiffusion_job.get("input_params") or {}
        assert params.get("num_designs") == 1
        assert params.get("diffuser_t") == 25
        assert params.get("hotspots") == "T305,T456"

    def test_qv_output(self, client, rfdiffusion_job):
        job_id = rfdiffusion_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        assert any("1_rfdiffusion.qv" in f for f in files), (
            f"1_rfdiffusion.qv missing: {files}"
        )


@pytest.mark.fc
class TestSyncProteinMPNN:
    def test_job_completed(self, proteinmpnn_job):
        assert proteinmpnn_job["status"] == "completed"

    def test_input_params_echo(self, proteinmpnn_job):
        params = proteinmpnn_job.get("input_params") or {}
        assert params.get("seqs_per_struct") == 1
        assert params.get("deterministic") is True

    def test_qv_output(self, client, proteinmpnn_job):
        job_id = proteinmpnn_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        assert any("2_proteinmpnn.qv" in f for f in files), (
            f"2_proteinmpnn.qv missing: {files}"
        )


@pytest.mark.fc
class TestSyncRF2:
    def test_job_completed(self, rf2_job):
        assert rf2_job["status"] == "completed"

    def test_input_params_echo(self, rf2_job):
        params = rf2_job.get("input_params") or {}
        assert params.get("num_recycles") == 2

    def test_qv_output(self, client, rf2_job):
        job_id = rf2_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        assert any("3_rf2.qv" in f for f in files), (
            f"3_rf2.qv missing: {files}"
        )


# ===================================================================
# Section 5: Job lifecycle (files, download, log, delete) on rfdiffusion job
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_files_endpoint(self, client, rfdiffusion_job):
        job_id = rfdiffusion_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/files")
        assert r.status_code == 200
        files = r.json()["files"]
        assert any("1_rfdiffusion.qv" in f for f in files)

    def test_single_file_download_qv(self, client, rfdiffusion_job):
        job_id = rfdiffusion_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        qv = next(f for f in files if "1_rfdiffusion.qv" in f)
        data = _download_bytes(client, job_id, qv)
        assert len(data) > 100, f".qv unexpectedly small: {len(data)} bytes"

    def test_job_log_endpoint(self, client, rfdiffusion_job):
        job_id = rfdiffusion_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0

    def test_job_download_zip(self, client, rfdiffusion_job):
        job_id = rfdiffusion_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any("1_rfdiffusion.qv" in n for n in names), (
            f"1_rfdiffusion.qv missing from zip: {names}"
        )

    def test_job_file_not_found(self, client, rfdiffusion_job):
        job_id = rfdiffusion_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/file/nonexistent.xyz")
        assert r.status_code == 404

    def test_job_delete(self, client, rf2_job):
        """DELETE the LAST stage's job — rfdiffusion + proteinmpnn must remain
        intact for the other lifecycle tests."""
        job_id = rf2_job["job_id"]
        r = _retry_delete(client, f"/api/jobs/{job_id}")
        assert r.status_code == 200
        assert r.json().get("status") == "deleted"
        assert _retry_get(client, f"/api/jobs/{job_id}").status_code == 404

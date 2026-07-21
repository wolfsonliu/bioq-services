"""FC integration tests for ppiflow-server (opt-in, submit/poll path).

Run only when FC tests are enabled::

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/ppiflow-server/tests/test_fc.py -v

PPIFlow has 5 sync endpoints:
  * ``/api/sample/binder``       — PPI binder design against a target PDB
  * ``/api/sample/antibody``     — heavy + light CDR design over a framework
  * ``/api/sample/nanobody``     — VHH (heavy-only) CDR design over a framework
  * ``/api/sample/monomer``      — unconditional monomer generation
  * ``/api/sample/scaffolding``  — motif scaffolding from a CSV

This file focuses on the **sync** submit/poll path.  For functional testing
prefer ``test_fc_task.py`` (async task mode) — async keeps the FC instance
alive for the whole pipeline, dedups by task id, and spawns fewer parallel
instances under the test load.  Here we run ONE sync inference (binder) to
prove the submit/poll path end-to-end; the other inference modes are
covered by ``test_fc_task.py``.

Scaffolding is opt-in: its motif CSV references PDB files that must already
exist on the FC NAS, so we only verify the endpoint is registered.
"""

from __future__ import annotations

import io
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "ppiflow-server"
SESSION_HEADER = "bioagent-session-id"

DATA_DIR = Path(__file__).resolve().parent / "data"
ANTIGEN_PDB = DATA_DIR / "1IJZ_IL13.pdb"
SCFV_FRAMEWORK_PDB = DATA_DIR / "6nou_scfv_framework.pdb"
NANOBODY_FRAMEWORK_PDB = DATA_DIR / "7eow_nanobody_framework.pdb"

# Long read timeout for the submit POST itself (cold start + multipart of
# ~150 KB PDB).  poll_job uses its own short reads.
TIMEOUT = httpx.Timeout(connect=30, read=600, write=120, pool=30)

# PPIFlow ~5-15 min per call; allow 30 min for cold-start NAS reads.
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
    r = client.get(f"/api/jobs/{job_id}/file/{path}")
    assert r.status_code == 200, f"download {path} failed: {r.status_code} {r.text!r}"
    return r.content


# ---------------------------------------------------------------------------
# Module-scoped inference: one minimal binder job shared across lifecycle tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binder_job(client: httpx.Client, session_headers: dict[str, str]) -> dict:
    """Submit a minimal binder design and poll to completion.

    Used by TestSyncBinder + TestJobLifecycle so we run ONE inference in
    this file (the other 4 endpoints are covered by ``test_fc_task.py``).
    """
    with open(ANTIGEN_PDB, "rb") as fh:
        r = client.post(
            "/api/sample/binder",
            files={"target": (ANTIGEN_PDB.name, fh, "chemical/x-pdb")},
            data={
                "target_chain": "C",
                "binder_chain": "A",
                "specified_hotspots": "C11,C14,C15",
                "samples_min_length": "60",
                "samples_max_length": "70",
                "samples_per_target": "1",
                "name": "fc_sync_binder",
            },
            headers=session_headers,
        )
    assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
    job_id = _assert_submitted(r.json())

    final = poll_job(
        client,
        "",
        job_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
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
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "ppiflow"
        assert "version" in body

    def test_healthz_detail(self, client):
        """Custom /healthz/detail reports NAS-mounted checkpoint presence."""
        r = client.get("/healthz/detail")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "ppiflow"
        # Weights externalized to NAS — verify presence + count.
        # See engineering/decisions/2026-06-26-service-weights-externalization.md.
        assert body["weights_dir"] == "/data/models/ppiflow/checkpoint"
        assert body["weights_loaded"] is True, (
            f"NAS checkpoints missing at {body['weights_dir']}: "
            f"ckpts_found={body.get('ckpts_found')}"
        )
        assert body["ckpts_found"] >= 4, (
            "expected all 4 .ckpt files (binder/antibody/nanobody/monomer); "
            f"got {body.get('ckpts_found')}"
        )
        assert body["max_concurrent_jobs"] >= 1
        assert isinstance(body.get("active_jobs"), int)

    def test_openapi_served(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        for p in (
            "/api/sample/binder",
            "/api/sample/antibody",
            "/api/sample/nanobody",
            "/api/sample/monomer",
            "/api/sample/scaffolding",
        ):
            assert p in spec["paths"], f"missing endpoint in OpenAPI: {p}"


# ===================================================================
# Section 2: Manifest
# ===================================================================


@pytest.mark.fc
class TestManifest:
    def test_sync_endpoints_listed(self, client):
        body = client.get("/api/manifest").json()
        paths = {e["path"] for e in body["endpoints"]}
        sync_endpoints = {
            "/api/sample/binder",
            "/api/sample/antibody",
            "/api/sample/nanobody",
            "/api/sample/monomer",
            "/api/sample/scaffolding",
        }
        assert sync_endpoints <= paths, (
            f"missing sync endpoints: {sync_endpoints - paths}"
        )
        # Task endpoints (optional) — only check they're a subset of expected.
        extras = paths - sync_endpoints
        expected_task = {f"/api/tasks/sample/{m}" for m in
                         ("binder", "antibody", "nanobody", "monomer", "scaffolding")}
        assert extras <= expected_task, (
            f"unexpected non-task endpoints: {extras - expected_task}"
        )

    def test_service_specific_weights(self, client):
        extras = client.get("/api/manifest").json()["service_specific"]
        weights = extras["weights"]
        assert "binder" in weights
        assert "antibody" in weights
        assert "nanobody" in weights
        assert "monomer" in weights

    def test_service_specific_tool_outputs(self, client):
        extras = client.get("/api/manifest").json()["service_specific"]
        outputs = extras["tool_outputs"]
        for mode in ("binder", "antibody", "nanobody", "monomer", "scaffolding"):
            assert mode in outputs, f"missing tool_outputs[{mode}]"

    def test_service_specific_uri_schemes(self, client):
        extras = client.get("/api/manifest").json()["service_specific"]
        schemes = extras["input_uri_schemes"]
        assert "upload" in schemes
        assert any("oss://" in k for k in schemes)
        assert any("file://" in k for k in schemes)

    def test_endpoint_examples_present(self, client):
        body = client.get("/api/manifest").json()
        by_path = {e["path"]: e for e in body["endpoints"]}
        for p in ("/api/sample/binder", "/api/sample/antibody",
                  "/api/sample/nanobody", "/api/sample/monomer",
                  "/api/sample/scaffolding"):
            assert by_path[p]["examples"], f"missing examples for {p}"


# ===================================================================
# Section 3: Error cases (no real job, quick)
# ===================================================================


@pytest.mark.fc
class TestErrors:
    def test_unknown_job_status_404(self, client):
        assert client.get("/api/jobs/missing-job-id").status_code == 404

    def test_unknown_job_files_404(self, client):
        assert client.get("/api/jobs/missing-job-id/files").status_code == 404

    def test_unknown_job_log_404(self, client):
        assert client.get("/api/jobs/missing-job-id/log").status_code == 404

    def test_unknown_job_download_404(self, client):
        assert client.get("/api/jobs/missing-job-id/download").status_code == 404

    def test_unknown_job_file_404(self, client):
        assert client.get("/api/jobs/missing-job-id/file/foo.pdb").status_code == 404

    def test_binder_rejects_missing_target(self, client):
        """Neither upload nor URI for binder target → 422."""
        r = client.post(
            "/api/sample/binder",
            data={
                "target_chain": "C", "binder_chain": "A",
                "samples_min_length": "60", "samples_max_length": "70",
                "samples_per_target": "1", "name": "missing",
            },
        )
        # Framework returns 422 from resolve_input
        assert r.status_code in (400, 422), (
            f"unexpected status: {r.status_code} {r.text!r}"
        )

    def test_antibody_rejects_missing_framework(self, client):
        with open(ANTIGEN_PDB, "rb") as ag:
            r = client.post(
                "/api/sample/antibody",
                files={"antigen": (ANTIGEN_PDB.name, ag, "chemical/x-pdb")},
                data={
                    "antigen_chain": "C", "heavy_chain": "A", "light_chain": "B",
                    "cdr_length": "CDRH1,5-5,CDRH2,5-5,CDRH3,5-5,CDRL1,5-5,CDRL2,3-3,CDRL3,5-5",
                    "samples_per_target": "1", "name": "missing_fw",
                },
            )
        assert r.status_code in (400, 422)


# ===================================================================
# Section 4: Sync binder inference (single shared job)
# ===================================================================


@pytest.mark.fc
class TestSyncBinder:
    def test_job_completed(self, binder_job):
        assert binder_job["status"] == "completed"

    def test_input_params_echo(self, binder_job):
        params = binder_job.get("input_params") or {}
        assert params.get("name") == "fc_sync_binder"
        assert params.get("target_chain") == "C"
        assert params.get("binder_chain") == "A"

    def test_duration_reasonable(self, binder_job):
        d = binder_job["duration_seconds"]
        assert d > 30, f"too fast for real PPIFlow work: {d}s"
        assert d < POLL_TIMEOUT_S

    def test_output_pdb_uses_name_prefix(self, client, binder_job):
        """PPIFlow's binder writes ``output/<name>_<idx>.pdb`` (flat, prefixed).

        NOTE: the adapter's ``tool_outputs`` docstring still says
        ``output/<name>/*.pdb`` for all 5 modes — that's correct for the
        non-binder modes but stale for binder.
        """
        job_id = binder_job["job_id"]
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        pdbs = [f for f in files if f.endswith(".pdb")]
        assert pdbs, f"no .pdb outputs: {files}"
        assert any(f.startswith("fc_sync_binder") for f in pdbs), (
            f"binder PDB outputs should be prefixed with the request name: {pdbs}"
        )


# ===================================================================
# Section 5: Job lifecycle (files, download, log, delete) on binder job
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_files_endpoint_lists_pdb(self, client, binder_job):
        job_id = binder_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/files")
        assert r.status_code == 200
        files = r.json()["files"]
        assert any(f.endswith(".pdb") for f in files), (
            f"no .pdb in outputs: {files}"
        )

    def test_single_file_download_pdb(self, client, binder_job):
        job_id = binder_job["job_id"]
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        pdb = next(f for f in files if f.endswith(".pdb"))
        data = _download_bytes(client, job_id, pdb)
        text = data.decode("utf-8", errors="replace")
        assert "ATOM" in text, "PDB should contain ATOM records"

    def test_job_log_endpoint(self, client, binder_job):
        job_id = binder_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0, "log should be non-empty for a completed job"

    def test_job_download_zip(self, client, binder_job):
        job_id = binder_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        assert "zip" in r.headers.get("content-type", "").lower() or len(r.content) > 100
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any(n.endswith(".pdb") for n in names), (
            f"PDB outputs missing from zip: {names}"
        )

    def test_job_file_not_found(self, client, binder_job):
        job_id = binder_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/file/nonexistent.xyz")
        assert r.status_code == 404

    def test_job_delete(self, client, binder_job):
        """DELETE last — kept at end so other lifecycle tests still see the job."""
        job_id = binder_job["job_id"]
        r = client.delete(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        assert r.json().get("status") == "deleted"
        assert client.get(f"/api/jobs/{job_id}").status_code == 404

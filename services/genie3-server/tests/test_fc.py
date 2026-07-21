"""FC integration tests for genie3-server (opt-in, sync submit/poll path).

Run only when FC tests are enabled::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/genie3-server/tests/test_fc.py -v

Genie3 exposes four generation endpoints:
  * ``/api/generate/unconditional`` — length-only, no dataset upload
  * ``/api/generate/motif``         — dataset zip (problems/ + motifs/)
  * ``/api/generate/binder``        — dataset zip (problems/ + targets/)
  * ``/api/generate``               — freeform YAML config (custom)

This file focuses on the **sync** submit/poll path.  Each endpoint's inference
run is module-scoped so that assertions across multiple tests reuse a single
GPU job.  For functional testing of the async path see ``test_fc_task.py``
(preferred — no HTTP-gateway recycle risk, platform-level dedup).

genie3 is deployed with ``max_concurrent_jobs=1`` so 429s from the FC gateway
are common under any parallel work; every HTTP call goes through
``_http_with_retry`` to absorb them.
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
import yaml

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "genie3-server"
SESSION_HEADER = "bioagent-session-id"

DATA_DIR = Path(__file__).resolve().parent / "data"
MOTIFBENCH = DATA_DIR / "motifbench"
BINDERTEST = DATA_DIR / "binder"

# Long read timeout for POST submits (cold start + multipart upload of the
# largest test payload — binder zip ~29 KB).  poll_job uses its own short reads.
TIMEOUT = httpx.Timeout(connect=30, read=600, write=600, pool=30)

# Genie3 unconditional at n_sample=1, min/max_length=50 → ~2-5 min.  Motif +
# binder + custom variants can run 5-15 min including evaluation steps.
# Allow 30 min per stage to absorb cold-start weight loads.
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


def _build_zip(files: dict[str, Path]) -> bytes:
    """Build an in-memory zip mapping archive paths → on-disk files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, src in files.items():
            zf.write(src, arcname=arcname)
    return buf.getvalue()


def _motif_zip() -> bytes:
    return _build_zip(
        {
            "problems/01_1LDB.json": MOTIFBENCH / "problems" / "01_1LDB.json",
            "motifs/01_1LDB.pdb": MOTIFBENCH / "motifs" / "01_1LDB.pdb",
        }
    )


def _binder_zip() -> bytes:
    return _build_zip(
        {
            "problems/01_bhrf1.json": BINDERTEST / "problems" / "01_bhrf1.json",
            "targets/pdb/01_bhrf1.pdb": BINDERTEST / "targets" / "pdb" / "01_bhrf1.pdb",
            "targets/pdb/01_bhrf1-chain_B.pdb": BINDERTEST / "targets" / "pdb" / "01_bhrf1-chain_B.pdb",
            "targets/fasta/01_bhrf1.fasta": BINDERTEST / "targets" / "fasta" / "01_bhrf1.fasta",
            "targets/fasta/01_bhrf1-chain_B.fasta": BINDERTEST / "targets" / "fasta" / "01_bhrf1-chain_B.fasta",
            "targets/msa/01_bhrf1.a3m": BINDERTEST / "targets" / "msa" / "01_bhrf1.a3m",
            "targets/msa/01_bhrf1-chain_B.a3m": BINDERTEST / "targets" / "msa" / "01_bhrf1-chain_B.a3m",
        }
    )


def _http_with_retry(
    call: Callable[[], httpx.Response],
    *,
    max_attempts: int = 20,
    backoff_s: int = 30,
) -> httpx.Response:
    """Run an HTTP call, retrying on FC's 429 ResourceExhausted.

    genie3-server runs with ``max_concurrent_jobs=1`` and a very tight FC
    concurrent-request budget, so even sequential GETs occasionally trip the
    429 gateway throttle.  Project memory:
    ``project_fc_http_polling_unreliable_at_concurrency.md``.
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
        f"failed: kind={body.get('failure_kind')} summary={body.get('error_summary')!r}"
    )
    assert body.get("started_at") is not None
    assert body.get("completed_at") is not None
    assert body.get("duration_seconds") is not None
    assert body["duration_seconds"] > 0
    assert body.get("output_count", 0) > 0
    assert body.get("output_total_bytes", 0) > 0


def _download_bytes(client: httpx.Client, job_id: str, path: str) -> bytes:
    r = _retry_get(client, f"/api/jobs/{job_id}/file/{path}")
    assert r.status_code == 200, f"download {path} failed: {r.status_code} {r.text!r}"
    return r.content


# ---------------------------------------------------------------------------
# Module-scoped inference fixtures.  Each endpoint runs ONCE per test module;
# per-endpoint assertions reuse the shared JobInfo dict.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unconditional_job(client: httpx.Client, session_headers: dict[str, str]) -> dict:
    """Minimal unconditional generation — 1 sample, 50-residue monomer."""
    r = _retry_post(
        client, "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
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


@pytest.fixture(scope="module")
def motif_job(client: httpx.Client, session_headers: dict[str, str]) -> dict:
    """Minimal motif scaffolding job — 1 sample from motifbench 01_1LDB."""
    r = _retry_post(
        client, "/api/generate/motif",
        files={"dataset": ("motif.zip", _motif_zip(), "application/zip")},
        data={
            "n_sample": "1",
            "batch_size": "1",
            "selections": "01_1LDB",
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


@pytest.fixture(scope="module")
def binder_job(client: httpx.Client, session_headers: dict[str, str]) -> dict:
    """Minimal binder design job — 1 sample from 01_bhrf1."""
    r = _retry_post(
        client, "/api/generate/binder",
        files={"dataset": ("binder.zip", _binder_zip(), "application/zip")},
        data={
            "n_sample": "1",
            "batch_size": "1",
            "selections": "01_bhrf1",
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


@pytest.fixture(scope="module")
def custom_job(client: httpx.Client, session_headers: dict[str, str]) -> dict:
    """Minimal custom YAML job — server rewrites paths.rootdir + paths.dataset."""
    config = {
        "experiment": {"name": "fc_smoke_custom"},
        "paths": {"rootdir": "PLACEHOLDER_OVERRIDDEN_BY_SERVER"},
        "generation": {
            "dataset": {
                "source": "unconditional",
                "min_length": 50,
                "max_length": 50,
                "length_step": 50,
                "n_sample": 1,
            },
            "sampler": {"sampler": {"direction_scale": 0.8}},
        },
        "evaluation": {"version": "unconditional", "folding": {"model_name": "esmfold"}},
    }
    r = _retry_post(
        client, "/api/generate",
        data={"config_yaml": yaml.safe_dump(config)},
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
        assert body["service"] == "genie3"
        assert "version" in body

    def test_healthz_detail(self, client):
        """Custom /healthz/detail reports NAS-mounted pretrained checkpoint presence."""
        r = _retry_get(client, "/healthz/detail")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "genie3"
        # Weights externalized to NAS — verify presence + file count.
        # See engineering/decisions/2026-06-26-service-weights-externalization.md.
        assert body["pretrained_dir"] == "/data/models/genie3/pretrained/v1"
        assert body["weights_loaded"] is True, (
            f"NAS weights missing at {body['pretrained_dir']}: "
            f"files_found={body.get('files_found')}"
        )
        assert body["files_found"] >= 1, (
            "expected at least 1 pretrained file (checkpoint / config); "
            f"got {body.get('files_found')}"
        )
        assert body["max_concurrent_jobs"] >= 1
        assert isinstance(body.get("active_jobs"), int)

    def test_openapi_served(self, client):
        r = _retry_get(client, "/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        for p in ("/api/generate/unconditional", "/api/generate/motif",
                  "/api/generate/binder", "/api/generate"):
            assert p in spec["paths"], f"missing endpoint in OpenAPI: {p}"


# ===================================================================
# Section 2: Manifest
# ===================================================================


@pytest.mark.fc
class TestManifest:
    def test_sync_endpoints_listed(self, client):
        body = _retry_get(client, "/api/manifest").json()
        paths = {e["path"] for e in body["endpoints"]}
        sync_endpoints = {
            "/api/generate/unconditional",
            "/api/generate/motif",
            "/api/generate/binder",
            "/api/generate",
        }
        assert sync_endpoints <= paths, (
            f"missing sync endpoints: {sync_endpoints - paths}"
        )
        # Any additional endpoints must be task-mode variants.
        extras = paths - sync_endpoints
        expected_task = {
            "/api/tasks/generate/unconditional",
            "/api/tasks/generate/motif",
            "/api/tasks/generate/binder",
            "/api/tasks/generate",
        }
        assert extras <= expected_task, (
            f"unexpected non-task endpoints: {extras - expected_task}"
        )

    def test_service_specific_tool_outputs(self, client):
        extras = _retry_get(client, "/api/manifest").json()["service_specific"]
        assert "tool_outputs" in extras
        assert "*.pdb" in extras["tool_outputs"]["all_modes"]

    def test_service_specific_config_tips(self, client):
        extras = _retry_get(client, "/api/manifest").json()["service_specific"]
        assert "config_tips" in extras
        assert "cond_strategy" in extras["config_tips"]
        assert "direction_scale" in extras["config_tips"]

    def test_service_specific_uri_schemes(self, client):
        extras = _retry_get(client, "/api/manifest").json()["service_specific"]
        assert "input_uri_schemes" in extras

    def test_endpoint_examples(self, client):
        body = _retry_get(client, "/api/manifest").json()
        by_path = {e["path"]: e for e in body["endpoints"]}
        for path in ("/api/generate/unconditional", "/api/generate/motif",
                     "/api/generate/binder", "/api/generate"):
            assert by_path[path]["examples"], f"no examples for {path}"


# ===================================================================
# Section 3: Error cases (fast, no GPU)
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
        assert _retry_get(client, "/api/jobs/missing-job-id/file/foo.pdb").status_code == 404

    def test_422_motif_bad_zip(self, client):
        r = _retry_post(
            client, "/api/generate/motif",
            files={"dataset": ("junk.zip", b"not a zip", "application/zip")},
        )
        assert r.status_code == 422

    def test_422_motif_zip_without_problems(self, client):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("random/file.txt", "x")
        r = _retry_post(
            client, "/api/generate/motif",
            files={"dataset": ("noproblems.zip", buf.getvalue(), "application/zip")},
        )
        assert r.status_code == 422
        assert "problems/" in r.json()["detail"].lower()

    def test_422_binder_bad_zip(self, client):
        r = _retry_post(
            client, "/api/generate/binder",
            files={"dataset": ("bad.zip", b"corrupt", "application/zip")},
        )
        assert r.status_code == 422

    def test_422_custom_invalid_yaml(self, client):
        r = _retry_post(
            client, "/api/generate",
            data={"config_yaml": "{ invalid yaml: ["},
        )
        assert r.status_code == 422

    def test_422_custom_yaml_not_a_dict(self, client):
        r = _retry_post(
            client, "/api/generate",
            data={"config_yaml": "- item1\n- item2\n"},
        )
        assert r.status_code == 422


# ===================================================================
# Section 4: Sync inference — unconditional
# ===================================================================


@pytest.mark.fc
class TestSyncUnconditional:
    def test_job_completed(self, unconditional_job):
        assert unconditional_job["status"] == "completed"

    def test_input_params_echo(self, unconditional_job):
        params = unconditional_job.get("input_params") or {}
        assert params.get("n_sample") == 1
        assert params.get("min_length") == 50
        assert params.get("max_length") == 50

    def test_pdb_output_present(self, client, unconditional_job):
        job_id = unconditional_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        pdb_files = [f for f in files if f.endswith(".pdb")]
        assert pdb_files, f"no PDB files in output: {files}"
        content = _download_bytes(client, job_id, pdb_files[0])
        assert b"ATOM" in content


# ===================================================================
# Section 5: Sync inference — motif scaffolding
# ===================================================================


@pytest.mark.fc
class TestSyncMotif:
    def test_job_completed(self, motif_job):
        assert motif_job["status"] == "completed"

    def test_input_params_echo(self, motif_job):
        params = motif_job.get("input_params") or {}
        assert params.get("selections") == "01_1LDB"
        assert params.get("n_sample") == 1

    def test_pdb_output_present(self, client, motif_job):
        job_id = motif_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        pdb_files = [f for f in files if f.endswith(".pdb")]
        assert pdb_files, f"no PDB files in output: {files}"
        content = _download_bytes(client, job_id, pdb_files[0])
        assert b"ATOM" in content


# ===================================================================
# Section 6: Sync inference — binder design
# ===================================================================


@pytest.mark.fc
class TestSyncBinder:
    def test_job_completed(self, binder_job):
        assert binder_job["status"] == "completed"

    def test_input_params_echo(self, binder_job):
        params = binder_job.get("input_params") or {}
        assert params.get("selections") == "01_bhrf1"
        assert params.get("n_sample") == 1

    def test_pdb_output_present(self, client, binder_job):
        job_id = binder_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        pdb_files = [f for f in files if f.endswith(".pdb")]
        assert pdb_files, f"no PDB files in output: {files}"
        content = _download_bytes(client, job_id, pdb_files[0])
        assert b"ATOM" in content


# ===================================================================
# Section 7: Sync inference — custom YAML
# ===================================================================


@pytest.mark.fc
class TestSyncCustom:
    def test_job_completed(self, custom_job):
        assert custom_job["status"] == "completed"

    def test_input_params_echo(self, custom_job):
        params = custom_job.get("input_params") or {}
        # Custom endpoint records only summary + num_devices.
        assert params.get("config_yaml") == "(user-supplied)"

    def test_pdb_output_present(self, client, custom_job):
        job_id = custom_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        pdb_files = [f for f in files if f.endswith(".pdb")]
        assert pdb_files, f"no PDB files in output: {files}"


# ===================================================================
# Section 8: Job lifecycle (files, download, log, delete) on unconditional_job
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, unconditional_job):
        job_id = unconditional_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == job_id
        assert body["status"] == "completed"

    def test_files_endpoint(self, client, unconditional_job):
        job_id = unconditional_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/files")
        assert r.status_code == 200
        files = r.json()["files"]
        assert any(f.endswith(".pdb") for f in files)

    def test_single_file_download_pdb(self, client, unconditional_job):
        job_id = unconditional_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        pdb = next(f for f in files if f.endswith(".pdb"))
        data = _download_bytes(client, job_id, pdb)
        assert b"ATOM" in data

    def test_job_log_endpoint(self, client, unconditional_job):
        job_id = unconditional_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0

    def test_job_download_zip(self, client, unconditional_job):
        job_id = unconditional_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        assert "application/zip" in r.headers.get("content-type", "")
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any(n.endswith(".pdb") for n in names)

    def test_job_file_not_found(self, client, unconditional_job):
        job_id = unconditional_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/file/nonexistent.xyz")
        assert r.status_code == 404

    def test_job_delete(self, client, custom_job):
        """DELETE the custom_job — the other jobs stay intact for their tests."""
        job_id = custom_job["job_id"]
        r = _retry_delete(client, f"/api/jobs/{job_id}")
        assert r.status_code == 200
        assert r.json().get("status") == "deleted"
        assert _retry_get(client, f"/api/jobs/{job_id}").status_code == 404

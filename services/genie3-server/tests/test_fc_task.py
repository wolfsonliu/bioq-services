"""FC async task mode tests for genie3-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/genie3-server/tests/test_fc_task.py -v

Validates the 4 ``/api/tasks/generate/...`` endpoints (unconditional / motif /
binder / custom) end-to-end against the deployed FC function in async task
mode (``X-Fc-Invocation-Type: Async``).

Async task mode pins the FC instance for the whole job (no 30s HTTP-gateway
recycle risk) and dedups by ``X-Fc-Async-Task-Id`` at the platform layer.

Payload sizing
--------------
FC's async invocation gateway caps the inbound event payload at 128 KiB.
All genie3 test datasets fit comfortably under that cap after zip
compression:

  * motif fixture (01_1LDB)   → ~3.5 KB zip
  * binder fixture (01_bhrf1) → ~29 KB zip
  * custom YAML               → few KB

So we can multipart-upload every fixture directly — no sync-bootstrap needed.
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest
import yaml

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "genie3-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
MOTIFBENCH = DATA_DIR / "motifbench"
BINDERTEST = DATA_DIR / "binder"

# Genie3 unconditional at n_sample=1, length=50 takes ~2-5 min on H20; motif /
# binder / custom variants can reach 5-15 min including evaluation.  Allow
# 30 min per stage to absorb cold-start weight loads.
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


# One task_id per endpoint — reused by that endpoint's poll fixture + all its
# per-endpoint assertions.
@pytest.fixture(scope="module")
def unconditional_task_id() -> str:
    return f"fc-async-uncond-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def motif_task_id() -> str:
    return f"fc-async-motif-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def binder_task_id() -> str:
    return f"fc-async-binder-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def custom_task_id() -> str:
    return f"fc-async-custom-{int(time.time())}-{uuid.uuid4().hex[:6]}"


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
    genie3-server runs with ``max_concurrent_jobs=1`` so even sequential GETs
    can trip the 429 window.
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
    # ``max_concurrent_jobs=1`` here → GET /api/jobs/<id> can 429 for
    # 4-7 min at a stretch.  Bump ``max_transient_errors`` well above the
    # framework default (10) so poll_job rides out throttle windows.
    # Effective retry buffer: 60 × 20s = 20 min of consecutive 429s.
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


def _build_zip(files: dict[str, Path]) -> bytes:
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


def _custom_yaml() -> str:
    return yaml.safe_dump(
        {
            "experiment": {"name": "fc_async_custom"},
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
    )


# ---------------------------------------------------------------------------
# Per-endpoint submit + poll fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unconditional_submit_response(
    client: httpx.Client,
    unconditional_task_id: str,
) -> httpx.Response:
    return client.post(
        "/api/tasks/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
        },
        headers=_async_headers(unconditional_task_id),
    )


@pytest.fixture(scope="module")
def unconditional_task(
    client: httpx.Client,
    unconditional_task_id: str,
    unconditional_submit_response: httpx.Response,
) -> dict:
    assert unconditional_submit_response.status_code == 202, (
        f"async unconditional submit returned "
        f"{unconditional_submit_response.status_code}: "
        f"{unconditional_submit_response.text!r}"
    )
    return _poll_to_completion(client, unconditional_task_id)


@pytest.fixture(scope="module")
def motif_submit_response(
    client: httpx.Client,
    motif_task_id: str,
) -> httpx.Response:
    return client.post(
        "/api/tasks/generate/motif",
        files={"dataset": ("motif.zip", _motif_zip(), "application/zip")},
        data={
            "n_sample": "1",
            "batch_size": "1",
            "selections": "01_1LDB",
        },
        headers=_async_headers(motif_task_id),
    )


@pytest.fixture(scope="module")
def motif_task(
    client: httpx.Client,
    motif_task_id: str,
    motif_submit_response: httpx.Response,
) -> dict:
    assert motif_submit_response.status_code == 202, (
        f"async motif submit returned {motif_submit_response.status_code}: "
        f"{motif_submit_response.text!r}"
    )
    return _poll_to_completion(client, motif_task_id)


@pytest.fixture(scope="module")
def binder_submit_response(
    client: httpx.Client,
    binder_task_id: str,
) -> httpx.Response:
    return client.post(
        "/api/tasks/generate/binder",
        files={"dataset": ("binder.zip", _binder_zip(), "application/zip")},
        data={
            "n_sample": "1",
            "batch_size": "1",
            "selections": "01_bhrf1",
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
def custom_submit_response(
    client: httpx.Client,
    custom_task_id: str,
) -> httpx.Response:
    return client.post(
        "/api/tasks/generate",
        data={"config_yaml": _custom_yaml()},
        headers=_async_headers(custom_task_id),
    )


@pytest.fixture(scope="module")
def custom_task(
    client: httpx.Client,
    custom_task_id: str,
    custom_submit_response: httpx.Response,
) -> dict:
    assert custom_submit_response.status_code == 202, (
        f"async custom submit returned {custom_submit_response.status_code}: "
        f"{custom_submit_response.text!r}"
    )
    return _poll_to_completion(client, custom_task_id)


# ===================================================================
# Section 1: Submit semantics + OpenAPI registration
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_unconditional_returns_202(self, unconditional_submit_response):
        assert unconditional_submit_response.status_code == 202, (
            f"expected 202; got {unconditional_submit_response.status_code} "
            f"body={unconditional_submit_response.text!r}"
        )

    def test_motif_returns_202(self, motif_submit_response):
        assert motif_submit_response.status_code == 202, (
            f"expected 202; got {motif_submit_response.status_code} "
            f"body={motif_submit_response.text!r}"
        )

    def test_binder_returns_202(self, binder_submit_response):
        assert binder_submit_response.status_code == 202, (
            f"expected 202; got {binder_submit_response.status_code} "
            f"body={binder_submit_response.text!r}"
        )

    def test_custom_returns_202(self, custom_submit_response):
        assert custom_submit_response.status_code == 202, (
            f"expected 202; got {custom_submit_response.status_code} "
            f"body={custom_submit_response.text!r}"
        )

    def test_task_endpoints_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200, (
            f"openapi.json fetch failed: {r.status_code} {r.text!r}"
        )
        spec = r.json()
        expected = {
            "/api/tasks/generate/unconditional",
            "/api/tasks/generate/motif",
            "/api/tasks/generate/binder",
            "/api/tasks/generate",
        }
        missing = expected - set(spec["paths"])
        assert not missing, (
            f"task endpoints missing from OpenAPI: {missing}; "
            f"settings.task_endpoints_enabled may be False"
        )


# ===================================================================
# Section 2: Per-endpoint completion + outputs
# ===================================================================


def _assert_completed_with_pdb(
    task: dict,
    task_id: str,
    client: httpx.Client,
    *,
    min_duration_s: float = 3.0,
) -> list[str]:
    """Assert a completed task produced at least one .pdb file.

    ``min_duration_s`` is a low-bar sanity check that the subprocess actually
    ran (defaults to 3s; slower endpoints pass a larger value).  Returns the
    list of output paths so callers can perform additional assertions.
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
    assert any(f.endswith(".pdb") for f in files), (
        f"no .pdb files in outputs: {files}"
    )
    return files


@pytest.mark.fc
class TestAsyncUnconditional:
    def test_completed(self, unconditional_task, unconditional_task_id, client):
        _assert_completed_with_pdb(
            unconditional_task, unconditional_task_id, client, min_duration_s=30,
        )

    def test_input_params_echoed(self, unconditional_task):
        params = unconditional_task.get("input_params") or {}
        assert params.get("n_sample") == 1
        assert params.get("min_length") == 50
        assert params.get("max_length") == 50

    def test_pdb_downloadable(self, client, unconditional_task_id, unconditional_task):
        files = _get_with_retry(
            client, f"/api/jobs/{unconditional_task_id}/files"
        ).json()["files"]
        pdb = next(f for f in files if f.endswith(".pdb"))
        r = _get_with_retry(client, f"/api/jobs/{unconditional_task_id}/file/{pdb}")
        assert r.status_code == 200
        assert b"ATOM" in r.content, (
            f"{pdb} does not contain ATOM records"
        )


@pytest.mark.fc
class TestAsyncMotif:
    def test_completed(self, motif_task, motif_task_id, client):
        _assert_completed_with_pdb(
            motif_task, motif_task_id, client, min_duration_s=30,
        )

    def test_input_params_echoed(self, motif_task):
        params = motif_task.get("input_params") or {}
        assert params.get("selections") == "01_1LDB"
        assert params.get("n_sample") == 1

    def test_pdb_downloadable(self, client, motif_task_id, motif_task):
        files = _get_with_retry(
            client, f"/api/jobs/{motif_task_id}/files"
        ).json()["files"]
        pdb = next(f for f in files if f.endswith(".pdb"))
        r = _get_with_retry(client, f"/api/jobs/{motif_task_id}/file/{pdb}")
        assert r.status_code == 200
        assert b"ATOM" in r.content


@pytest.mark.fc
class TestAsyncBinder:
    def test_completed(self, binder_task, binder_task_id, client):
        _assert_completed_with_pdb(
            binder_task, binder_task_id, client, min_duration_s=30,
        )

    def test_input_params_echoed(self, binder_task):
        params = binder_task.get("input_params") or {}
        assert params.get("selections") == "01_bhrf1"
        assert params.get("n_sample") == 1

    def test_pdb_downloadable(self, client, binder_task_id, binder_task):
        files = _get_with_retry(
            client, f"/api/jobs/{binder_task_id}/files"
        ).json()["files"]
        pdb = next(f for f in files if f.endswith(".pdb"))
        r = _get_with_retry(client, f"/api/jobs/{binder_task_id}/file/{pdb}")
        assert r.status_code == 200
        assert b"ATOM" in r.content


@pytest.mark.fc
class TestAsyncCustom:
    def test_completed(self, custom_task, custom_task_id, client):
        _assert_completed_with_pdb(
            custom_task, custom_task_id, client, min_duration_s=30,
        )

    def test_input_params_echoed(self, custom_task):
        # Custom endpoint records a small echo model with num_devices +
        # ``config_yaml_summary``.  n_sample / min_length live inside the user
        # YAML and are not surfaced by the echo model.
        params = custom_task.get("input_params") or {}
        assert "config_yaml_summary" in params, (
            f"custom echo params missing config_yaml_summary: {params}"
        )

    def test_pdb_downloadable(self, client, custom_task_id, custom_task):
        files = _get_with_retry(
            client, f"/api/jobs/{custom_task_id}/files"
        ).json()["files"]
        pdb = next(f for f in files if f.endswith(".pdb"))
        r = _get_with_retry(client, f"/api/jobs/{custom_task_id}/file/{pdb}")
        assert r.status_code == 200
        assert b"ATOM" in r.content


# ===================================================================
# Section 3: Job lifecycle on the unconditional task (cheapest fixture)
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, unconditional_task_id, unconditional_task):
        r = _get_with_retry(client, f"/api/jobs/{unconditional_task_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == unconditional_task_id
        assert body["status"] == "completed"

    def test_log_endpoint(self, client, unconditional_task_id, unconditional_task):
        r = _get_with_retry(client, f"/api/jobs/{unconditional_task_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0, "log should be non-empty for a completed job"

    def test_download_zip(self, client, unconditional_task_id, unconditional_task):
        r = _get_with_retry(client, f"/api/jobs/{unconditional_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any(n.endswith(".pdb") for n in names), (
            f"no .pdb in zip: {names}"
        )


# ===================================================================
# Section 4: Duplicate dedup — FC platform layer rejects repeat task_id
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
        unconditional_task_id: str,
        unconditional_task: dict,
    ):
        first_created_at = unconditional_task["created_at"]
        first_completed_at = unconditional_task["completed_at"]
        first_n_sample = (unconditional_task.get("input_params") or {}).get("n_sample")

        # Resubmit same task_id with different n_sample to prove dedup.
        r2 = client.post(
            "/api/tasks/generate/unconditional",
            data={
                "n_sample": "2",  # different from first run's 1
                "batch_size": "1",
                "min_length": "50",
                "max_length": "50",
                "length_step": "50",
            },
            headers=_async_headers(unconditional_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → framework "
            f"dedups); got {r2.status_code} body={r2.text!r}"
        )

        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(client, f"/api/jobs/{unconditional_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("n_sample") == first_n_sample, (
            "duplicate async submit must not overwrite input_params"
        )

"""FC async task mode tests for promera-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/promera-server/tests/test_fc_task.py -v

Validates the `/api/tasks/cofold` and `/api/tasks/design` endpoints end-to-end
against the deployed FC function in async task mode
(`X-Fc-Invocation-Type: Async`).

Async task mode means the HTTP request to FC returns HTTP 202 immediately;
the server-side `execute_task` blocks the function instance for the entire
pipeline lifetime so FC won't recycle mid-run.  The framework persists
JobInfo to NAS at each state transition, so we observe progress via
`GET /api/jobs/{task_id}`.

Two long-running module-scoped runs (cofold + design) plus a duplicate-dedup
check that piggybacks on the cofold result.  Each minimal call uses 1 sample
/ 1 backbone / 50 diffusion steps to keep runtime within a few minutes per
endpoint.

After long polling runs FC's HTTP gateway sometimes returns 429 on follow-up
GETs (see project memory `project_fc_http_polling_unreliable_at_concurrency`),
so auxiliary status/files/download requests go through `_get_with_retry`.
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "promera-server"
DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_TARGET = DATA_DIR / "test_target.json"

# Minimal-cost knobs: 1 sample / 1 backbone / 50 steps.  Real pipelines
# default to 5 samples / 10 backbones / 200 steps, so this is ~50× faster.
COFOLD_MIN_PARAMS = {
    "num_seeds": "1",
    "diffusion_samples": "1",
    "diffusion_steps": "50",
}
DESIGN_MIN_PARAMS = {
    "design_type": "minibinder",
    "num_backbones": "1",
    "diffusion_steps": "50",
    "inverse_folder_type": "none",
}

# Promera's diffusion pipelines fit comfortably within 30 min for these
# minimal params; give a 45 min ceiling for cold-start + queue.
POLL_TIMEOUT_S = 2700
POLL_INTERVAL_S = 20

TIMEOUT = httpx.Timeout(connect=30, read=120, write=60, pool=30)


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
def cofold_task_id() -> str:
    return f"fc-async-cofold-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def design_task_id() -> str:
    return f"fc-async-design-{int(time.time())}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_submit(
    client: httpx.Client,
    endpoint: str,
    *,
    task_id: str,
    file_field: str,
    file_path: Path,
    file_mime: str = "application/json",
    **form_fields: str,
) -> httpx.Response:
    """POST to a task endpoint with FC async task headers."""
    with open(file_path, "rb") as fh:
        return client.post(
            endpoint,
            data=form_fields,
            files={file_field: (file_path.name, fh.read(), file_mime)},
            headers={
                "X-Fc-Invocation-Type": "Async",
                "X-Bioagent-Job-Id": task_id,
                "X-Fc-Async-Task-Id": task_id,
            },
        )


def _get_with_retry(
    client: httpx.Client,
    path: str,
    *,
    max_attempts: int = 8,
    backoff_s: int = 15,
) -> httpx.Response:
    """GET that retries on FC's HTTP-gateway 429 throttling.

    After a long-running async task, the FC HTTP gateway can rate-limit
    subsequent GETs to `/api/jobs/...`.  This is a platform-layer
    artifact, not a promera-server bug — see project memory
    `project_fc_http_polling_unreliable_at_concurrency.md`.  Production
    code should use FCDispatcher.get_status (the FC GetAsyncTask API);
    here we just retry the HTTP call so the test still observes the
    underlying state.
    """
    last: httpx.Response | None = None
    for attempt in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


# ---------------------------------------------------------------------------
# Module-scoped runs: one submit per task endpoint.
# Each `*_submit_response` is the raw 202 response; each `*_task` is the
# final JobInfo after polling to completion.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cofold_submit_response(client: httpx.Client, cofold_task_id: str) -> httpx.Response:
    return _async_submit(
        client,
        "/api/tasks/cofold",
        task_id=cofold_task_id,
        file_field="input_schema",
        file_path=TEST_TARGET,
        **COFOLD_MIN_PARAMS,
    )


@pytest.fixture(scope="module")
def cofold_task(
    client: httpx.Client,
    cofold_task_id: str,
    cofold_submit_response: httpx.Response,
) -> dict:
    assert cofold_submit_response.status_code == 202, (
        f"cofold async submit returned {cofold_submit_response.status_code}: "
        f"{cofold_submit_response.text!r}.  Async task mode must be enabled in "
        f"the FC console for this function."
    )
    final = poll_job(
        client,
        "",
        cofold_task_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
    )
    assert final["status"] == "completed", f"cofold did not complete: {final}"
    return final


@pytest.fixture(scope="module")
def design_submit_response(client: httpx.Client, design_task_id: str) -> httpx.Response:
    return _async_submit(
        client,
        "/api/tasks/design",
        task_id=design_task_id,
        file_field="target_schema",
        file_path=TEST_TARGET,
        **DESIGN_MIN_PARAMS,
    )


@pytest.fixture(scope="module")
def design_task(
    client: httpx.Client,
    design_task_id: str,
    design_submit_response: httpx.Response,
) -> dict:
    assert design_submit_response.status_code == 202, (
        f"design async submit returned {design_submit_response.status_code}: "
        f"{design_submit_response.text!r}.  Async task mode must be enabled in "
        f"the FC console for this function."
    )
    final = poll_job(
        client,
        "",
        design_task_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
    )
    assert final["status"] == "completed", f"design did not complete: {final}"
    return final


# ===================================================================
# Section 1: Submit semantics — proves FC async task mode is wired up.
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_cofold_returns_202(self, cofold_submit_response):
        """FC's async invocation of /api/tasks/cofold must return HTTP 202."""
        assert cofold_submit_response.status_code == 202, (
            f"expected 202; got {cofold_submit_response.status_code} "
            f"body={cofold_submit_response.text!r}"
        )

    def test_design_returns_202(self, design_submit_response):
        """FC's async invocation of /api/tasks/design must return HTTP 202."""
        assert design_submit_response.status_code == 202, (
            f"expected 202; got {design_submit_response.status_code} "
            f"body={design_submit_response.text!r}"
        )

    def test_task_endpoints_registered_in_openapi(self, client):
        spec = client.get("/openapi.json").json()
        paths = spec.get("paths", {})
        assert "/api/tasks/cofold" in paths, (
            "/api/tasks/cofold missing from OpenAPI; settings.task_endpoints_enabled "
            "may be False on the deployed function"
        )
        assert "/api/tasks/design" in paths, (
            "/api/tasks/design missing from OpenAPI; settings.task_endpoints_enabled "
            "may be False on the deployed function"
        )


# ===================================================================
# Section 2: Cofold lifecycle — task ran inside the FC instance to completion.
# ===================================================================


@pytest.mark.fc
class TestAsyncCofoldRunsToCompletion:
    def test_completed_status(self, cofold_task):
        assert cofold_task["status"] == "completed"

    def test_started_and_completed_timestamps(self, cofold_task):
        assert cofold_task.get("started_at") is not None
        assert cofold_task.get("completed_at") is not None

    def test_duration_recorded(self, cofold_task):
        d = cofold_task.get("duration_seconds")
        assert d is not None and d > 0
        # Even minimal cofold needs ~20s of model loading + inference;
        # <10s means the subprocess did not actually run.
        assert d > 10, f"duration {d}s too short for real cofold work"

    def test_outputs_present(self, cofold_task):
        assert cofold_task.get("output_count", 0) > 0
        assert cofold_task.get("output_total_bytes", 0) > 0

    def test_input_params_echoed(self, cofold_task):
        params = cofold_task.get("input_params") or {}
        assert params.get("num_seeds") == 1
        assert params.get("diffusion_samples") == 1
        assert params.get("diffusion_steps") == 50


# ===================================================================
# Section 3: Cofold outputs — subprocess produced predicted structure CIF(s).
# ===================================================================


@pytest.mark.fc
class TestAsyncCofoldOutputs:
    def test_files_listing_includes_cif(self, client, cofold_task_id, cofold_task):
        r = _get_with_retry(client, f"/api/jobs/{cofold_task_id}/files")
        assert r.status_code == 200, f"files GET failed: {r.status_code} {r.text!r}"
        files = r.json()["files"]
        assert any(f.endswith(".cif") for f in files), (
            f"no .cif in outputs: {files}"
        )

    def test_cif_downloadable(self, client, cofold_task_id, cofold_task):
        r = _get_with_retry(client, f"/api/jobs/{cofold_task_id}/files")
        assert r.status_code == 200
        cifs = [f for f in r.json()["files"] if f.endswith(".cif")]
        assert cifs, "no .cif files in output"
        download = _get_with_retry(client, f"/api/jobs/{cofold_task_id}/file/{cifs[0]}")
        assert download.status_code == 200
        text = download.content.decode("utf-8", errors="replace")
        assert "_atom_site" in text, "CIF should contain _atom_site loop"

    def test_download_zip(self, client, cofold_task_id, cofold_task):
        r = _get_with_retry(client, f"/api/jobs/{cofold_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any(n.endswith(".cif") for n in names), (
            f".cif missing from zip: {names}"
        )


# ===================================================================
# Section 4: Design lifecycle — task ran inside the FC instance to completion.
# ===================================================================


@pytest.mark.fc
class TestAsyncDesignRunsToCompletion:
    def test_completed_status(self, design_task):
        assert design_task["status"] == "completed"

    def test_started_and_completed_timestamps(self, design_task):
        assert design_task.get("started_at") is not None
        assert design_task.get("completed_at") is not None

    def test_duration_recorded(self, design_task):
        d = design_task.get("duration_seconds")
        assert d is not None and d > 0
        assert d > 10, f"duration {d}s too short for real design work"

    def test_outputs_present(self, design_task):
        assert design_task.get("output_count", 0) > 0
        assert design_task.get("output_total_bytes", 0) > 0

    def test_input_params_echoed(self, design_task):
        params = design_task.get("input_params") or {}
        assert params.get("design_type") == "minibinder"
        assert params.get("num_backbones") == 1
        assert params.get("diffusion_steps") == 50
        assert params.get("inverse_folder_type") == "none"


# ===================================================================
# Section 5: Design outputs — subprocess produced backbone CIF(s).
# ===================================================================


@pytest.mark.fc
class TestAsyncDesignOutputs:
    def test_files_listing_includes_backbone(self, client, design_task_id, design_task):
        r = _get_with_retry(client, f"/api/jobs/{design_task_id}/files")
        assert r.status_code == 200, f"files GET failed: {r.status_code} {r.text!r}"
        files = r.json()["files"]
        assert any("backbone.cif" in f for f in files), (
            f"backbone.cif missing from outputs: {files}"
        )

    def test_backbone_downloadable(self, client, design_task_id, design_task):
        r = _get_with_retry(client, f"/api/jobs/{design_task_id}/files")
        assert r.status_code == 200
        bb = [f for f in r.json()["files"] if "backbone.cif" in f]
        assert bb, "no backbone.cif in output"
        download = _get_with_retry(client, f"/api/jobs/{design_task_id}/file/{bb[0]}")
        assert download.status_code == 200
        text = download.content.decode("utf-8", errors="replace")
        assert "_atom_site" in text, "backbone CIF should contain _atom_site loop"


# ===================================================================
# Section 6: Identity — X-Bioagent-Job-Id propagates to JobInfo.job_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncTaskIdentity:
    def test_cofold_job_id_matches_task_id(self, cofold_task, cofold_task_id):
        assert cofold_task["job_id"] == cofold_task_id, (
            "task endpoint must use X-Bioagent-Job-Id as the JobInfo.job_id"
        )

    def test_design_job_id_matches_task_id(self, design_task, design_task_id):
        assert design_task["job_id"] == design_task_id

    def test_cofold_visible_via_status_endpoint(self, client, cofold_task_id, cofold_task):
        # cofold_task fixture already polled to completion; confirm a fresh
        # GET sees the same record (rules out polling-only artifacts).
        r = _get_with_retry(client, f"/api/jobs/{cofold_task_id}")
        assert r.status_code == 200, f"status GET failed: {r.status_code} {r.text!r}"
        body = r.json()
        assert body["status"] == "completed"
        assert body["job_id"] == cofold_task_id


# ===================================================================
# Section 7: Duplicate dedup — FC platform layer rejects repeat task_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmitting the same X-Fc-Async-Task-Id after completion.

    Per the FC async task mode contract (engineering/decisions/
    2026-06-17-fc-async-task-mode.md and project memory
    `project_fc_async_dedup_at_platform_layer.md`), FC dedups by
    X-Fc-Async-Task-Id at the platform layer — a duplicate returns 409
    without ever invoking the function.

    If FC's behavior ever changes to forward the duplicate, the framework
    layer (`execute_task`) checks the JobInfo store and returns the existing
    record without re-running.  Either path is acceptable; what must NOT
    happen is a second subprocess run that overwrites the first's outputs.

    We exercise this on /api/tasks/cofold only — the framework code path
    is shared between cofold and design, so testing one suffices.
    """

    def test_duplicate_does_not_rerun(
        self, client: httpx.Client, cofold_task_id: str, cofold_task: dict
    ):
        first_created_at = cofold_task["created_at"]
        first_completed_at = cofold_task["completed_at"]
        first_diffusion_steps = (
            (cofold_task.get("input_params") or {}).get("diffusion_steps")
        )

        # Resubmit with the SAME task_id but a different diffusion_steps to
        # prove the second body wasn't applied.
        r2 = _async_submit(
            client,
            "/api/tasks/cofold",
            task_id=cofold_task_id,
            file_field="input_schema",
            file_path=TEST_TARGET,
            num_seeds="1",
            diffusion_samples="1",
            diffusion_steps="99",  # different from first run's 50
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → framework "
            f"dedups); got {r2.status_code} body={r2.text!r}"
        )

        # If FC forwarded the call, give the framework dedup check a moment.
        if r2.status_code == 202:
            time.sleep(30)

        re_query_resp = _get_with_retry(client, f"/api/jobs/{cofold_task_id}")
        assert re_query_resp.status_code == 200, (
            f"status GET failed: {re_query_resp.status_code} {re_query_resp.text!r}"
        )
        re_query = re_query_resp.json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("diffusion_steps") == first_diffusion_steps, (
            "duplicate async submit must not overwrite input_params with second body"
        )

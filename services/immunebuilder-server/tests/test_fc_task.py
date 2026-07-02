"""FC async task mode tests for immunebuilder-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/immunebuilder-server/tests/test_fc_task.py -v

Validates the 3 ``/api/tasks/predict_*`` endpoints (antibody / nanobody /
tcr) end-to-end against the deployed FC function in async task mode
(``X-Fc-Invocation-Type: Async``).

Async task mode pins one FC instance for the entire pipeline (no 30s HTTP
gateway recycle) and dedups by ``X-Fc-Async-Task-Id`` at the platform
layer.  Each inference has a single module-scoped task fixture so the
test session spawns one instance per endpoint, not per assertion — with
1 GPU concurrent-instance quota this is the only way to fit all three
predictors + lifecycle checks in one pytest run.

All three inputs are form-only (short amino-acid strings), well under FC's
128 KiB async payload cap — no file staging needed.

After long polling runs FC's HTTP gateway sometimes returns 429 on
follow-up GETs (see project memory
``project_fc_http_polling_unreliable_at_concurrency``), so auxiliary
status/files/download requests go through ``_get_with_retry``.
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

SERVICE = "immunebuilder-server"

# ---------------------------------------------------------------------------
# Example sequences — real Ig / TCR domain sequences (same as test_fc.py).
# ---------------------------------------------------------------------------

HEAVY_SEQ = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYT"
    "RYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
LIGHT_SEQ = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPS"
    "RFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)
NANOBODY_SEQ = (
    "QVQLVESGGGLVQAGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAINSGGGSTYY"
    "PDSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAKDGYGGSFDYWGQGTQVTVSS"
)
ALPHA_SEQ = (
    "METLLGVSLVILWLQLARVNSQQGEEDPQALSIQEGENATMNCSYKTSINNLQWYRQNSGR"
    "GLVHLILIRSNEREKHSGRLRVTLDTSKKSSSLLITASRAADTASYFCAVNFGGGKLIFGQGTELSVKP"
)
BETA_SEQ = (
    "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVP"
    "NGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSFSTCSANYGYTFGSGTRLTVVEDL"
)

# ImmuneBuilder is CPU/GPU-mixed and each ensemble+refinement pass runs in
# ~30-90s.  Give a generous ceiling for cold start on the Tesla instance
# type + NAS weight load (~500 MB) + refinement.
POLL_TIMEOUT_S = 1200
POLL_INTERVAL_S = 15

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
def antibody_task_id() -> str:
    return f"fc-async-ab-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def nanobody_task_id() -> str:
    return f"fc-async-nb-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def tcr_task_id() -> str:
    return f"fc-async-tcr-{int(time.time())}-{uuid.uuid4().hex[:6]}"


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
    max_attempts: int = 10,
    backoff_s: int = 20,
) -> httpx.Response:
    """GET that retries on FC HTTP-gateway 429 throttling.

    After a long-running async task, the FC HTTP gateway can rate-limit
    subsequent GETs to ``/api/jobs/...``.  This is a platform-layer
    artifact, not an immunebuilder-server bug — see project memory
    ``project_fc_http_polling_unreliable_at_concurrency.md``.
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
    final = poll_job(
        client,
        "",
        task_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


def _assert_pdb_content(content: bytes) -> None:
    text = content.decode("utf-8", errors="replace")
    assert "ATOM" in text, "PDB file should contain ATOM records"


# ---------------------------------------------------------------------------
# Per-endpoint submit fixtures (module-scoped — one inference per endpoint)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def antibody_submit_response(
    client: httpx.Client, antibody_task_id: str
) -> httpx.Response:
    return client.post(
        "/api/tasks/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "name": "async_ab",
            "save_all_models": "true",
            "numbering_scheme": "imgt",
        },
        headers=_async_headers(antibody_task_id),
    )


@pytest.fixture(scope="module")
def antibody_task(
    client: httpx.Client,
    antibody_task_id: str,
    antibody_submit_response: httpx.Response,
) -> dict:
    assert antibody_submit_response.status_code == 202, (
        f"async antibody submit returned {antibody_submit_response.status_code}: "
        f"{antibody_submit_response.text!r}"
    )
    return _poll_to_completion(client, antibody_task_id)


@pytest.fixture(scope="module")
def nanobody_submit_response(
    client: httpx.Client, nanobody_task_id: str
) -> httpx.Response:
    return client.post(
        "/api/tasks/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "async_nb",
            "save_all_models": "true",
        },
        headers=_async_headers(nanobody_task_id),
    )


@pytest.fixture(scope="module")
def nanobody_task(
    client: httpx.Client,
    nanobody_task_id: str,
    nanobody_submit_response: httpx.Response,
) -> dict:
    assert nanobody_submit_response.status_code == 202, (
        f"async nanobody submit returned {nanobody_submit_response.status_code}: "
        f"{nanobody_submit_response.text!r}"
    )
    return _poll_to_completion(client, nanobody_task_id)


@pytest.fixture(scope="module")
def tcr_submit_response(
    client: httpx.Client, tcr_task_id: str
) -> httpx.Response:
    return client.post(
        "/api/tasks/predict_tcr",
        data={
            "alpha_sequence": ALPHA_SEQ,
            "beta_sequence": BETA_SEQ,
            "name": "async_tcr",
            "save_all_models": "true",
        },
        headers=_async_headers(tcr_task_id),
    )


@pytest.fixture(scope="module")
def tcr_task(
    client: httpx.Client,
    tcr_task_id: str,
    tcr_submit_response: httpx.Response,
) -> dict:
    assert tcr_submit_response.status_code == 202, (
        f"async tcr submit returned {tcr_submit_response.status_code}: "
        f"{tcr_submit_response.text!r}"
    )
    return _poll_to_completion(client, tcr_task_id)


# ===================================================================
# Section 1: Submit semantics + OpenAPI registration
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_antibody_returns_202(self, antibody_submit_response):
        assert antibody_submit_response.status_code == 202, (
            f"expected 202; got {antibody_submit_response.status_code} "
            f"body={antibody_submit_response.text!r}"
        )

    def test_nanobody_returns_202(self, nanobody_submit_response):
        assert nanobody_submit_response.status_code == 202, (
            f"expected 202; got {nanobody_submit_response.status_code} "
            f"body={nanobody_submit_response.text!r}"
        )

    def test_tcr_returns_202(self, tcr_submit_response):
        assert tcr_submit_response.status_code == 202, (
            f"expected 202; got {tcr_submit_response.status_code} "
            f"body={tcr_submit_response.text!r}"
        )

    def test_task_endpoints_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        expected = {
            "/api/tasks/predict_antibody",
            "/api/tasks/predict_nanobody",
            "/api/tasks/predict_tcr",
        }
        missing = expected - set(spec.get("paths", {}))
        assert not missing, (
            f"task endpoints missing from OpenAPI: {missing}; "
            f"settings.task_endpoints_enabled may be False"
        )


# ===================================================================
# Section 2: Per-endpoint completion + outputs.
# ===================================================================


def _assert_completed_with_final_model(
    task: dict, task_id: str, client: httpx.Client
) -> list[str]:
    """Common lifecycle + output assertions.  Returns the file listing."""
    assert task["status"] == "completed"
    assert task["job_id"] == task_id
    assert task.get("started_at") is not None
    assert task.get("completed_at") is not None
    d = task.get("duration_seconds")
    assert d is not None and d > 0, f"duration missing: {d!r}"
    assert task.get("output_count", 0) > 0
    assert task.get("output_total_bytes", 0) > 0

    r = _get_with_retry(client, f"/api/jobs/{task_id}/files")
    assert r.status_code == 200, f"files GET failed: {r.status_code} {r.text!r}"
    files = r.json()["files"]
    assert "final_model.pdb" in files, f"final_model.pdb missing from outputs: {files}"
    return files


@pytest.mark.fc
class TestAsyncAntibody:
    def test_completed(self, antibody_task, antibody_task_id, client):
        files = _assert_completed_with_final_model(antibody_task, antibody_task_id, client)
        # save_all_models=True → ensemble + error estimates persisted.
        assert "rank0_unrefined.pdb" in files, f"rank0_unrefined.pdb missing: {files}"
        assert "error_estimates.npy" in files, f"error_estimates.npy missing: {files}"

    def test_input_params_echoed(self, antibody_task):
        params = antibody_task.get("input_params") or {}
        assert params.get("name") == "async_ab"
        assert params.get("numbering_scheme") == "imgt"
        assert params.get("save_all_models") is True
        assert params.get("heavy_sequence") == HEAVY_SEQ
        assert params.get("light_sequence") == LIGHT_SEQ

    def test_final_model_downloadable(self, client, antibody_task_id, antibody_task):
        r = _get_with_retry(
            client, f"/api/jobs/{antibody_task_id}/file/final_model.pdb"
        )
        assert r.status_code == 200
        _assert_pdb_content(r.content)


@pytest.mark.fc
class TestAsyncNanobody:
    def test_completed(self, nanobody_task, nanobody_task_id, client):
        files = _assert_completed_with_final_model(nanobody_task, nanobody_task_id, client)
        assert "rank0_unrefined.pdb" in files, f"rank0_unrefined.pdb missing: {files}"
        assert "error_estimates.npy" in files, f"error_estimates.npy missing: {files}"

    def test_input_params_echoed(self, nanobody_task):
        params = nanobody_task.get("input_params") or {}
        assert params.get("name") == "async_nb"
        assert params.get("heavy_sequence") == NANOBODY_SEQ

    def test_final_model_downloadable(self, client, nanobody_task_id, nanobody_task):
        r = _get_with_retry(
            client, f"/api/jobs/{nanobody_task_id}/file/final_model.pdb"
        )
        assert r.status_code == 200
        _assert_pdb_content(r.content)


@pytest.mark.fc
class TestAsyncTcr:
    def test_completed(self, tcr_task, tcr_task_id, client):
        files = _assert_completed_with_final_model(tcr_task, tcr_task_id, client)
        assert "rank0_unrefined.pdb" in files, f"rank0_unrefined.pdb missing: {files}"
        assert "error_estimates.npy" in files, f"error_estimates.npy missing: {files}"

    def test_input_params_echoed(self, tcr_task):
        params = tcr_task.get("input_params") or {}
        assert params.get("name") == "async_tcr"
        assert params.get("alpha_sequence") == ALPHA_SEQ
        assert params.get("beta_sequence") == BETA_SEQ

    def test_final_model_downloadable(self, client, tcr_task_id, tcr_task):
        r = _get_with_retry(
            client, f"/api/jobs/{tcr_task_id}/file/final_model.pdb"
        )
        assert r.status_code == 200
        _assert_pdb_content(r.content)


# ===================================================================
# Section 3: Identity — X-Bioagent-Job-Id propagates to JobInfo.job_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncTaskIdentity:
    def test_antibody_job_id_matches_task_id(self, antibody_task, antibody_task_id):
        assert antibody_task["job_id"] == antibody_task_id, (
            "task endpoint must use X-Bioagent-Job-Id as JobInfo.job_id"
        )

    def test_nanobody_job_id_matches_task_id(self, nanobody_task, nanobody_task_id):
        assert nanobody_task["job_id"] == nanobody_task_id

    def test_tcr_job_id_matches_task_id(self, tcr_task, tcr_task_id):
        assert tcr_task["job_id"] == tcr_task_id


# ===================================================================
# Section 4: Job lifecycle on the antibody task (single shared inference).
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_job_visible_via_status_endpoint(
        self, client, antibody_task_id, antibody_task
    ):
        r = _get_with_retry(client, f"/api/jobs/{antibody_task_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == antibody_task_id
        assert body["status"] == "completed"

    def test_job_log_endpoint(self, client, antibody_task_id, antibody_task):
        r = _get_with_retry(client, f"/api/jobs/{antibody_task_id}/log")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == antibody_task_id
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)

    def test_job_download_zip(self, client, antibody_task_id, antibody_task):
        r = _get_with_retry(client, f"/api/jobs/{antibody_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any(n.endswith("final_model.pdb") for n in names), (
            f"final_model.pdb missing from zip: {names}"
        )

    def test_single_file_download_missing_returns_404(
        self, client, antibody_task_id, antibody_task
    ):
        r = _get_with_retry(
            client, f"/api/jobs/{antibody_task_id}/file/nonexistent.xyz"
        )
        assert r.status_code == 404


# ===================================================================
# Section 5: Duplicate dedup — FC platform rejects repeat task_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmit same X-Fc-Async-Task-Id after completion.

    Per the FC async task mode contract (engineering/decisions/
    2026-06-17-fc-async-task-mode.md and project memory
    ``project_fc_async_dedup_at_platform_layer.md``), FC dedups by
    ``X-Fc-Async-Task-Id`` at the platform layer — a duplicate returns 409
    without invoking the function.  If FC forwards anyway, the framework
    layer (``execute_task``) returns the existing JobInfo without re-running.
    """

    def test_duplicate_does_not_rerun(
        self,
        client: httpx.Client,
        tcr_task_id: str,
        tcr_task: dict,
    ):
        first_created_at = tcr_task["created_at"]
        first_completed_at = tcr_task["completed_at"]
        first_name = (tcr_task.get("input_params") or {}).get("name")

        # Resubmit same task_id with a different name.  Neither the second
        # body nor a new completed_at should stick.
        r2 = client.post(
            "/api/tasks/predict_tcr",
            data={
                "alpha_sequence": ALPHA_SEQ,
                "beta_sequence": BETA_SEQ,
                "name": "should_not_apply",
            },
            headers=_async_headers(tcr_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → framework "
            f"dedups); got {r2.status_code} body={r2.text!r}"
        )

        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(client, f"/api/jobs/{tcr_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("name") == first_name, (
            "duplicate async submit must not overwrite input_params"
        )

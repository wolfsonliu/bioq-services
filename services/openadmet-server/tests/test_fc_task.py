"""FC async task mode tests for openadmet-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/openadmet-server/tests/test_fc_task.py -v

Validates ``/api/tasks/predict`` and ``/api/tasks/compare`` end-to-end
under FC async task mode (``X-Fc-Invocation-Type: Async``).  Async task
mode pins the FC instance for the whole inference (no HTTP-gateway 30 s
recycle risk) and dedups by ``X-Fc-Async-Task-Id`` at the platform layer.

Small predict against 3-5 SMILES + one chemprop-chemeleon model takes
~60-120 s (cold-start + torch import + CheMeleon foundation load).
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url

SERVICE = "openadmet-server"

LOSARTAN = "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

# All 6 chemprop-chemeleon models pre-staged on NAS (see design doc §7.2).
# The `TestAllModels` sweep runs one async predict per model to confirm each
# loads + infers + writes an OADMET_PRED_ column.  `pxr-chemeleon-baseline`
# is the one model trained with input_col=OPENADMET_SMILES (the other 5 use
# OPENADMET_CANONICAL_SMILES) — the inline-SMILES alias CSV covers both.
ALL_MODELS = [
    "herg-chemeleon-baseline",
    "cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1",
    "cyp1a2-cyp2d6-cyp3a4-cyp3c9-chemeleon-baseline",
    "microsomal-clearance-chemeleon-v1",
    "permeability-logd-ppb-chemeleon-baseline",
    "pxr-chemeleon-baseline",
]

# Small predict: 60-180 s cold + inference + CheMeleon load.  Buffer to
# 30 min to absorb NAS-mount latency and FC 429 window.
POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 20

TIMEOUT = httpx.Timeout(connect=30, read=300, write=60, pool=30)

DATA_DIR = Path(__file__).resolve().parent / "data"


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
def predict_task_id() -> str:
    return f"fc-async-predict-{int(time.time())}-{uuid.uuid4().hex[:6]}"


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
    """GET that retries on FC 429 throttling (post-long-run gateway limits)."""
    last: httpx.Response | None = None
    for _ in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


def _poll_to_completion(client: httpx.Client, task_id: str) -> dict:
    """Poll ``/api/jobs/<id>`` until terminal, treating HTTP 429 as transient.

    Framework's ``poll_job`` calls ``raise_for_status`` on 429 which makes the
    fixture bomb out under FC gateway rate-limiting (project memory
    ``project_fc_http_polling_unreliable_at_concurrency``).  We implement a
    plain retry loop here that keeps polling on 429 / 5xx / connection error.
    """
    deadline = time.monotonic() + POLL_TIMEOUT_S
    transient = 0
    max_transient = 120  # ~40 min of 20 s intervals of pure 429 tolerated

    while time.monotonic() < deadline:
        try:
            r = client.get(f"/api/jobs/{task_id}")
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
            transient += 1
            if transient > max_transient:
                raise
            time.sleep(POLL_INTERVAL_S)
            continue

        if r.status_code in (429, 500, 502, 503, 504):
            transient += 1
            if transient > max_transient:
                raise RuntimeError(
                    f"transient status {r.status_code} exceeded budget on {task_id}"
                )
            time.sleep(POLL_INTERVAL_S)
            continue

        if r.status_code == 404:
            # Task not yet visible on this instance — normal early on.
            time.sleep(POLL_INTERVAL_S)
            continue

        r.raise_for_status()
        body = r.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            assert body["status"] == "completed", (
                f"task ended in state {body['status']}: {body}"
            )
            return body
        time.sleep(POLL_INTERVAL_S)

    raise RuntimeError(
        f"task {task_id} did not reach terminal state within {POLL_TIMEOUT_S}s"
    )


# ---------------------------------------------------------------------------
# Submit + task fixtures — one predict inference reused across assertions.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def predict_submit_response(
    client: httpx.Client, predict_task_id: str
) -> httpx.Response:
    """Async predict submit — 3 SMILES + herg-chemeleon-baseline model."""
    return client.post(
        "/api/tasks/predict",
        data={
            "input_smiles": f"{LOSARTAN},{ASPIRIN},{CAFFEINE}",
            "model_names": '["herg-chemeleon-baseline"]',
            "accelerator": "gpu",
        },
        headers=_async_headers(predict_task_id),
    )


@pytest.fixture(scope="module")
def predict_task(
    client: httpx.Client,
    predict_task_id: str,
    predict_submit_response: httpx.Response,
) -> dict:
    assert predict_submit_response.status_code == 202, (
        f"async predict submit returned "
        f"{predict_submit_response.status_code}: "
        f"{predict_submit_response.text!r}"
    )
    return _poll_to_completion(client, predict_task_id)


# ===================================================================
# Section 1: Submit semantics + OpenAPI registration
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_predict_returns_202(self, predict_submit_response):
        assert predict_submit_response.status_code == 202, (
            f"expected 202; got {predict_submit_response.status_code} "
            f"body={predict_submit_response.text!r}"
        )

    def test_task_endpoints_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        expected = {"/api/tasks/predict", "/api/tasks/compare"}
        missing = expected - set(paths)
        assert not missing, (
            f"task endpoints missing from OpenAPI: {missing}; "
            "settings.task_endpoints_enabled may be False"
        )


# ===================================================================
# Section 2: Completion + outputs.
# ===================================================================


@pytest.mark.fc
class TestAsyncPredict:
    def test_completed(self, predict_task, predict_task_id):
        task = predict_task
        assert task["status"] == "completed"
        assert task["job_id"] == predict_task_id
        assert task.get("started_at") is not None
        assert task.get("completed_at") is not None
        d = task.get("duration_seconds")
        # CheMeleon load + torch init + inference — even minimal call
        # should exceed 10 s.  Anything shorter suggests subprocess died.
        assert d is not None and d > 10, (
            f"duration {d}s too short — subprocess may have short-circuited"
        )
        assert task.get("output_count", 0) > 0
        assert task.get("output_total_bytes", 0) > 0

    def test_predictions_csv_present(
        self, client, predict_task_id, predict_task
    ):
        r = _get_with_retry(client, f"/api/jobs/{predict_task_id}/files")
        assert r.status_code == 200
        files = r.json()["files"]
        assert any("predictions.csv" in f for f in files), (
            f"predictions.csv missing from outputs: {files}"
        )

    def test_input_params_echoed(self, predict_task):
        params = predict_task.get("input_params") or {}
        assert params.get("model_names") == ["herg-chemeleon-baseline"]
        assert params.get("accelerator") == "gpu"
        assert LOSARTAN in (params.get("input_smiles") or "")

    def test_predictions_csv_downloadable_and_has_pred_col(
        self, client, predict_task_id, predict_task
    ):
        files = _get_with_retry(
            client, f"/api/jobs/{predict_task_id}/files"
        ).json()["files"]
        target = next(f for f in files if "predictions.csv" in f)
        r = _get_with_retry(
            client, f"/api/jobs/{predict_task_id}/file/{target}"
        )
        assert r.status_code == 200
        text = r.content.decode("utf-8", errors="replace")
        lines = text.splitlines()
        assert len(lines) >= 2, f"predictions.csv should have header + ≥ 1 row; got {lines[:5]!r}"
        header = lines[0]
        assert "OADMET_PRED_" in header, (
            f"predictions.csv header missing OADMET_PRED_* column: {header!r}"
        )


# ===================================================================
# Section 3: Identity — X-Bioagent-Job-Id propagates to JobInfo.job_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncTaskIdentity:
    def test_job_id_matches_task_id(self, predict_task, predict_task_id):
        assert predict_task["job_id"] == predict_task_id, (
            "task endpoint must use X-Bioagent-Job-Id as JobInfo.job_id"
        )


# ===================================================================
# Section 4: Job lifecycle endpoints.
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, predict_task_id, predict_task):
        r = _get_with_retry(client, f"/api/jobs/{predict_task_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == predict_task_id
        assert body["status"] == "completed"

    def test_log_endpoint(self, client, predict_task_id, predict_task):
        r = _get_with_retry(client, f"/api/jobs/{predict_task_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0

    def test_download_zip(self, client, predict_task_id, predict_task):
        r = _get_with_retry(client, f"/api/jobs/{predict_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any("predictions.csv" in n for n in names), (
            f"predictions.csv missing from zip: {names}"
        )

    def test_single_file_download_missing_returns_404(
        self, client, predict_task_id, predict_task
    ):
        r = _get_with_retry(
            client, f"/api/jobs/{predict_task_id}/file/nonexistent.xyz"
        )
        assert r.status_code == 404


# ===================================================================
# Section 5: Duplicate dedup — FC platform rejects repeat task_id.
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmit the same X-Fc-Async-Task-Id after completion — must not re-run.

    Per FC platform contract, duplicate X-Fc-Async-Task-Id returns 409 without
    invoking the function.  If FC forwards anyway, framework `execute_task`
    returns existing JobInfo without re-running.  Both are acceptable outcomes.
    """

    def test_duplicate_does_not_rerun(
        self,
        client: httpx.Client,
        predict_task_id: str,
        predict_task: dict,
    ):
        first_created_at = predict_task["created_at"]
        first_completed_at = predict_task["completed_at"]
        first_accelerator = (predict_task.get("input_params") or {}).get("accelerator")

        # Resubmit same task_id with a different accelerator to verify
        # neither the new params nor a fresh completed_at stick.
        r2 = client.post(
            "/api/tasks/predict",
            data={
                "input_smiles": LOSARTAN,
                "model_names": '["herg-chemeleon-baseline"]',
                "accelerator": "cpu",  # different from first run's 'gpu'
            },
            headers=_async_headers(predict_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (framework dedup); "
            f"got {r2.status_code} body={r2.text!r}"
        )

        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(
            client, f"/api/jobs/{predict_task_id}"
        ).json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("accelerator") == (
            first_accelerator
        ), "duplicate async submit must not overwrite input_params"


# ===================================================================
# Section 6: All-models sweep — one async predict per pre-staged model.
# ===================================================================


@pytest.fixture(scope="module", params=ALL_MODELS)
def model_predict_result(request, client: httpx.Client) -> dict:
    """Submit + poll one async predict for each pre-staged model.

    Parametrized over ALL_MODELS, so pytest produces one instance per model.
    Serial (FC GPU concurrency is 1); each inference is ~40-100 s.  Uses the
    3-SMILES inline batch, which writes all input_col aliases so every model's
    data.yaml::input_col resolves regardless of naming convention.
    """
    model = request.param
    task_id = f"fc-allm-{model[:20]}-{int(time.time())}-{uuid.uuid4().hex[:4]}"
    resp = client.post(
        "/api/tasks/predict",
        data={
            "input_smiles": f"{LOSARTAN},{ASPIRIN},{CAFFEINE}",
            "model_names": f'["{model}"]',
            "accelerator": "gpu",
        },
        headers=_async_headers(task_id),
    )
    assert resp.status_code == 202, (
        f"{model}: async submit returned {resp.status_code}: {resp.text!r}"
    )
    job = _poll_to_completion(client, task_id)
    return {"model": model, "task_id": task_id, "job": job}


@pytest.mark.fc
class TestAllModels:
    def test_registry_has_all_expected_models(self, client: httpx.Client):
        """Guard against NAS drift — every ALL_MODELS entry must be registered."""
        body = _get_with_retry(client, "/api/models").json()
        names = {m["name"] for m in body["models"]}
        missing = set(ALL_MODELS) - names
        assert not missing, f"expected models missing from /api/models: {missing}"

    def test_model_completes_with_prediction_column(
        self, client: httpx.Client, model_predict_result: dict
    ):
        model = model_predict_result["model"]
        task_id = model_predict_result["task_id"]
        job = model_predict_result["job"]

        assert job["status"] == "completed", f"{model}: {job}"
        assert job.get("output_count", 0) > 0, f"{model}: no outputs"

        r = _get_with_retry(client, f"/api/jobs/{task_id}/file/predictions.csv")
        assert r.status_code == 200, f"{model}: predictions.csv {r.status_code}"
        lines = r.content.decode("utf-8", "replace").splitlines()
        assert len(lines) >= 2, f"{model}: predictions.csv has no data rows: {lines[:3]}"
        header = lines[0]
        assert "OADMET_PRED_" in header, (
            f"{model}: predictions.csv header has no OADMET_PRED_ column: {header!r}"
        )
        # Each of the 3 input SMILES should yield a data row.
        assert len(lines) - 1 >= 3, (
            f"{model}: expected >=3 prediction rows, got {len(lines) - 1}"
        )

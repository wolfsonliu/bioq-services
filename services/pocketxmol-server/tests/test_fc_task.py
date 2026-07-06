"""FC async task-mode integration tests for pocketxmol-server (opt-in).

Marked ``@pytest.mark.fc``, skipped by default.  Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/pocketxmol-server/tests/test_fc_task.py -v

Validates ``/api/tasks/<name>`` endpoints end-to-end under FC async task
mode (``X-Fc-Invocation-Type: Async``).  Async task mode pins the FC
instance for the whole job (no 30 s HTTP-gateway recycle) and dedups by
``X-Fc-Async-Task-Id`` at the platform layer.
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

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "pocketxmol-server"
DATA_DIR = Path(__file__).resolve().parent / "data"

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


@pytest.fixture(scope="module")
def dock_task_id() -> str:
    return f"fc-async-dock-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def sbdd_task_id() -> str:
    return f"fc-async-sbdd-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def pepdesign_task_id() -> str:
    return f"fc-async-pepdesign-{int(time.time())}-{uuid.uuid4().hex[:6]}"


def _async_headers(task_id: str) -> dict[str, str]:
    return {
        "X-Fc-Invocation-Type": "Async",
        "X-Bioagent-Job-Id": task_id,
        "X-Fc-Async-Task-Id": task_id,
    }


def _get_with_retry(
    call: Callable[[], httpx.Response],
    *,
    max_attempts: int = 20,
    backoff_s: int = 30,
) -> httpx.Response:
    last: httpx.Response | None = None
    for _ in range(max_attempts):
        last = call()
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


def _retry_get(client: httpx.Client, path: str, **kw: Any) -> httpx.Response:
    return _get_with_retry(lambda: client.get(path, **kw))


def _poll_to_completion(client: httpx.Client, task_id: str) -> dict:
    final = poll_job(
        client, "", task_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        max_transient_errors=60,
    )
    assert final["status"] == "completed", f"task not completed: {final}"
    return final


# ---------------------------------------------------------------------------
# Per-endpoint submit fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dock_submit_resp(client, dock_task_id):
    with open(DATA_DIR / "8C7Y_TXV_protein.pdb", "rb") as fp, \
            open(DATA_DIR / "8C7Y_TXV_ligand_start_conf.sdf", "rb") as fl:
        return client.post(
            "/api/tasks/dock",
            files={
                "protein": ("protein.pdb", fp.read(), "chemical/x-pdb"),
                "ligand": ("ligand.sdf", fl.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "num_samples": "3",
                "batch_size": "3",
                "pocket_coord": "[-8.257, 85.181, 19.050]",
                "pocket_radius": "15",
            },
            headers=_async_headers(dock_task_id),
        )


@pytest.fixture(scope="module")
def dock_task(client, dock_task_id, dock_submit_resp) -> dict:
    assert dock_submit_resp.status_code == 202, (
        f"async dock submit returned {dock_submit_resp.status_code}: "
        f"{dock_submit_resp.text!r}"
    )
    return _poll_to_completion(client, dock_task_id)


@pytest.fixture(scope="module")
def sbdd_submit_resp(client, sbdd_task_id):
    with open(DATA_DIR / "2ar9_A.pdb", "rb") as fp:
        return client.post(
            "/api/tasks/sbdd",
            files={"protein": ("protein.pdb", fp.read(), "chemical/x-pdb")},
            data={
                "num_samples": "5",
                "batch_size": "5",
                "pocket_coord": "[-8.1603, 36.6972, 38.7714]",
                "pocket_radius": "15",
                "mode": "simple",
            },
            headers=_async_headers(sbdd_task_id),
        )


@pytest.fixture(scope="module")
def sbdd_task(client, sbdd_task_id, sbdd_submit_resp) -> dict:
    assert sbdd_submit_resp.status_code == 202, (
        f"async sbdd submit returned {sbdd_submit_resp.status_code}: "
        f"{sbdd_submit_resp.text!r}"
    )
    return _poll_to_completion(client, sbdd_task_id)


@pytest.fixture(scope="module")
def pepdesign_submit_resp(client, pepdesign_task_id):
    with open(DATA_DIR / "3bik_A.pdb", "rb") as fp, \
            open(DATA_DIR / "3bik_A_pocket_coord.sdf", "rb") as fr:
        return client.post(
            "/api/tasks/pepdesign",
            files={
                "protein": ("protein.pdb", fp.read(), "chemical/x-pdb"),
                "ref_ligand": ("ref.sdf", fr.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "mode": "denovo_linear",
                "pep_length": "5",
                "num_samples": "3",
                "batch_size": "3",
                "pocket_radius": "20",
            },
            headers=_async_headers(pepdesign_task_id),
        )


@pytest.fixture(scope="module")
def pepdesign_task(client, pepdesign_task_id, pepdesign_submit_resp) -> dict:
    assert pepdesign_submit_resp.status_code == 202, (
        f"async pepdesign submit returned {pepdesign_submit_resp.status_code}: "
        f"{pepdesign_submit_resp.text!r}"
    )
    return _poll_to_completion(client, pepdesign_task_id)


pytestmark = pytest.mark.fc


# ===========================================================================
# Section 1: submit semantics + OpenAPI
# ===========================================================================
class TestAsyncSubmit:
    def test_dock_returns_202(self, dock_submit_resp):
        assert dock_submit_resp.status_code == 202

    def test_sbdd_returns_202(self, sbdd_submit_resp):
        assert sbdd_submit_resp.status_code == 202

    def test_pepdesign_returns_202(self, pepdesign_submit_resp):
        assert pepdesign_submit_resp.status_code == 202

    def test_task_endpoints_registered_in_openapi(self, client):
        r = _retry_get(client, "/openapi.json")
        assert r.status_code == 200
        expected = {
            "/api/tasks/dock", "/api/tasks/sbdd", "/api/tasks/linking",
            "/api/tasks/optimize", "/api/tasks/pepdesign",
            "/api/tasks/confidence",
        }
        missing = expected - set(r.json()["paths"])
        assert not missing, (
            f"task endpoints missing: {missing}; "
            f"settings.task_endpoints_enabled may be False"
        )


# ===========================================================================
# Section 2: Completion + output per endpoint
# ===========================================================================
def _assert_task_completed(task: dict, task_id: str, client: httpx.Client,
                           *, min_duration_s: float = 5.0):
    assert task["status"] == "completed"
    assert task["job_id"] == task_id
    d = task.get("duration_seconds")
    assert d is not None and d > min_duration_s, (
        f"duration {d}s too short (min {min_duration_s}s) — did subprocess run?"
    )
    assert task.get("output_count", 0) > 0


class TestAsyncDock:
    def test_completed(self, dock_task, dock_task_id, client):
        _assert_task_completed(dock_task, dock_task_id, client)
        files = _retry_get(client, f"/api/jobs/{dock_task_id}/files").json()["files"]
        assert any(f.endswith(".sdf") for f in files)


class TestAsyncSbdd:
    def test_completed(self, sbdd_task, sbdd_task_id, client):
        _assert_task_completed(sbdd_task, sbdd_task_id, client)
        files = _retry_get(client, f"/api/jobs/{sbdd_task_id}/files").json()["files"]
        assert any(f.endswith(".sdf") for f in files)


class TestAsyncPepDesign:
    def test_completed(self, pepdesign_task, pepdesign_task_id, client):
        _assert_task_completed(pepdesign_task, pepdesign_task_id, client)
        files = _retry_get(client, f"/api/jobs/{pepdesign_task_id}/files").json()["files"]
        assert any(f.endswith(".pdb") or f.endswith(".sdf") for f in files)


# ===========================================================================
# Section 3: Lifecycle (using cheapest fixture)
# ===========================================================================
class TestJobLifecycle:
    def test_status_endpoint(self, sbdd_task, sbdd_task_id, client):
        body = _retry_get(client, f"/api/jobs/{sbdd_task_id}").json()
        assert body["status"] == "completed"

    def test_log_endpoint(self, sbdd_task, sbdd_task_id, client):
        r = _retry_get(client, f"/api/jobs/{sbdd_task_id}/log")
        assert r.status_code == 200

    def test_download_zip(self, sbdd_task, sbdd_task_id, client):
        r = _retry_get(client, f"/api/jobs/{sbdd_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any(n.endswith(".sdf") for n in zf.namelist())


# ===========================================================================
# Section 4: Platform-layer dedup
# ===========================================================================
class TestAsyncDuplicateDedup:
    """Re-submit same X-Fc-Async-Task-Id → FC platform layer 409 (won't
    hit function) or framework layer 202 returning existing JobInfo.
    Either way, the job must not re-run."""

    def test_duplicate_does_not_rerun(self, client, sbdd_task_id, sbdd_task):
        first_created = sbdd_task["created_at"]
        first_completed = sbdd_task["completed_at"]

        with open(DATA_DIR / "2ar9_A.pdb", "rb") as fp:
            r2 = client.post(
                "/api/tasks/sbdd",
                files={"protein": ("protein.pdb", fp.read(), "chemical/x-pdb")},
                data={
                    "num_samples": "50",  # different value — must not take effect
                    "pocket_coord": "[0, 0, 0]",  # different — must not take effect
                    "pocket_radius": "15",
                    "mode": "simple",
                },
                headers=_async_headers(sbdd_task_id),
            )
        assert r2.status_code in (202, 409), (
            f"expected 409 or 202; got {r2.status_code} {r2.text!r}"
        )
        if r2.status_code == 202:
            time.sleep(15)

        re_query = _retry_get(client, f"/api/jobs/{sbdd_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created, "created_at was reset"
        assert re_query["completed_at"] == first_completed, "task was re-run"

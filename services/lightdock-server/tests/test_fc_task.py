"""FC async task mode tests for lightdock-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/lightdock-server/tests/test_fc_task.py -v

Validates /api/tasks/dock under FC async task mode (``X-Fc-Invocation-Type:
Async``). The 1czy protein-peptide fixture (receptor 104 KB + ligand 4 KB ≈
108 KB total) stays under the 128 KiB async task-mode payload limit, so inputs
upload directly via multipart — no NAS bootstrap / URI indirection needed.
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

SERVICE = "lightdock-server"
DATA_DIR = Path(__file__).resolve().parent / "data"
RECEPTOR = DATA_DIR / "receptor.pdb"
LIGAND = DATA_DIR / "ligand.pdb"

POLL_TIMEOUT_S = 3600
POLL_INTERVAL_S = 20
TIMEOUT = httpx.Timeout(connect=30, read=300, write=600, pool=30)

TINY = {"swarms": "2", "glowworms": "5", "steps": "3", "top": "3"}


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


def _async_headers(task_id: str) -> dict[str, str]:
    return {
        "X-Fc-Invocation-Type": "Async",
        "X-Bioagent-Job-Id": task_id,
        "X-Fc-Async-Task-Id": task_id,
    }


def _files() -> dict:
    return {
        "receptor": ("receptor.pdb", RECEPTOR.read_bytes(), "chemical/x-pdb"),
        "ligand": ("ligand.pdb", LIGAND.read_bytes(), "chemical/x-pdb"),
    }


def _get_with_retry(client, path, *, max_attempts=20, backoff_s=30):
    last = None
    for _ in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    return last


def _poll(client, task_id: str) -> dict:
    final = poll_job(
        client, "", task_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S, max_transient_errors=60,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


@pytest.fixture(scope="module")
def task_id() -> str:
    return f"fc-async-dock-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def dock_submit(client, task_id):
    return client.post(
        "/api/tasks/dock",
        files=_files(),
        data=TINY,
        headers=_async_headers(task_id),
    )


@pytest.fixture(scope="module")
def dock_task(client, task_id, dock_submit) -> dict:
    assert dock_submit.status_code == 202, (
        f"async submit returned {dock_submit.status_code}: {dock_submit.text!r}"
    )
    return _poll(client, task_id)


# ===================================================================
# Section 1: submit semantics + OpenAPI
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_returns_202(self, dock_submit):
        assert dock_submit.status_code == 202

    def test_task_endpoint_registered(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200
        assert "/api/tasks/dock" in r.json()["paths"]


# ===================================================================
# Section 2: completion + output
# ===================================================================


@pytest.mark.fc
class TestAsyncDock:
    def test_completed(self, dock_task, task_id):
        assert dock_task["status"] == "completed"
        assert dock_task["job_id"] == task_id
        assert dock_task.get("output_count", 0) > 0

    def test_output_downloadable(self, client, task_id, dock_task):
        files = _get_with_retry(client, f"/api/jobs/{task_id}/files").json()["files"]
        assert any("top_1.pdb" in f for f in files), files


# ===================================================================
# Section 3: lifecycle
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status(self, client, task_id, dock_task):
        body = _get_with_retry(client, f"/api/jobs/{task_id}").json()
        assert body["status"] == "completed"

    def test_download_zip(self, client, task_id, dock_task):
        r = _get_with_retry(client, f"/api/jobs/{task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any("top_1.pdb" in n for n in zf.namelist()), zf.namelist()


# ===================================================================
# Section 4: platform-layer dedup
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    def test_duplicate_does_not_rerun(self, client, task_id, dock_task):
        first_created = dock_task["created_at"]
        first_completed = dock_task["completed_at"]
        r2 = client.post(
            "/api/tasks/dock",
            files=_files(),
            data={**TINY, "top": "1"},  # different — must not take effect
            headers=_async_headers(task_id),
        )
        assert r2.status_code in (202, 409), f"got {r2.status_code} {r2.text!r}"
        if r2.status_code == 202:
            time.sleep(15)
        re_query = _get_with_retry(client, f"/api/jobs/{task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created
        assert re_query["completed_at"] == first_completed

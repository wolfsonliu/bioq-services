"""FC async task mode tests for lasermpnn-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    pytest -m fc services/lasermpnn-server/tests/test_fc_task.py -v

Validates POST /api/tasks/design in FC async task mode (X-Fc-Invocation-Type:
Async). The input PDB is passed as a file:// URI to the example vendored into
the image (/opt/lasermpnn/LASErMPNN/example_pdbs/), so the async event payload
stays well under FC's 128 KiB cap (see project memory feedback_fc_async_payload_128kib).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "lasermpnn-server"

EXAMPLE_PDB = "file:///opt/lasermpnn/LASErMPNN/example_pdbs/4jnj-1_prot.pdb"

POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 15
TIMEOUT = httpx.Timeout(connect=30, read=120, write=60, pool=30)


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


def _get_with_retry(client, path, *, max_attempts=10, backoff_s=20):
    last = None
    for _ in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


@pytest.fixture(scope="module")
def design_task_id() -> str:
    return f"fc-async-design-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def design_submit(client, design_task_id) -> httpx.Response:
    return client.post(
        "/api/tasks/design",
        data={
            "designs_per_input": "1",
            "designs_per_batch": "1",
            "sequence_temp": "0.3",
            "pdb_uri": EXAMPLE_PDB,
        },
        headers=_async_headers(design_task_id),
    )


@pytest.fixture(scope="module")
def design_task(client, design_task_id, design_submit) -> dict:
    assert design_submit.status_code == 202, (
        f"async submit returned {design_submit.status_code}: {design_submit.text!r}"
    )
    final = poll_job(
        client, "", design_task_id, timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
    )
    assert final["status"] == "completed", final
    return final


@pytest.mark.fc
class TestAsyncDesign:
    def test_returns_202(self, design_submit):
        assert design_submit.status_code == 202, design_submit.text

    def test_task_endpoints_in_openapi(self, client):
        spec = _get_with_retry(client, "/openapi.json").json()
        assert "/api/tasks/design" in spec["paths"]
        assert "/api/tasks/design_ligandmpnn" in spec["paths"]

    def test_completed_with_outputs(self, client, design_task, design_task_id):
        assert design_task["job_id"] == design_task_id
        assert design_task.get("output_count", 0) > 0
        files = _get_with_retry(
            client, f"/api/jobs/{design_task_id}/files",
        ).json()["files"]
        assert any(n.endswith(".pdb") for n in files), files

    def test_duplicate_does_not_rerun(self, client, design_task, design_task_id):
        first_created = design_task["created_at"]
        first_completed = design_task["completed_at"]
        r2 = client.post(
            "/api/tasks/design",
            data={
                "designs_per_input": "1",
                "designs_per_batch": "1",
                "pdb_uri": EXAMPLE_PDB,
            },
            headers=_async_headers(design_task_id),
        )
        assert r2.status_code in (202, 409), r2.text
        if r2.status_code == 202:
            time.sleep(10)
        re_q = _get_with_retry(client, f"/api/jobs/{design_task_id}").json()
        assert re_q["created_at"] == first_created
        assert re_q["completed_at"] == first_completed

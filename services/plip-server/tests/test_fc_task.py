"""FC async task-mode tests for plip-server (`/api/tasks/profile`).

Marked `@pytest.mark.fc`, skipped by default. Run with:

    RUN_FC_TESTS=1 pytest services/plip-server/tests/test_fc_task.py

Exercises the blocking task endpoint via the framework's job store. URL is read
from `services/services.yaml` via `bioagent_service.fc_testing`.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
PDB = DATA_DIR / "1vsn.pdb"

pytestmark = pytest.mark.fc

TIMEOUT_S = 1800


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("plip-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


def test_profile_task_end_to_end(client: httpx.Client, base_url: str) -> None:
    job_id = "plip-" + uuid.uuid4().hex[:16]
    with open(PDB, "rb") as f:
        r = client.post(
            "/api/tasks/profile",
            headers={
                "X-Fc-Invocation-Type": "Async",
                "X-Bioagent-Job-Id": job_id,
                "X-Fc-Async-Task-Id": job_id,
            },
            files={"input_pdb": (PDB.name, f, "chemical/x-pdb")},
            data={"name": "fc_task"},
        )
    assert r.status_code in (200, 202), r.text

    final = poll_job(client, base_url, job_id, timeout_s=TIMEOUT_S)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any(f.endswith("fc_task.xml") for f in files), files

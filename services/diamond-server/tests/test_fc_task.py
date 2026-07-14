"""FC async task-mode tests for diamond-server (`/api/tasks/*`).

Marked `@pytest.mark.fc`, skipped by default. Run with:

    RUN_FC_TESTS=1 pytest services/diamond-server/tests/test_fc_task.py

Exercises the blocking task endpoints via the framework's job store. URL is read
from `services/services.yaml` via `bioagent_service.fc_testing`.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
QUERY = DATA_DIR / "query.faa"
SUBJECT = DATA_DIR / "subject.faa"

pytestmark = pytest.mark.fc

TIMEOUT_S = 1800


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("diamond-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


def test_blastp_task_end_to_end(client: httpx.Client, base_url: str) -> None:
    job_id = "diamond-" + uuid.uuid4().hex[:16]
    with open(QUERY, "rb") as fq, open(SUBJECT, "rb") as fs:
        r = client.post(
            "/api/tasks/blastp",
            headers={
                "X-Fc-Invocation-Type": "Async",
                "X-Bioagent-Job-Id": job_id,
                "X-Fc-Async-Task-Id": job_id,
            },
            files={"query": (QUERY.name, fq, "text/plain"), "subject": (SUBJECT.name, fs, "text/plain")},
            data={"name": "fc_task"},
        )
    assert r.status_code in (200, 202), r.text

    final = poll_job(client, base_url, job_id, timeout_s=TIMEOUT_S)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any(f.endswith("fc_task.tsv") for f in files), files


def test_cluster_task_end_to_end(client: httpx.Client, base_url: str) -> None:
    job_id = "diamond-" + uuid.uuid4().hex[:16]
    with open(SUBJECT, "rb") as fs:
        r = client.post(
            "/api/tasks/cluster",
            headers={
                "X-Fc-Invocation-Type": "Async",
                "X-Bioagent-Job-Id": job_id,
                "X-Fc-Async-Task-Id": job_id,
            },
            files={"sequences": (SUBJECT.name, fs, "text/plain")},
            data={"algorithm": "cluster", "name": "fc_clust"},
        )
    assert r.status_code in (200, 202), r.text

    final = poll_job(client, base_url, job_id, timeout_s=TIMEOUT_S)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any(f.endswith("fc_clust.clusters.tsv") for f in files), files

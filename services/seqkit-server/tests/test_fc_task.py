"""FC async task-mode tests for seqkit-server (`/api/tasks/stats|revcomp`).

Marked `@pytest.mark.fc`, skipped by default. Run with:

    RUN_FC_TESTS=1 pytest services/seqkit-server/tests/test_fc_task.py

Unlike larger-fixture services, the input here is ~280 bytes — far under FC's
128 KiB async invocation payload cap — so the async path uploads the file
directly (no sync bootstrap staging needed).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import pytest
from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "seqkit-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
FASTA = DATA_DIR / "input.fasta"

pytestmark = pytest.mark.fc

POLL_TIMEOUT_S = 300
TIMEOUT = httpx.Timeout(connect=30, read=120, write=60, pool=30)


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


def _task_headers(job_id: str) -> dict:
    return {
        "X-Fc-Invocation-Type": "Async",
        "X-Bioagent-Job-Id": job_id,
        "X-Fc-Async-Task-Id": job_id,
    }


# ---- Smoke ----

def test_healthz(client: httpx.Client) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "seqkit"


# ---- stats task ----

def test_stats_task_end_to_end(client: httpx.Client, base_url: str) -> None:
    job_id = "seqkit-" + uuid.uuid4().hex[:16]
    with open(FASTA, "rb") as f:
        r = client.post(
            "/api/tasks/stats",
            headers=_task_headers(job_id),
            files={"input_fasta": (FASTA.name, f.read(), "text/plain")},
        )
    assert r.status_code in (200, 202), r.text

    final = poll_job(client, base_url, job_id, timeout_s=POLL_TIMEOUT_S)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any(f.endswith("stats.tsv") for f in files), files


def test_stats_task_idempotent_replay(client: httpx.Client, base_url: str) -> None:
    """Same job id twice → the second call returns the existing job, no re-run."""
    job_id = "seqkit-idem-" + uuid.uuid4().hex[:12]
    payload = FASTA.read_bytes()

    r1 = client.post(
        "/api/tasks/stats",
        headers=_task_headers(job_id),
        files={"input_fasta": (FASTA.name, payload, "text/plain")},
    )
    assert r1.status_code in (200, 202), r1.text
    final = poll_job(client, base_url, job_id, timeout_s=POLL_TIMEOUT_S)
    assert final["status"] == "completed", final

    r2 = client.post(
        "/api/tasks/stats",
        headers=_task_headers(job_id),
        files={"input_fasta": (FASTA.name, payload, "text/plain")},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["job_id"] == job_id
    assert r2.json()["status"] == "completed"


# ---- revcomp task ----

def test_revcomp_task_end_to_end(client: httpx.Client, base_url: str) -> None:
    job_id = "seqkit-rc-" + uuid.uuid4().hex[:12]
    with open(FASTA, "rb") as f:
        r = client.post(
            "/api/tasks/revcomp",
            headers=_task_headers(job_id),
            files={"input_fasta": (FASTA.name, f.read(), "text/plain")},
            data={"seq_type": "dna"},
        )
    assert r.status_code in (200, 202), r.text

    final = poll_job(client, base_url, job_id, timeout_s=POLL_TIMEOUT_S)
    assert final["status"] == "completed", final

    out = client.get(f"/api/jobs/{job_id}/file/output/revcomp.fasta")
    out.raise_for_status()
    assert "TCAGGCTAACGGTCAGTTACGCAT" in out.text  # seq1 revcomp, hand-checked

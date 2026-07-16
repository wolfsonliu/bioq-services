"""FC async task-mode tests for plip-server (`/api/tasks/profile`).

Marked `@pytest.mark.fc`, skipped by default. Run with:

    RUN_FC_TESTS=1 pytest services/plip-server/tests/test_fc_task.py

PDB source — sync bootstrap, then ``file://``
---------------------------------------------
FC's async invocation gateway caps the inbound event payload at 128 KiB
(``EntityTooLarge`` 400 otherwise), so we cannot multipart-upload the ~186 KB
``tests/data/1vsn.pdb`` on the async path. The SYNC HTTP path has no such cap
and writes the upload to ``<jobs_base_dir>/<job_id>/input/input.pdb`` on NAS as
a side effect of ``JobRunner.submit`` (build_argv runs synchronously before the
job is queued). So a session-scoped ``staged_pdb_uri`` fixture does one sync
POST to ``/api/profile`` and returns ``file://<that NAS path>``; the async
submit then passes ``input_pdb_uri=<staged_pdb_uri>`` as a form field.

Override the staged URI with ``PLIP_TEST_PDB_NAS_PATH`` if you've pre-staged it
elsewhere on NAS and want to skip the bootstrap call.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "plip-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
PDB = DATA_DIR / "1vsn.pdb"

pytestmark = pytest.mark.fc

# JobsBaseDir on the FC instance — must match settings.jobs_base_dir
# (Dockerfile ``PLIP_JOBS_BASE_DIR``).
JOBS_BASE_DIR_ON_FC = "/data/plip_jobs"

# Optional pre-staged NAS path; skips the bootstrap sync upload when set.
PRESTAGED_PDB_NAS_PATH = os.environ.get("PLIP_TEST_PDB_NAS_PATH")

POLL_TIMEOUT_S = 1800
TIMEOUT = httpx.Timeout(connect=30, read=120, write=60, pool=30)


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture(scope="module")
def staged_pdb_uri(client: httpx.Client) -> str:
    """One-time sync upload that lands the PDB on the FC NAS; returns file:// URI.

    JobRunner.submit saves the upload synchronously (build_argv runs before the
    job is queued), so the file exists on NAS by the time the response returns.
    """
    if PRESTAGED_PDB_NAS_PATH:
        return f"file://{PRESTAGED_PDB_NAS_PATH}"

    with open(PDB, "rb") as fh:
        r = client.post(
            "/api/profile",
            files={"input_pdb": (PDB.name, fh.read(), "chemical/x-pdb")},
            data={"name": "bootstrap"},
        )
    assert r.status_code == 200, f"bootstrap sync upload failed: {r.status_code} {r.text!r}"
    job_id = r.json()["job_id"]
    return f"file://{JOBS_BASE_DIR_ON_FC}/{job_id}/input/input.pdb"


def test_profile_task_end_to_end(client: httpx.Client, base_url: str, staged_pdb_uri: str) -> None:
    job_id = "plip-" + uuid.uuid4().hex[:16]
    r = client.post(
        "/api/tasks/profile",
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": job_id,
            "X-Fc-Async-Task-Id": job_id,
        },
        data={"input_pdb_uri": staged_pdb_uri, "name": "fc_task"},
    )
    assert r.status_code in (200, 202), r.text

    final = poll_job(client, base_url, job_id, timeout_s=POLL_TIMEOUT_S)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any(f.endswith("fc_task.xml") for f in files), files

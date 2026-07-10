"""FC async task mode tests for iggm-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/iggm-server/tests/test_fc_task.py -v

Validates POST /api/tasks/{design,epitope} in FC async task mode
(X-Fc-Invocation-Type: Async).  Inputs are passed as file:// URIs to the
examples vendored into the image (/opt/iggm/examples/), so the async event
payload stays tiny — well under FC's 128 KiB cap (the antigen PDB itself is
~265 KB and could not be multipart-uploaded in async mode; see project memory
feedback_fc_async_payload_128kib).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "iggm-server"

EX_DIR = "/opt/iggm/examples"
AB_FASTA = f"file://{EX_DIR}/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta"
COMPLEX_FASTA = f"file://{EX_DIR}/fasta.files.native/8hpu_M_N_A.fasta"
ANTIGEN_PDB = f"file://{EX_DIR}/pdb.files.native/8hpu_M_N_A.pdb"

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


# ---------------------------------------------------------------------------
# Epitope (fast) — async task submit/poll + dedup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def epitope_task_id() -> str:
    return f"fc-async-epi-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def epitope_submit(client, epitope_task_id) -> httpx.Response:
    return client.post(
        "/api/tasks/epitope",
        data={"fasta_uri": COMPLEX_FASTA, "antigen_uri": ANTIGEN_PDB},
        headers=_async_headers(epitope_task_id),
    )


@pytest.fixture(scope="module")
def epitope_task(client, epitope_task_id, epitope_submit) -> dict:
    assert epitope_submit.status_code == 202, (
        f"async submit returned {epitope_submit.status_code}: {epitope_submit.text!r}"
    )
    final = poll_job(client, "", epitope_task_id, timeout_s=600, interval_s=10)
    assert final["status"] == "completed", final
    return final


@pytest.mark.fc
class TestAsyncEpitope:
    def test_returns_202(self, epitope_submit):
        assert epitope_submit.status_code == 202, epitope_submit.text

    def test_task_endpoints_in_openapi(self, client):
        spec = _get_with_retry(client, "/openapi.json").json()
        assert "/api/tasks/design" in spec["paths"]
        assert "/api/tasks/epitope" in spec["paths"]

    def test_completed_with_epitope_json(self, client, epitope_task, epitope_task_id):
        assert epitope_task["job_id"] == epitope_task_id
        r = _get_with_retry(client, f"/api/jobs/{epitope_task_id}/file/epitope.json")
        assert r.status_code == 200
        assert isinstance(r.json()["epitope"], list)

    def test_duplicate_does_not_rerun(self, client, epitope_task, epitope_task_id):
        first_created = epitope_task["created_at"]
        first_completed = epitope_task["completed_at"]
        r2 = client.post(
            "/api/tasks/epitope",
            data={"fasta_uri": COMPLEX_FASTA, "antigen_uri": ANTIGEN_PDB},
            headers=_async_headers(epitope_task_id),
        )
        assert r2.status_code in (202, 409), r2.text
        if r2.status_code == 202:
            time.sleep(10)
        re_q = _get_with_retry(client, f"/api/jobs/{epitope_task_id}").json()
        assert re_q["created_at"] == first_created
        assert re_q["completed_at"] == first_completed


# ---------------------------------------------------------------------------
# Design (GPU, minutes) — async task submit/poll + outputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def design_task_id() -> str:
    return f"fc-async-design-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def design_submit(client, design_task_id) -> httpx.Response:
    return client.post(
        "/api/tasks/design",
        data={
            "run_task": "design",
            "steps": "5",
            "num_samples": "1",
            "seed": "42",
            "fasta_uri": AB_FASTA,
            "antigen_uri": ANTIGEN_PDB,
        },
        headers=_async_headers(design_task_id),
    )


@pytest.fixture(scope="module")
def design_task(client, design_task_id, design_submit) -> dict:
    assert design_submit.status_code == 202, (
        f"async design submit returned {design_submit.status_code}: {design_submit.text!r}"
    )
    final = poll_job(
        client, "", design_task_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
    )
    assert final["status"] == "completed", final
    return final


@pytest.mark.fc
class TestAsyncDesign:
    def test_returns_202(self, design_submit):
        assert design_submit.status_code == 202, design_submit.text

    def test_completed(self, design_task, design_task_id):
        assert design_task["status"] == "completed"
        assert design_task["job_id"] == design_task_id
        d = design_task.get("duration_seconds")
        assert d is not None and d > 3.0, f"duration {d}s too short"
        assert design_task.get("output_count", 0) > 0

    def test_input_params_echoed(self, design_task):
        params = design_task.get("input_params") or {}
        assert params.get("run_task") == "design"
        assert params.get("steps") == 5
        assert params.get("seed") == 42

    def test_outputs_present(self, client, design_task, design_task_id):
        files = _get_with_retry(
            client, f"/api/jobs/{design_task_id}/files"
        ).json()["files"]
        assert any(n.endswith(".pdb") for n in files), files
        assert any(n.endswith(".fasta") for n in files), files

"""End-to-end tests against the deployed mmseqs2 Function Compute service.

Marked ``@pytest.mark.fc``, skipped by default. Run with::

    pytest -m fc services/mmseqs2-server/tests/test_fc.py

The base URL is read from ``services/aliyun_fc_url.md`` — update that file
after deploying a new tag in the FC console.

Two layered surfaces are exercised:

* **ColabFold protocol** (``/ticket/*`` + ``/result/download/*``) — the
  legacy submit-and-poll path; clients like boltz-server's
  ``--msa_server_url`` and the upstream ColabFold notebooks talk to this.
* **FC async task mode** (``/api/tasks/*``) — newer atomic-task path,
  invoked via ``X-Fc-Invocation-Type: Async`` + ``X-Bioagent-Job-Id``.  Echoes
  the patterns used by boltz-server / immunebuilder-server.

Inference cases use ``mode=env`` (UniRef30 + ColabFoldDB) with a short
ubiquitin-like sequence so MSA finishes within FC's instance lifetime.
"""

from __future__ import annotations

import io
import tarfile
import time
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

pytestmark = pytest.mark.fc

# Small monomer (52 aa) — keeps the MSA quick on the GPU subset DB.
SHORT_MONOMER = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVN"
MONOMER_Q = f">probe1\n{SHORT_MONOMER}\n"

# Two-chain heterodimer for paired tests; both chains short.
PAIRED_Q = (
    f">chainA\n{SHORT_MONOMER}\n"
    f">chainB\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDY\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("mmseqs2-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    # mmseqs2 MSA cold-start ~60s, search ~3-10 min for short seqs.  Match
    # other heavy services' generous read timeout.
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(300.0)) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_submit_and_poll(
    client: httpx.Client,
    base_url: str,
    endpoint: str,
    data: dict,
    *,
    task_id: str,
    timeout_s: int = 2400,
) -> tuple[str, dict, list[str]]:
    """POST as FC async task; poll ``/api/jobs/{task_id}`` until terminal.

    Sends ``X-Fc-Invocation-Type: Async`` + matching X-Bioagent-Job-Id /
    X-Fc-Async-Task-Id so FC enqueues the work and dedups duplicate
    invocations at the platform layer.  Asserts the submit returned 202.

    Returns ``(task_id, final_jobinfo, files)``.
    """
    r = client.post(
        endpoint,
        data=data,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    assert r.status_code == 202, (
        f"expected 202 from async invocation; got {r.status_code} body={r.text!r}.  "
        f"Check that FC console has async task mode enabled for this function."
    )

    final = poll_job(client, base_url, task_id, timeout_s=timeout_s, interval_s=20)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{task_id}/files").json()["files"]
    return task_id, final, files


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "mmseqs2"


def test_healthz_detail_exposes_db_signal(client: httpx.Client) -> None:
    body = client.get("/healthz/detail").json()
    assert body["service"] == "mmseqs2"
    assert "db_loaded" in body
    assert "gpu_free_mb" in body


def test_manifest_lists_all_endpoints(client: httpx.Client) -> None:
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    # ColabFold protocol surface
    assert "/ticket/msa" in paths
    assert "/ticket/pair" in paths
    # FC async task surface (added alongside ColabFold protocol)
    assert "/api/tasks/msa" in paths
    assert "/api/tasks/pair" in paths


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-mmseqs-job").status_code == 404


# ---------------------------------------------------------------------------
# Validation (no GPU work, quick)
# ---------------------------------------------------------------------------


def test_ticket_msa_invalid_mode_returns_protocol_error(client: httpx.Client) -> None:
    """ColabFold protocol: invalid input → 200 + {"status": "ERROR"} (NOT 4xx)."""
    r = client.post("/ticket/msa", data={"q": MONOMER_Q, "mode": "nonsense"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ERROR"
    assert "id" not in body


def test_tasks_msa_invalid_mode_returns_422(client: httpx.Client) -> None:
    """Task endpoints use HTTP 4xx for input errors (distinct from ColabFold)."""
    r = client.post("/api/tasks/msa", data={"q": MONOMER_Q, "mode": "nonsense"})
    assert r.status_code == 422


def test_tasks_msa_paired_mode_rejected(client: httpx.Client) -> None:
    r = client.post(
        "/api/tasks/msa", data={"q": MONOMER_Q, "mode": "pairgreedy"}
    )
    assert r.status_code == 422


def test_tasks_pair_single_chain_rejected(client: httpx.Client) -> None:
    r = client.post(
        "/api/tasks/pair", data={"q": MONOMER_Q, "mode": "pairgreedy"}
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Inference: FC async task mode (the production path)
# ---------------------------------------------------------------------------


def test_async_task_msa_minimal(client: httpx.Client, base_url: str) -> None:
    """Async-invoke /api/tasks/msa, poll JobInfo to completion.

    Validates the full async task pipeline:
      - HTTP 202 on submit (async task mode is on)
      - task_id from X-Bioagent-Job-Id is used as JobInfo.job_id
      - server runs orchestrator synchronously inside the FC instance
      - .a3m files land in the job's output dir
      - JobInfo.input_params does NOT echo the raw query string ``q``
    """
    task_id = f"fc-async-msa-{int(time.time())}"
    task_id, final, files = _async_submit_and_poll(
        client, base_url, "/api/tasks/msa",
        {"q": MONOMER_Q, "mode": "env"},
        task_id=task_id,
    )

    assert final["job_id"] == task_id
    assert final["completed_at"] is not None
    assert final["duration_seconds"] is not None and final["duration_seconds"] > 0

    a3m_files = [f for f in files if f.endswith(".a3m")]
    assert a3m_files, f"no .a3m output in: {files}"

    # Privacy: the raw query sequence must not be echoed in JobInfo.input_params.
    serialized = repr(final.get("input_params"))
    assert SHORT_MONOMER not in serialized
    assert final["input_params"]["mode"] == "env"
    assert final["input_params"]["sequence_count"] == 1


def test_async_task_msa_honors_x_bioagent_job_id(
    client: httpx.Client, base_url: str,
) -> None:
    """X-Bioagent-Job-Id flows end-to-end into JobInfo.job_id."""
    task_id = f"fc-async-id-{int(time.time())}"
    r = client.post(
        "/api/tasks/msa",
        data={"q": MONOMER_Q, "mode": "env"},
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    assert r.status_code == 202

    final = poll_job(client, base_url, task_id, timeout_s=2400, interval_s=20)
    assert final["status"] == "completed", final
    assert final["job_id"] == task_id


def test_async_task_duplicate_rejected_at_fc_platform_layer(
    client: httpx.Client, base_url: str,
) -> None:
    """Same X-Fc-Async-Task-Id twice → FC platform dedups at HTTP 409.

    Mirrors boltz-server's parallel test: FC's async task mode dedupes by
    X-Fc-Async-Task-Id before the function is even invoked, so the second
    submit returns 409 without burning a cold-start.  Server-side framework
    dedup is the fallback for invocation paths that bypass FC (LocalDispatcher,
    direct curl, future K8s backend).
    """
    task_id = f"fc-async-dup-{int(time.time())}"
    payload_first = {"q": MONOMER_Q, "mode": "env"}
    payload_second = {"q": PAIRED_Q, "mode": "all"}  # diff payload; must be ignored

    r1 = client.post(
        "/api/tasks/msa",
        data=payload_first,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    assert r1.status_code == 202

    final = poll_job(client, base_url, task_id, timeout_s=2400, interval_s=20)
    assert final["status"] == "completed", final
    first_created_at = final["created_at"]
    first_seq_count = final["input_params"]["sequence_count"]

    r2 = client.post(
        "/api/tasks/msa",
        data=payload_second,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    # FC platform layer rejects (409) OR accepts (202) and server-side dedup
    # kicks in.  Either is acceptable per
    # engineering/decisions/2026-06-17-fc-async-task-mode.md.
    assert r2.status_code in (202, 409), (
        f"expected 409 (FC dedup) or 202 (server dedups); got {r2.status_code}"
    )
    if r2.status_code == 202:
        time.sleep(30)  # let server-side dedup finalize

    re_query = client.get(f"/api/jobs/{task_id}").json()
    assert re_query["status"] == "completed"
    assert re_query["created_at"] == first_created_at, (
        "duplicate async invoke must not reset created_at"
    )
    assert re_query["input_params"]["sequence_count"] == first_seq_count, (
        "duplicate async invoke must not overwrite original input_params"
    )


# ---------------------------------------------------------------------------
# Inference: ColabFold protocol (sync submit + poll + download)
# ---------------------------------------------------------------------------


def test_ticket_msa_minimal_full_lifecycle(
    client: httpx.Client, base_url: str,
) -> None:
    """Submit via /ticket/msa, poll /ticket/{id} → COMPLETE, then download tarball."""
    r = client.post("/ticket/msa", data={"q": MONOMER_Q, "mode": "env"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "PENDING"
    job_id = body["id"]

    # Poll the ColabFold ticket endpoint — the protocol the upstream client uses.
    deadline = time.monotonic() + 2400
    status = "PENDING"
    while time.monotonic() < deadline:
        s = client.get(f"/ticket/{job_id}").json()
        status = s["status"]
        if status in ("COMPLETE", "ERROR"):
            break
        time.sleep(20)
    assert status == "COMPLETE", f"final ticket status: {status}"

    # Download the .a3m tarball.
    dl = client.get(f"/result/download/{job_id}")
    assert dl.status_code == 200
    with tarfile.open(fileobj=io.BytesIO(dl.content), mode="r:gz") as tf:
        names = tf.getnames()
    assert any(n.endswith(".a3m") for n in names), f"no .a3m in tarball: {names}"

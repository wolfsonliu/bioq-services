"""FC tests for mmseqs2-server — ColabFold-protocol (sync HTTP) surface.

Marked ``@pytest.mark.fc``, skipped by default. Run with::

    pytest -m fc services/mmseqs2-server/tests/test_fc.py

Base URL is read from ``services/aliyun_fc_url.md`` — update that file after
deploying a new tag.

This file covers the **sync** side (ColabFold protocol) of the service:

* ``POST /ticket/msa`` — submit a monomer (unpaired) MSA
* ``GET  /ticket/<id>`` — poll ColabFold-protocol status
* ``GET  /result/download/<id>`` — download the .a3m tarball
* Smoke + manifest + validation (both /ticket/* and /api/tasks/* validation
  paths return quickly and share fixtures cheaply here)

The **async task** (``/api/tasks/*``) surface lives in
:mod:`test_fc_task.py` — patterned after
:mod:`services.rfantibody-server.tests.test_fc_task`. Async task mode has
better FC instance utilization (FC platform manages queueing / affinity)
whereas sync submit + poll needs client-side session-affinity handling to
avoid burning one FC instance per poll.  This file consolidates sync
lifecycle assertions onto a single ``msa_job`` module fixture so we run
exactly ONE MSA computation for the entire sync surface.
"""

from __future__ import annotations

import io
import tarfile
import time
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "mmseqs2-server"
SESSION_HEADER = "bioagent-session-id"

pytestmark = pytest.mark.fc

# Small monomer (52 aa) — keeps the MSA quick on the GPU subset DB.
SHORT_MONOMER = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVN"
MONOMER_Q = f">probe1\n{SHORT_MONOMER}\n"

# Two-chain heterodimer used by /ticket/pair validation cases.
PAIRED_Q = (
    f">chainA\n{SHORT_MONOMER}\n"
    f">chainB\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDY\n"
)

TIMEOUT = httpx.Timeout(connect=30, read=300, write=60, pool=30)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_with_retry(
    client: httpx.Client,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    max_attempts: int = 20,
    backoff_s: int = 30,
) -> httpx.Response:
    """GET that retries on FC HTTP-gateway 429 throttling.

    See project memory ``project_fc_http_polling_unreliable_at_concurrency``.
    """
    last: httpx.Response | None = None
    for _ in range(max_attempts):
        last = client.get(path, headers=headers or {})
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


# ---------------------------------------------------------------------------
# Module-scoped ticket-submit fixture — one MSA reused across all sync tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def msa_job(client: httpx.Client) -> dict:
    """Submit one monomer MSA via /ticket/msa and poll to COMPLETE.

    Uses session affinity end-to-end: the framework's session middleware
    echoes ``job_id`` into the response's ``bioagent-session-id`` header on
    the initial POST, and we send that header on every subsequent GET so FC
    routes polls back to the instance that owns the job (avoids the "one
    instance per poll" fanout that plain-sync tests otherwise cause).

    Returns a dict with ``job_id``, ``session`` (header dict) and
    ``final`` (the last /ticket/<id> body — COMPLETE).  Consumed by
    every ``TestTicketLifecycle`` case, so we pay the MSA cost once.
    """
    r = client.post("/ticket/msa", data={"q": MONOMER_Q, "mode": "env"})
    assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
    body = r.json()
    assert body["status"] == "PENDING", body
    job_id = body["id"]

    # Framework's session middleware echoes job_id as the affinity header
    # value.  Fall back to job_id if for some reason the header wasn't set
    # (e.g. middleware disabled) — session affinity then just becomes best
    # effort rather than mandatory.
    session_id = r.headers.get(SESSION_HEADER, job_id)
    session = {SESSION_HEADER: session_id}

    # Poll the ColabFold-protocol status endpoint until terminal.  Interval
    # is generous (20 s) so we don't hammer FC while an MSA is running.
    deadline = time.monotonic() + 2400
    final_ticket: dict = {}
    while time.monotonic() < deadline:
        s = _get_with_retry(client, f"/ticket/{job_id}", headers=session)
        assert s.status_code == 200, f"/ticket/{job_id} GET failed: {s.status_code} {s.text!r}"
        final_ticket = s.json()
        if final_ticket["status"] in ("COMPLETE", "ERROR"):
            break
        time.sleep(20)
    assert final_ticket.get("status") == "COMPLETE", (
        f"ticket did not complete within budget: {final_ticket}"
    )
    return {"job_id": job_id, "session": session, "final": final_ticket}


# ===================================================================
# Section 1: Smoke — no job submission needed
# ===================================================================


@pytest.mark.fc
class TestSmoke:
    def test_healthz(self, client: httpx.Client) -> None:
        r = _get_with_retry(client, "/healthz")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "mmseqs2"

    def test_healthz_detail_exposes_db_signal(self, client: httpx.Client) -> None:
        r = _get_with_retry(client, "/healthz/detail")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["service"] == "mmseqs2"
        assert "db_loaded" in body
        assert "gpu_free_mb" in body
        assert "active_jobs" in body

    def test_openapi_served(self, client: httpx.Client) -> None:
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200, r.text

    def test_unknown_job_returns_404(self, client: httpx.Client) -> None:
        r = _get_with_retry(client, "/api/jobs/missing-mmseqs-job")
        assert r.status_code == 404


# ===================================================================
# Section 2: Manifest — both surfaces should be advertised
# ===================================================================


@pytest.mark.fc
class TestManifest:
    def test_service_name(self, client: httpx.Client) -> None:
        body = _get_with_retry(client, "/api/manifest").json()
        assert body["service"] == "mmseqs2"

    def test_endpoints_include_both_surfaces(self, client: httpx.Client) -> None:
        paths = {e["path"] for e in _get_with_retry(client, "/api/manifest").json()["endpoints"]}
        # ColabFold-protocol surface
        assert "/ticket/msa" in paths
        assert "/ticket/pair" in paths
        # FC async task surface
        assert "/api/tasks/msa" in paths
        assert "/api/tasks/pair" in paths


# ===================================================================
# Section 3: Validation errors — cheap (no GPU work), covers both surfaces
# ===================================================================


@pytest.mark.fc
class TestTicketValidation:
    """ColabFold protocol: bad input → 200 + ``{"status": "ERROR"}`` (NOT 4xx)."""

    def test_invalid_mode_returns_protocol_error(self, client: httpx.Client) -> None:
        r = client.post("/ticket/msa", data={"q": MONOMER_Q, "mode": "nonsense"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ERROR"
        assert "id" not in body

    def test_empty_q_returns_protocol_error(self, client: httpx.Client) -> None:
        r = client.post("/ticket/msa", data={"q": "", "mode": "env"})
        assert r.status_code == 200
        assert r.json()["status"] == "ERROR"

    def test_paired_mode_on_ticket_msa_returns_error(self, client: httpx.Client) -> None:
        r = client.post("/ticket/msa", data={"q": MONOMER_Q, "mode": "pairgreedy"})
        assert r.status_code == 200
        assert r.json()["status"] == "ERROR"

    def test_pair_single_chain_returns_error(self, client: httpx.Client) -> None:
        r = client.post("/ticket/pair", data={"q": MONOMER_Q, "mode": "pairgreedy"})
        assert r.status_code == 200
        assert r.json()["status"] == "ERROR"


@pytest.mark.fc
class TestTaskValidation:
    """Task endpoints use standard HTTP 4xx for input errors (distinct from
    ColabFold's 200+ERROR envelope) — callers there speak JobInfo."""

    def test_invalid_mode_returns_422(self, client: httpx.Client) -> None:
        r = client.post("/api/tasks/msa", data={"q": MONOMER_Q, "mode": "nonsense"})
        assert r.status_code == 422

    def test_paired_mode_on_msa_endpoint_returns_422(self, client: httpx.Client) -> None:
        r = client.post("/api/tasks/msa", data={"q": MONOMER_Q, "mode": "pairgreedy"})
        assert r.status_code == 422

    def test_pair_single_chain_returns_422(self, client: httpx.Client) -> None:
        r = client.post("/api/tasks/pair", data={"q": MONOMER_Q, "mode": "pairgreedy"})
        assert r.status_code == 422


# ===================================================================
# Section 4: ColabFold-protocol lifecycle — one MSA, many assertions
# ===================================================================


@pytest.mark.fc
class TestTicketLifecycle:
    def test_status_is_complete(self, msa_job: dict) -> None:
        assert msa_job["final"]["status"] == "COMPLETE"
        assert msa_job["final"]["id"] == msa_job["job_id"]

    def test_jobinfo_endpoint_agrees(
        self, client: httpx.Client, msa_job: dict,
    ) -> None:
        """``GET /api/jobs/<id>`` (framework lifecycle) should agree with the
        ColabFold ``/ticket/<id>`` view on the same job."""
        r = _get_with_retry(
            client, f"/api/jobs/{msa_job['job_id']}", headers=msa_job["session"],
        )
        assert r.status_code == 200
        info = r.json()
        assert info["job_id"] == msa_job["job_id"]
        assert info["status"] == "completed"
        # Privacy: raw sequence must NEVER appear in JobInfo.input_params.
        assert SHORT_MONOMER not in repr(info.get("input_params"))
        # Summary fields the ticket path populated for us.
        assert info["input_params"]["mode"] == "env"
        assert info["input_params"]["sequence_count"] == 1

    def test_result_download_returns_tar_gz_with_a3m(
        self, client: httpx.Client, msa_job: dict,
    ) -> None:
        r = _get_with_retry(
            client, f"/result/download/{msa_job['job_id']}", headers=msa_job["session"],
        )
        assert r.status_code == 200
        # Content-type is application/x-tar; the payload is a gzipped tarball
        # (ColabFold client uses ``tarfile.open(fileobj=..., mode='r|gz')``).
        assert r.headers.get("content-type", "").startswith("application/x-tar")
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
            names = tf.getnames()
        assert any(n.endswith(".a3m") for n in names), (
            f"no .a3m in downloaded tarball: {names}"
        )

    def test_files_endpoint_lists_a3m(
        self, client: httpx.Client, msa_job: dict,
    ) -> None:
        """Framework lifecycle GET /api/jobs/<id>/files also lists the .a3m."""
        r = _get_with_retry(
            client, f"/api/jobs/{msa_job['job_id']}/files", headers=msa_job["session"],
        )
        assert r.status_code == 200
        files = r.json()["files"]
        assert any(f.endswith(".a3m") for f in files), f"no .a3m in files: {files}"


# ===================================================================
# Section 5: A duplicate poll after completion should still succeed
# (regression guard for session-affinity + job-store round-trip).
# ===================================================================


@pytest.mark.fc
class TestPostCompletionQueries:
    def test_repeat_ticket_poll_is_still_complete(
        self, client: httpx.Client, msa_job: dict,
    ) -> None:
        # We already saw COMPLETE inside the fixture; hitting the endpoint
        # again should be idempotent (no accidental transition to ERROR /
        # PENDING).
        r = _get_with_retry(
            client, f"/ticket/{msa_job['job_id']}", headers=msa_job["session"],
        )
        assert r.status_code == 200
        assert r.json()["status"] == "COMPLETE"

    def test_poll_job_helper_reaches_terminal(
        self, client: httpx.Client, base_url: str, msa_job: dict,
    ) -> None:
        """Framework's ``poll_job`` (JobInfo lifecycle) can also observe the
        terminal state — useful for downstream clients like ensemble-server
        that speak JobInfo, not the ColabFold protocol."""
        final = poll_job(
            client, "", msa_job["job_id"],
            timeout_s=60, interval_s=5,
            extra_headers=msa_job["session"],
        )
        assert final["status"] == "completed"

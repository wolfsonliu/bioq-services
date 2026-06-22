"""End-to-end tests against the deployed ensemble-server Function Compute service.

Marked `@pytest.mark.fc`, skipped by default.  Run with:

    pytest -m fc services/ensemble-server/tests/test_fc.py

The base URL is read from `services/aliyun_fc_url.md`.  The URL listed there
is the *VPC internal* URL (`*-vpc.fcapp.run`) — these tests must be executed
from a machine on the VPC (e.g. via VPN), not from the public internet.

Because the URL is VPC-internal and `auth.bypass_vpc=True` is the default,
no API key / JWT is needed.  The bypass is verified by the smoke tests
succeeding without any auth headers.

The end-to-end submit test fans out to GPU FC services (alphafold / esmfold2 /
boltz).  By default it uses only `esmfold2` (fastest, no MSA) and tolerates a
`failed` terminal state, just asserting that the orchestration pipeline runs
to completion (NAS persistence + sub-task lifecycle).  Set
`ENSEMBLE_E2E_REQUIRE_SUCCESS=1` to require at least one method to succeed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url

pytestmark = pytest.mark.fc

# Tiny test sequence — short enough that ESMFold can fold in a few minutes
# without an MSA.
SHORT_PROTEIN = "MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"

TERMINAL_SUB_STATUSES = {"succeeded", "failed", "cached"}


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("ensemble-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(60.0)) as c:
        yield c


def _is_terminal(job: dict) -> bool:
    """An ensemble job is terminal when every sub-task has reached a terminal
    state.  The server also sets `completed_at` at that point — either signal
    is sufficient but we check both to be robust to schema changes.
    """
    if job.get("completed_at"):
        return True
    sub_tasks = job.get("sub_tasks") or {}
    if not sub_tasks:
        return False
    return all(
        (s.get("status") in TERMINAL_SUB_STATUSES) for s in sub_tasks.values()
    )


def _poll_ensemble_job(
    client: httpx.Client,
    task_id: str,
    *,
    timeout_s: int = 1800,
    interval_s: int = 20,
) -> dict:
    """Poll GET /v1/jobs/{task_id} until every sub-task reaches a terminal state."""
    deadline = time.monotonic() + timeout_s
    body: dict = {}
    while time.monotonic() < deadline:
        r = client.get(f"/v1/jobs/{task_id}")
        r.raise_for_status()
        body = r.json()
        if _is_terminal(body):
            return body
        time.sleep(interval_s)
    raise TimeoutError(
        f"ensemble job {task_id!r} did not finish within {timeout_s}s; "
        f"last body: {body!r}"
    )


# =====================================================================
# Smoke — no GPU work, no fan-out.  Also verifies VPC auth bypass:
# all requests are sent without any auth header, so success proves
# bypass_vpc is active for the VPC URL.
# =====================================================================

def test_healthz(client: httpx.Client) -> None:
    r = client.get("/v1/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "ensemble"
    assert "version" in body


def test_manifest(client: httpx.Client) -> None:
    r = client.get("/v1/manifest")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "ensemble"
    assert "folding" in body["task_kinds"]
    assert isinstance(body["methods"]["folding"], list)


def test_list_folding_methods(client: httpx.Client) -> None:
    r = client.get("/v1/methods", params={"task_kind": "folding"})
    r.raise_for_status()
    body = r.json()
    assert body["task_kind"] == "folding"
    methods = {m["name"] for m in body["methods"]}
    assert methods, "no folding methods registered; check ENSEMBLE_FC_METHODS__*"
    for m in body["methods"]:
        assert "options_schema" in m
        assert "estimated_runtime_seconds_default" in m


def test_list_methods_unknown_task_kind(client: httpx.Client) -> None:
    r = client.get("/v1/methods", params={"task_kind": "no-such-kind"})
    assert r.status_code == 422


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/v1/jobs/missing-ensemble-job-id").status_code == 404


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_submit_unknown_method_returns_422(client: httpx.Client) -> None:
    payload = {
        "input": {
            "sequences": [{"id": "A", "sequence": SHORT_PROTEIN}],
            "msa_mode": "empty",
        },
        "methods": ["no-such-method"],
    }
    r = client.post("/v1/folding/ensemble", json=payload)
    assert r.status_code == 422


# =====================================================================
# End-to-end fan-out + aggregation.  Submits a real folding ensemble
# job and polls to terminal state.  Uses only `esmfold2` (no MSA, fast)
# to keep runtime predictable.
# =====================================================================

def test_folding_ensemble_submit_and_terminal(client: httpx.Client) -> None:
    """Submit a real folding ensemble job and poll to a terminal state.

    By default tolerates a `failed` terminal state (since downstream FC
    services or platform creds may be misconfigured).  Set
    ENSEMBLE_E2E_REQUIRE_SUCCESS=1 to require at least one method to succeed.
    """
    available = {
        m["name"]
        for m in client.get("/v1/methods", params={"task_kind": "folding"}).json()["methods"]
    }
    method = "esmfold2" if "esmfold2" in available else next(iter(available))
    assert method, "no folding methods registered — cannot run e2e test"

    payload = {
        "input": {
            "sequences": [{"id": "A", "sequence": SHORT_PROTEIN}],
            "msa_mode": "empty",
        },
        "methods": [method],
    }
    r = client.post("/v1/folding/ensemble", json=payload)
    assert r.status_code == 202, f"submit failed: {r.status_code} {r.text!r}"
    body = r.json()
    task_id = body["task_id"]
    assert body["status"] == "accepted"
    assert body["requested_methods"] == [method]

    final = _poll_ensemble_job(client, task_id, timeout_s=1800, interval_s=20)

    # Basic schema assertions on the persisted ensemble job.
    assert final["task_id"] == task_id
    assert final["task_kind"] == "folding"
    assert final["customer_id"]                # set by auth layer; VPC bypass → "internal_vpc"
    assert final["completed_at"]               # set when all sub-tasks terminate
    assert method in final["sub_tasks"]

    sub = final["sub_tasks"][method]
    assert sub["status"] in TERMINAL_SUB_STATUSES, sub

    if os.environ.get("ENSEMBLE_E2E_REQUIRE_SUCCESS"):
        assert sub["status"] in ("succeeded", "cached"), (
            f"sub-task did not succeed: {sub.get('error_summary')!r}"
        )
        assert final["aggregated_output"] is not None
    elif sub["status"] == "failed":
        # Surface the failure context so the test output is useful for
        # debugging downstream configuration issues without erroring.
        print(
            f"\n[ensemble e2e] sub-task {method!r} failed: "
            f"{sub.get('error_summary')!r}"
        )

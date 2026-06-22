"""End-to-end tests against the deployed ensemble-server Function Compute service.

Marked `@pytest.mark.fc`, skipped by default.  Run with:

    pytest -m fc services/ensemble-server/tests/test_fc.py

The base URL is read from `services/aliyun_fc_url.md`.  The URL listed there
is the *VPC internal* URL (`*-vpc.fcapp.run`) — these tests must be executed
from a machine on the VPC (e.g. via VPN), not from the public internet.

Because the URL is VPC-internal and `auth.bypass_vpc=True` is the default,
no API key / JWT is needed.  The bypass is verified by the smoke tests
succeeding without any auth headers.

The submit-and-poll test is opt-in (marked `slow`) because it fans out to
GPU FC services (alphafold / esmfold2 / boltz) that may each take 5-30 min.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url

pytestmark = pytest.mark.fc

# Tiny test sequence — short enough that ESMFold/Boltz can fold in a few minutes.
SHORT_PROTEIN = "MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("ensemble-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(60.0)) as c:
        yield c


def _poll_ensemble_job(
    client: httpx.Client,
    task_id: str,
    *,
    timeout_s: int = 3600,
    interval_s: int = 30,
) -> dict:
    """Poll GET /v1/jobs/{task_id} until status is terminal.

    Ensemble jobs reach terminal states: completed | partial | failed.
    """
    deadline = time.monotonic() + timeout_s
    body: dict = {}
    while time.monotonic() < deadline:
        r = client.get(f"/v1/jobs/{task_id}")
        r.raise_for_status()
        body = r.json()
        if body["status"] in ("completed", "partial", "failed"):
            return body
        time.sleep(interval_s)
    raise TimeoutError(
        f"ensemble job {task_id!r} did not finish within {timeout_s}s; "
        f"last status: {body.get('status', '<unknown>')}"
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
    # At least one folding method must be registered for the service to be useful.
    assert methods, f"no folding methods registered; check ENSEMBLE_FC_METHODS__*"


def test_list_methods_unknown_task_kind(client: httpx.Client) -> None:
    r = client.get("/v1/methods", params={"task_kind": "no-such-kind"})
    assert r.status_code == 422


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/v1/jobs/missing-ensemble-job-id").status_code == 404


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


# =====================================================================
# End-to-end fan-out + aggregation.  Submits a real folding ensemble
# job and polls to terminal state.  Opt-in: slow + depends on the
# downstream FC services being healthy.
# =====================================================================

@pytest.mark.slow
def test_folding_ensemble_submit_and_poll(client: httpx.Client) -> None:
    """Submit a folding ensemble, poll to terminal state, sanity-check the result."""
    # Determine which methods are registered first; use them all.
    methods_body = client.get("/v1/methods", params={"task_kind": "folding"}).json()
    methods = [m["name"] for m in methods_body["methods"]]
    assert methods, "no folding methods registered — cannot run e2e test"

    payload = {
        "input": {
            "sequences": [{"id": "A", "sequence": SHORT_PROTEIN}],
            "msa_mode": "empty",
        },
        "methods": methods,
    }
    r = client.post("/v1/folding/ensemble", json=payload)
    assert r.status_code == 202, f"submit failed: {r.status_code} {r.text!r}"
    body = r.json()
    task_id = body["task_id"]
    assert body["status"] == "accepted"
    assert set(body["requested_methods"]) == set(methods)

    final = _poll_ensemble_job(client, task_id, timeout_s=3600, interval_s=30)
    assert final["status"] in ("completed", "partial"), final
    # At least one method should have produced a structure for the test to pass.
    completed = [
        r for r in final.get("results", []) if r.get("status") == "completed"
    ]
    assert completed, f"no method completed successfully: {final.get('results')}"

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
# End-to-end fan-out + aggregation.  Submits real folding ensemble jobs
# and polls to terminal state.
#
# Default coverage: esmfold2 + boltz parametrized (each ~2-4 min), plus
# one multi-method aggregation test that combines them.  AlphaFold is
# excluded by default because its full MSA + 5-model pipeline takes
# ~30 min, which is impractical for routine CI.  Set
# ENSEMBLE_E2E_INCLUDE_ALPHAFOLD=1 to include it.
#
# Each test tolerates a `failed` terminal state by default so a
# downstream config issue doesn't crash the suite; set
# ENSEMBLE_E2E_REQUIRE_SUCCESS=1 to enforce real success.
# =====================================================================

FAST_METHODS = ["esmfold2", "boltz"]


def _submit_ensemble(client: httpx.Client, methods: list[str]) -> str:
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
    assert body["status"] == "accepted"
    assert sorted(body["requested_methods"]) == sorted(methods)
    return body["task_id"]


def _assert_terminal_shape(final: dict, task_id: str, methods: list[str]) -> None:
    """Assert structural invariants of a terminal EnsembleJob."""
    assert final["task_id"] == task_id
    assert final["task_kind"] == "folding"
    assert final["customer_id"]                # VPC bypass → "internal_vpc"
    assert final["completed_at"]               # set when all sub-tasks terminate
    for m in methods:
        assert m in final["sub_tasks"], final["sub_tasks"]
        assert final["sub_tasks"][m]["status"] in TERMINAL_SUB_STATUSES


def _assert_success_if_required(final: dict, methods: list[str]) -> None:
    """If ENSEMBLE_E2E_REQUIRE_SUCCESS is set, enforce that every method succeeded
    and the aggregator populated `aggregated_output`."""
    require = bool(os.environ.get("ENSEMBLE_E2E_REQUIRE_SUCCESS"))
    for m in methods:
        sub = final["sub_tasks"][m]
        if require:
            assert sub["status"] in ("succeeded", "cached"), (
                f"sub-task {m!r} did not succeed: {sub.get('error_summary')!r}"
            )
        elif sub["status"] == "failed":
            print(
                f"\n[ensemble e2e] sub-task {m!r} failed: "
                f"{sub.get('error_summary')!r}"
            )
    if require:
        assert final["aggregated_output"] is not None


@pytest.mark.parametrize("method", FAST_METHODS)
def test_folding_ensemble_single_method(
    client: httpx.Client, method: str,
) -> None:
    """Submit a real folding job using one method, poll to a terminal state."""
    available = {
        m["name"]
        for m in client.get("/v1/methods", params={"task_kind": "folding"}).json()["methods"]
    }
    if method not in available:
        pytest.skip(f"method {method!r} not registered in deployed service")

    task_id = _submit_ensemble(client, [method])
    final = _poll_ensemble_job(client, task_id, timeout_s=1800, interval_s=20)
    _assert_terminal_shape(final, task_id, [method])
    _assert_success_if_required(final, [method])


def test_folding_ensemble_multi_method_aggregation(client: httpx.Client) -> None:
    """Submit an ensemble across multiple fast methods to exercise aggregation.

    Cross-method ranking is the actual MVP use-case for ensemble-server, so
    this verifies that:
      1. fan-out to N methods all returns 202 from a single submit
      2. polling sees them transition independently
      3. the aggregator populates `aggregated_output` once all succeed
    """
    available = {
        m["name"]
        for m in client.get("/v1/methods", params={"task_kind": "folding"}).json()["methods"]
    }
    methods = [m for m in FAST_METHODS if m in available]
    if os.environ.get("ENSEMBLE_E2E_INCLUDE_ALPHAFOLD") and "alphafold" in available:
        methods.append("alphafold")
    if len(methods) < 2:
        pytest.skip(f"need ≥2 registered fast methods; have {methods!r}")

    task_id = _submit_ensemble(client, methods)
    # AlphaFold can take 30+ minutes, so widen the timeout when included.
    timeout = 4200 if "alphafold" in methods else 1800
    final = _poll_ensemble_job(client, task_id, timeout_s=timeout, interval_s=30)
    _assert_terminal_shape(final, task_id, methods)
    _assert_success_if_required(final, methods)

    # When all sub-tasks succeed, aggregator must populate ensemble_ranking
    # spanning all methods.  This is only enforced when REQUIRE_SUCCESS is on.
    if os.environ.get("ENSEMBLE_E2E_REQUIRE_SUCCESS"):
        agg = final["aggregated_output"]
        ranking = agg.get("ensemble_ranking", [])
        assert ranking, f"empty ensemble_ranking in aggregated_output: {agg!r}"
        ranked_methods = {entry["method"] for entry in ranking}
        assert ranked_methods == set(methods), (
            f"ensemble_ranking missing methods: ranked={ranked_methods!r} "
            f"expected={set(methods)!r}"
        )

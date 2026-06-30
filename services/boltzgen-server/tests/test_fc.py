"""End-to-end tests against the deployed BoltzGen Function Compute service.

Marked ``@pytest.mark.fc``, skipped by default.  Run with::

    RUN_FC_TESTS=1 pytest -m fc services/boltzgen-server/tests/test_fc.py -v

BoltzGen has two pairs of endpoints:
  * ``/api/design``                  — full binder design pipeline (submit/poll)
  * ``/api/inverse_fold``            — inverse-fold-only mode (submit/poll)
  * ``/api/tasks/design``            — async task mode (FC blocking 202)
  * ``/api/tasks/inverse_fold``      — async task mode (FC blocking 202)

The inference tests use ``num_designs=2, budget=2`` to keep FC GPU time
manageable; tighter values risk producing 0 outputs and a flaky test.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"

FC_DESIGN_YAML = DATA_DIR / "fc_design.yaml"
INVERSE_FOLD_YAML = DATA_DIR / "inverse_fold.yaml"
DUMMY_TARGET_CIF = DATA_DIR / "dummy_target.cif"

pytestmark = pytest.mark.fc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("boltzgen-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_with_retry(
    client: httpx.Client,
    path: str,
    *,
    max_attempts: int = 8,
    backoff_s: int = 15,
) -> httpx.Response:
    """GET that tolerates FC HTTP-gateway 429s.

    FC returns 429 with a ResourceExhausted body when the account-level GPU
    quota is hit by another service.  This is a platform-layer artifact
    unrelated to boltzgen-server — see project memory
    ``project_fc_http_polling_unreliable_at_concurrency.md``.  Retry until
    the gateway lets the request through, or we exhaust attempts.
    """
    import time as _t
    last: httpx.Response | None = None
    for _attempt in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        _t.sleep(backoff_s)
    assert last is not None
    return last


def _assert_submitted(resp_json: dict) -> None:
    assert "job_id" in resp_json
    assert resp_json["status"] in ("pending", "running")
    assert resp_json["created_at"] is not None
    assert resp_json["input_params"] is not None
    assert isinstance(resp_json["input_params"], dict)


def _assert_completed(job: dict, base_url: str, client: httpx.Client) -> None:
    assert job["status"] == "completed", (
        f"failed: kind={job.get('failure_kind')} summary={job.get('error_summary')!r}"
    )

    assert job["created_at"] is not None
    assert job["started_at"] is not None
    assert job["completed_at"] is not None
    assert job["duration_seconds"] is not None
    assert job["duration_seconds"] > 0
    assert job["input_params"] is not None
    assert isinstance(job["input_params"], dict)
    assert job["output_count"] is not None
    assert job["output_count"] > 0
    assert job["output_total_bytes"] is not None
    assert job["output_total_bytes"] > 0

    files = client.get(f"/api/jobs/{job['job_id']}/files").json()["files"]
    assert files, "no output files"


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "boltzgen"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    """Custom /healthz/detail override (v0.0.10+) reports NAS weight presence."""
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "boltzgen"
    # Weights externalized to NAS — verify both expected paths exist.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    assert body["weights_loaded"] is True, (
        f"NAS weights missing: {body.get('weights_missing')}"
    )
    assert body["weights_dir"] == "/data/models/boltzgen/weights"
    assert body["moldir"] == "/data/models/boltzgen/moldir"
    assert body["weights_missing"] == {}
    assert body["max_concurrent_jobs"] >= 1


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    """Manifest must list the 2 sync endpoints; async task variants accepted."""
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    sync_endpoints = {"/api/design", "/api/inverse_fold"}
    assert sync_endpoints <= paths, (
        f"expected sync endpoints {sync_endpoints} ⊆ paths, got {paths}"
    )
    extras = paths - sync_endpoints
    assert extras <= {"/api/tasks/design", "/api/tasks/inverse_fold"}, (
        f"unexpected non-task endpoints: {extras}"
    )


def test_manifest_extras_have_protocols_and_models(client: httpx.Client) -> None:
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "protocols" in extras
    assert "models" in extras
    assert "tool_outputs" in extras
    assert "design" in extras["tool_outputs"]


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_design_rejects_missing_yaml(client: httpx.Client) -> None:
    """Posting /api/design with no YAML and no URI must return 422."""
    r = client.post(
        "/api/design",
        data={"protocol": "protein-anything", "num_designs": "2"},
    )
    # framework returns 422 from `resolve_input` or upstream validation
    assert r.status_code in (400, 422), f"unexpected status: {r.status_code} {r.text!r}"


# ---------------------------------------------------------------------------
# Inference — design (protein-only, no ref files)
# ---------------------------------------------------------------------------


def test_design_protein_binder(client: httpx.Client, base_url: str) -> None:
    """Protein binder design: 40-60aa binder for a 19-mer peptide.

    Uses num_designs=2, budget=2 to keep FC GPU time manageable.
    Protocol is ``protein-anything`` (default).
    """
    with open(FC_DESIGN_YAML, "rb") as fh:
        r = client.post(
            "/api/design",
            files={"design_yaml": (FC_DESIGN_YAML.name, fh, "application/x-yaml")},
            data={
                "protocol": "protein-anything",
                "num_designs": "2",
                "budget": "2",
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["protocol"] == "protein-anything"
    assert submit["input_params"]["num_designs"] == 2
    assert submit["input_params"]["budget"] == 2

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)

    # design output should include at least one .pdb / .cif candidate per the
    # service_specific.tool_outputs description.
    files = client.get(f"/api/jobs/{submit['job_id']}/files").json()["files"]
    structures = [f for f in files if f.endswith(".pdb") or f.endswith(".cif")]
    assert structures, f"no structure files in outputs: {files}"


# ---------------------------------------------------------------------------
# Inference — inverse_fold (uses a backbone CIF + design selection YAML)
# ---------------------------------------------------------------------------


def test_inverse_fold_basic(client: httpx.Client, base_url: str) -> None:
    """Inverse-fold-only: provide a backbone CIF, BoltzGen designs sequence.

    Skips the upstream diffusion step (no scaffold sampling), runs
    inverse_folding -> folding -> analysis -> filtering on the supplied
    structure. Faster than full /api/design.
    """
    with open(INVERSE_FOLD_YAML, "rb") as fh_yaml, \
         open(DUMMY_TARGET_CIF, "rb") as fh_cif:
        r = client.post(
            "/api/inverse_fold",
            files=[
                ("design_yaml", (INVERSE_FOLD_YAML.name, fh_yaml.read(),
                                 "application/x-yaml")),
                ("ref_files", (DUMMY_TARGET_CIF.name, fh_cif.read(),
                               "chemical/x-cif")),
            ],
            data={
                "protocol": "protein-anything",
                "num_designs": "2",
                "budget": "2",
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["num_designs"] == 2

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)


# ---------------------------------------------------------------------------
# Task endpoints (FC async task mode) — submit semantics only
# ---------------------------------------------------------------------------


def test_design_task_endpoint_registered(client: httpx.Client) -> None:
    """Task endpoint MUST be registered in OpenAPI when async mode is enabled."""
    r = _get_with_retry(client, "/openapi.json")
    r.raise_for_status()
    spec = r.json()
    assert "/api/tasks/design" in spec["paths"], (
        "task endpoint missing from OpenAPI; "
        "BOLTZGEN_TASK_ENDPOINTS_ENABLED may be False on the deployed function."
    )
    assert "/api/tasks/inverse_fold" in spec["paths"]

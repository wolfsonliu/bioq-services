"""End-to-end tests against the deployed ODesign Function Compute service.

Marked ``@pytest.mark.fc``, skipped by default.  Run with::

    pytest -m fc services/odesign-server/tests/test_fc.py

ODesign has one endpoint:
  * ``/api/design`` — unified biomolecular interaction design

The design test uses a pure-sequence protein JSON spec (no reference files)
with ``n_sample=2, seeds=[42]`` to keep FC GPU time manageable.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"

FC_DESIGN_JSON = DATA_DIR / "fc_design.json"

pytestmark = pytest.mark.fc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("odesign-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    assert body["service"] == "odesign"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "odesign"
    assert body["jobs_base_dir_exists"] is True


def test_manifest_lists_endpoint(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/design"}


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ---------------------------------------------------------------------------
# Inference — design (protein-only, no ref files)
# ---------------------------------------------------------------------------


def test_design_protein_binder(client: httpx.Client, base_url: str) -> None:
    """Protein binder design: 40-60aa binder for a 19-mer peptide.

    Uses n_sample=2, seeds=[42] to keep FC GPU time manageable.
    Model: odesign_base_prot_flex (default).
    """
    with open(FC_DESIGN_JSON, "rb") as fh:
        r = client.post(
            "/api/design",
            files={"input_json": (FC_DESIGN_JSON.name, fh, "application/json")},
            data={
                "model": "odesign_base_prot_flex",
                "n_sample": "2",
                "seeds": "[42]",
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["model"] == "odesign_base_prot_flex"
    assert submit["input_params"]["n_sample"] == 2

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)

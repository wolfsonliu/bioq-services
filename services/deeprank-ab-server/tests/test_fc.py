"""End-to-end tests against the deployed DeepRank-Ab Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/deeprank-ab-server/tests/test_fc.py

DeepRank-Ab has one endpoint:
  * `/api/score` — score an antibody-antigen docking complex PDB

Test PDB ships in `tests/data/` (copied from upstream example/).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_PDB = DATA_DIR / "test.pdb"

pytestmark = pytest.mark.fc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("deeprank-ab-server", start=Path(__file__))


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
    csv_files = [f for f in files if f.endswith("_predictions.csv")]
    assert csv_files, "no predictions CSV in output"


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "deeprank-ab"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "deeprank-ab"
    assert body["jobs_base_dir_exists"] is True


def test_manifest_lists_score_endpoint(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/score"}


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_score_antibody_antigen(client: httpx.Client, base_url: str) -> None:
    """Score the example antibody-antigen complex (chains H, L, A)."""
    with open(TEST_PDB, "rb") as fh:
        r = client.post(
            "/api/score",
            files={"input_pdb": (TEST_PDB.name, fh, "chemical/x-pdb")},
            data={
                "heavy_chain_id": "H",
                "light_chain_id": "L",
                "antigen_chain_id": "A",
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["heavy_chain_id"] == "H"
    assert submit["input_params"]["light_chain_id"] == "L"
    assert submit["input_params"]["antigen_chain_id"] == "A"

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)

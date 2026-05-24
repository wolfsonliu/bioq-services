"""End-to-end tests against the deployed RFantibody Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/rfantibody-server/tests/test_fc.py

RFantibody is a pipeline: rfdiffusion → proteinmpnn → rf2. We exercise each
endpoint at least once, but chain them via module-scoped fixtures so we don't
re-run the expensive rfdiffusion step three times — the downstream tests pull
their inputs via `input_uri=job://<id>/...` URIs.

If `test_rfdiffusion_minimal_job` fails, the proteinmpnn/rf2 tests will be
marked as errored automatically (fixture dependency).

Test PDBs ship in `tests/data/` (copied from upstream RFantibody examples) so
the suite is self-contained — no dependency on `opensource/RFantibody/`.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"

TARGET_PDB = DATA_DIR / "rsv_site3.pdb"
ANTIBODY_FRAMEWORK_PDB = DATA_DIR / "hu-4D5-8_Fv.pdb"

pytestmark = pytest.mark.fc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("rfantibody-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
        yield c


def _assert_submitted(resp_json: dict) -> None:
    """Validate the immediate POST response has expected fields."""
    assert "job_id" in resp_json
    assert resp_json["status"] in ("pending", "running")
    assert resp_json["created_at"] is not None
    assert resp_json["input_params"] is not None
    assert isinstance(resp_json["input_params"], dict)


def _assert_completed(job: dict, base_url: str, client: httpx.Client) -> dict:
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
    return job


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "rfantibody"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "rfantibody"
    assert body["jobs_base_dir_exists"] is True


def test_manifest_lists_three_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/rfdiffusion", "/api/proteinmpnn", "/api/rf2"}


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ---------------------------------------------------------------------------
# Chained inference — module-scoped fixtures so rfdiffusion runs once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rfdiffusion_job(client: httpx.Client, base_url: str) -> dict:
    """Run rfdiffusion once, return completed job dict. Downstream tests pull its .qv."""
    with open(TARGET_PDB, "rb") as t, open(ANTIBODY_FRAMEWORK_PDB, "rb") as f:
        r = client.post(
            "/api/rfdiffusion",
            files={
                "target": (TARGET_PDB.name, t, "chemical/x-pdb"),
                "framework": (ANTIBODY_FRAMEWORK_PDB.name, f, "chemical/x-pdb"),
            },
            data={
                "num_designs": "1",
                "diffuser_t": "25",
                "design_loops": "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13",
                "hotspots": "T305,T456",
                "deterministic": "true",
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["num_designs"] == 1
    assert submit["input_params"]["design_loops"] == "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13"
    assert submit["input_params"]["hotspots"] == "T305,T456"

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)
    return final


@pytest.fixture(scope="module")
def proteinmpnn_job(
    client: httpx.Client, base_url: str, rfdiffusion_job: dict
) -> dict:
    """Run proteinmpnn off the rfdiffusion output (via job:// URI)."""
    rfdiffusion_job_id = rfdiffusion_job["job_id"]
    r = client.post(
        "/api/proteinmpnn",
        data={
            "input_uri": f"job://{rfdiffusion_job_id}/1_rfdiffusion.qv",
            "seqs_per_struct": "1",
            "deterministic": "true",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["seqs_per_struct"] == 1
    assert submit["input_params"]["deterministic"] is True

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)
    return final


def test_rfdiffusion_minimal_job(
    client: httpx.Client, base_url: str, rfdiffusion_job: dict
) -> None:
    """Verify the rfdiffusion fixture job actually produced a .qv."""
    files = client.get(f"/api/jobs/{rfdiffusion_job['job_id']}/files").json()["files"]
    names = {f["path"] for f in files} if isinstance(files[0], dict) else set(files)
    assert any("1_rfdiffusion.qv" in n for n in names), f"missing .qv: {names}"


def test_proteinmpnn_minimal_job(
    client: httpx.Client, base_url: str, proteinmpnn_job: dict
) -> None:
    files = client.get(f"/api/jobs/{proteinmpnn_job['job_id']}/files").json()["files"]
    names = {f["path"] for f in files} if isinstance(files[0], dict) else set(files)
    assert any("2_proteinmpnn.qv" in n for n in names), f"missing .qv: {names}"


def test_rf2_minimal_job(
    client: httpx.Client, base_url: str, proteinmpnn_job: dict
) -> None:
    """Run rf2 off the proteinmpnn output."""
    proteinmpnn_job_id = proteinmpnn_job["job_id"]
    r = client.post(
        "/api/rf2",
        data={
            "input_uri": f"job://{proteinmpnn_job_id}/2_proteinmpnn.qv",
            "num_recycles": "2",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["num_recycles"] == 2

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)
    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    names = {f["path"] for f in files} if isinstance(files[0], dict) else set(files)
    assert any("3_rf2.qv" in n for n in names), f"missing .qv: {names}"

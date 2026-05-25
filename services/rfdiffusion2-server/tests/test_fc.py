"""End-to-end tests against the deployed RFdiffusion2 Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/rfdiffusion2-server/tests/test_fc.py

RFdiffusion2 has three endpoints:
  * `/api/generate/active_site`           — atomic motif + ligand scaffolding
  * `/api/generate/small_molecule_binder` — RASA-conditioned binder design
  * `/api/generate`                       — raw contig + freeform Hydra overrides

Test PDBs ship in `tests/data/` (copied from upstream RFdiffusion2 benchmarks)
so the suite is self-contained. Each job generates 1 design with diffuser_t=10
(instead of default 100) to keep FC GPU time manageable (~10-15 min per endpoint
on slower FC instances). Quality doesn't matter — only pipeline correctness.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"

ACTIVE_SITE_PDB = DATA_DIR / "M0584_1ldm.pdb"
SM_BINDER_PDB = DATA_DIR / "trimmed_ec2_M0151_NO_ORI_zero_com0.pdb"

pytestmark = pytest.mark.fc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("rfdiffusion2-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_submitted(resp_json: dict) -> None:
    """Validate the immediate POST response has expected fields."""
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
    assert body["service"] == "rfdiffusion2"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "rfdiffusion2"
    assert body["jobs_base_dir_exists"] is True


def test_manifest_lists_three_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {
        "/api/generate/active_site",
        "/api/generate/small_molecule_binder",
        "/api/generate",
    }


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_active_site_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Active-site scaffolding: lactate dehydrogenase NAD+OXM (unindexed).

    Uses diffuser_t=10 to keep GPU time manageable on slower FC instances
    (~10 min instead of ~100 min at the default t=100). Quality doesn't
    matter — this test only validates the end-to-end pipeline.
    """
    contig_atoms = {
        "A106": "NE,CD,CZ",
        "A166": "OD1,CG",
        "A169": "NH2,CZ",
        "A193": "NE2,CD2,CE1",
    }
    with open(ACTIVE_SITE_PDB, "rb") as fh:
        r = client.post(
            "/api/generate/active_site",
            files={"input_pdb": (ACTIVE_SITE_PDB.name, fh, "chemical/x-pdb")},
            data={
                "contigs": "46,A106-106,59,A166-166,2,A169-169,23,A193-193,46",
                "contig_atoms": json.dumps(contig_atoms),
                "ligand": "NAD,OXM",
                "contig_as_guidepost": "true",
                "num_designs": "1",
                "diffuser_t": "10",
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["ligand"] == "NAD,OXM"
    assert submit["input_params"]["contig_as_guidepost"] is True
    assert submit["input_params"]["num_designs"] == 1
    assert submit["input_params"]["diffuser_t"] == 10

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)


def test_small_molecule_binder_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Small-molecule binder: 50aa buried binder around PH2 (RASA=0).

    Uses diffuser_t=10 + shorter contigs to keep FC GPU time under ~15 min.
    """
    with open(SM_BINDER_PDB, "rb") as fh:
        r = client.post(
            "/api/generate/small_molecule_binder",
            files={"input_pdb": (SM_BINDER_PDB.name, fh, "chemical/x-pdb")},
            data={
                "contigs": "50",
                "length": "50-50",
                "ligand": "PH2",
                "rasa_active": "true",
                "rasa_target": "0",
                "num_designs": "1",
                "diffuser_t": "10",
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["ligand"] == "PH2"
    assert submit["input_params"]["rasa_active"] is True
    assert submit["input_params"]["num_designs"] == 1
    assert submit["input_params"]["diffuser_t"] == 10

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)


def test_custom_active_site_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Freeform endpoint — active-site via raw contigs + extra_overrides.

    Uses the same M0584_1ldm PDB but drives it through the generic /api/generate
    endpoint with explicit Hydra overrides instead of the typed active_site params.
    Uses diffuser_t=10 for fast FC testing.
    """
    contig_atoms = {
        "A106": "NE,CD,CZ",
        "A166": "OD1,CG",
        "A169": "NH2,CZ",
        "A193": "NE2,CD2,CE1",
    }
    with open(ACTIVE_SITE_PDB, "rb") as fh:
        r = client.post(
            "/api/generate",
            files={"input_pdb": (ACTIVE_SITE_PDB.name, fh, "chemical/x-pdb")},
            data={
                "contigs": "46,A106-106,59,A166-166,2,A169-169,23,A193-193,46",
                "config_name": "aa",
                "input_pdb_required": "true",
                "ligand": "NAD,OXM",
                "num_designs": "1",
                "diffuser_t": "10",
                "extra_overrides": json.dumps({
                    "inference.contig_as_guidepost": True,
                    "contigmap.contig_atoms": contig_atoms,
                }),
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["contigs"] == "46,A106-106,59,A166-166,2,A169-169,23,A193-193,46"
    assert submit["input_params"]["num_designs"] == 1
    assert submit["input_params"]["ligand"] == "NAD,OXM"
    assert submit["input_params"]["diffuser_t"] == 10

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)

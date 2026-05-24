"""End-to-end tests against the deployed RFdiffusion Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/rfdiffusion-server/tests/test_fc.py

Each endpoint is hit with the smallest viable job RFdiffusion will accept:
1 design, 25 timesteps (vs the 50 default — halves wall-clock), short contigs.
Total ~5-10 min per endpoint depending on FC GPU class.

Test PDBs ship in `tests/data/` (copied from upstream RFdiffusion examples) so
the suite is self-contained — no dependency on `opensource/RFdiffusion/`.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"

MOTIF_PDB = DATA_DIR / "5TPN.pdb"          # RSV-F, used by upstream design_motifscaffolding.sh
BINDER_TARGET_PDB = DATA_DIR / "insulin_target.pdb"  # used by design_ppi.sh

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("rfdiffusion-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "rfdiffusion"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "rfdiffusion"
    assert body["jobs_base_dir_exists"] is True


def test_manifest_lists_five_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {
        "/api/generate/unconditional",
        "/api/generate/motif",
        "/api/generate/binder",
        "/api/generate/symmetry",
        "/api/generate",
    }


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ---------------------------------------------------------------------------
# Inference — one minimal job per endpoint.
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


def test_unconditional_minimal_job(client: httpx.Client, base_url: str) -> None:
    """1 design, 60-residue monomer, halved timesteps."""
    r = client.post(
        "/api/generate/unconditional",
        data={
            "num_designs": "1",
            "diffuser_t": "25",
            "min_length": "60",
            "max_length": "60",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["num_designs"] == 1
    assert submit["input_params"]["min_length"] == 60

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)


def test_motif_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Motif scaffolding around RSV-F site 5 (upstream's canonical demo)."""
    with open(MOTIF_PDB, "rb") as fh:
        r = client.post(
            "/api/generate/motif",
            files={"input_pdb": (MOTIF_PDB.name, fh, "chemical/x-pdb")},
            data={
                "contigs": "10-40/A163-181/10-40",
                "num_designs": "1",
                "diffuser_t": "25",
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["contigs"] == "10-40/A163-181/10-40"
    assert submit["input_params"]["num_designs"] == 1

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)


def test_binder_minimal_job(client: httpx.Client, base_url: str) -> None:
    """PPI binder against insulin receptor — upstream design_ppi.sh, shrunk."""
    with open(BINDER_TARGET_PDB, "rb") as fh:
        r = client.post(
            "/api/generate/binder",
            files={"input_pdb": (BINDER_TARGET_PDB.name, fh, "chemical/x-pdb")},
            data={
                "contigs": "A1-150/0 70-70",
                "hotspots": "A59,A83,A91",
                "num_designs": "1",
                "diffuser_t": "25",
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["contigs"] == "A1-150/0 70-70"
    assert submit["input_params"]["hotspots"] == "A59,A83,A91"

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)


def test_symmetry_minimal_job(client: httpx.Client, base_url: str) -> None:
    """C2 symmetric dimer, 80 residues total (40/chain)."""
    r = client.post(
        "/api/generate/symmetry",
        data={
            "symmetry": "c2",
            "total_length": "80",
            "num_designs": "1",
            "diffuser_t": "25",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["symmetry"] == "c2"
    assert submit["input_params"]["total_length"] == 80

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)


def test_custom_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Freeform endpoint — unconditional via raw contig (no input PDB)."""
    r = client.post(
        "/api/generate",
        data={
            "contigs": "60-60",
            "config_name": "base",
            "num_designs": "1",
            "diffuser_t": "25",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["contigs"] == "60-60"
    assert submit["input_params"]["num_designs"] == 1

    final = poll_job(client, base_url, submit["job_id"])
    _assert_completed(final, base_url, client)

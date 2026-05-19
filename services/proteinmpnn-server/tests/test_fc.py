"""End-to-end tests against the deployed ProteinMPNN Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/proteinmpnn-server/tests/test_fc.py

URL resolves via `services/aliyun_fc_url.md`. Test PDB ships in `tests/data/`
(monomer example 5L33, ~180 residues — copied from upstream ProteinMPNN so the
suite is self-contained). Each inference call generates 2 sequences max.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

TEST_PDB = Path(__file__).resolve().parent / "data" / "5L33.pdb"

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("proteinmpnn-server", start=Path(__file__))


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
    assert body["service"] == "proteinmpnn"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "proteinmpnn"
    assert body["jobs_base_dir_exists"] is True


def test_manifest_lists_three_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/design", "/api/score", "/api/probs"}


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _assert_completed(job: dict, base_url: str, client: httpx.Client) -> None:
    assert job["status"] == "completed", (
        f"failed: kind={job.get('failure_kind')} summary={job.get('error_summary')!r}"
    )
    files = client.get(f"/api/jobs/{job['job_id']}/files").json()["files"]
    assert files, "no output files"


def test_design_minimal_job(client: httpx.Client, base_url: str) -> None:
    """2 sequences, default vanilla v_48_020. Output is a FASTA in seqs/."""
    with open(TEST_PDB, "rb") as fh:
        r = client.post(
            "/api/design",
            files={"pdb": (TEST_PDB.name, fh, "chemical/x-pdb")},
            data={
                "name": "fc_smoke_design",
                "num_seq_per_target": "2",
                "batch_size": "1",
            },
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _assert_completed(final, base_url, client)


def test_score_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Score-only path: writes per-position scores for 2 random-sampled seqs."""
    with open(TEST_PDB, "rb") as fh:
        r = client.post(
            "/api/score",
            files={"pdb": (TEST_PDB.name, fh, "chemical/x-pdb")},
            data={"name": "fc_smoke_score", "num_seq_per_target": "2"},
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _assert_completed(final, base_url, client)


def test_probs_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Per-residue AA probabilities — default conditional mode."""
    with open(TEST_PDB, "rb") as fh:
        r = client.post(
            "/api/probs",
            files={"pdb": (TEST_PDB.name, fh, "chemical/x-pdb")},
            data={"name": "fc_smoke_probs"},
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _assert_completed(final, base_url, client)

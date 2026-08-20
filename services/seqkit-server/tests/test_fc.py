"""End-to-end tests against a deployed seqkit-server FC service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/seqkit-server/tests/test_fc.py
    # or
    RUN_FC_TESTS=1 pytest services/seqkit-server/tests/test_fc.py

Fixtures live in `tests/data/` so the suite is self-contained. URL is read from
`services.yaml` via `bioq_service.fc_testing`.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from bioq_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
FASTA = DATA_DIR / "input.fasta"

pytestmark = pytest.mark.fc

TIMEOUT_S = 300


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("seqkit-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(60.0)) as c:
        yield c


# ---- Smoke ----

def test_healthz(client: httpx.Client) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "seqkit"


def test_healthz_detail(client: httpx.Client) -> None:
    body = client.get("/healthz/detail").json()
    assert body["service"] == "seqkit"
    assert body["ready"] is True, body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/stats" in paths
    assert "/api/revcomp" in paths


def test_422_bad_seq_type(client: httpx.Client) -> None:
    with open(FASTA, "rb") as f:
        r = client.post("/api/revcomp", files={"input_fasta": (FASTA.name, f, "text/plain")},
                        data={"seq_type": "nope"})
    assert r.status_code == 422


# ---- Inference: stats ----

def test_stats_end_to_end(client: httpx.Client, base_url: str) -> None:
    with open(FASTA, "rb") as f:
        submit = client.post("/api/stats", files={"input_fasta": (FASTA.name, f, "text/plain")})
    submit.raise_for_status()
    job = submit.json()
    job_id = job["job_id"]
    assert job["status"] in ("pending", "running")

    final = poll_job(client, base_url, job_id, timeout_s=TIMEOUT_S)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any(f.endswith("stats.tsv") for f in files), files

    # Deterministic content: the fixture has 4 sequences totalling 102 bp.
    r = client.get(f"/api/jobs/{job_id}/file/output/stats.tsv")
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    header = lines[0].split("\t")
    row = dict(zip(header, lines[1].split("\t")))
    assert row["num_seqs"] == "4"
    assert row["sum_len"] == "102"
    assert row["min_len"] == "8"
    assert row["max_len"] == "40"


# ---- Inference: revcomp ----

def test_revcomp_end_to_end(client: httpx.Client, base_url: str) -> None:
    with open(FASTA, "rb") as f:
        submit = client.post(
            "/api/revcomp",
            files={"input_fasta": (FASTA.name, f, "text/plain")},
            data={"seq_type": "dna"},
        )
    submit.raise_for_status()
    job_id = submit.json()["job_id"]

    final = poll_job(client, base_url, job_id, timeout_s=TIMEOUT_S)
    assert final["status"] == "completed", final

    r = client.get(f"/api/jobs/{job_id}/file/output/revcomp.fasta")
    r.raise_for_status()
    # seq1 revcomp (hand-checked) + the palindromic control must be unchanged.
    assert "TCAGGCTAACGGTCAGTTACGCAT" in r.text
    assert "AACCGGTT" in r.text

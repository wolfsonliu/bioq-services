"""End-to-end tests against a deployed diamond-server FC service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/diamond-server/tests/test_fc.py
    # or
    RUN_FC_TESTS=1 pytest services/diamond-server/tests/test_fc.py

Fixtures live in `tests/data/` so the suite is self-contained. URL is read from
`services/services.yaml` via `bioagent_service.fc_testing`.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
QUERY = DATA_DIR / "query.faa"
SUBJECT = DATA_DIR / "subject.faa"

pytestmark = pytest.mark.fc

TIMEOUT_S = 1800


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("diamond-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


def _submit_blastp(client: httpx.Client, **data) -> dict:
    with open(QUERY, "rb") as fq, open(SUBJECT, "rb") as fs:
        r = client.post(
            "/api/blastp",
            files={"query": (QUERY.name, fq, "text/plain"), "subject": (SUBJECT.name, fs, "text/plain")},
            data={"name": "fc_smoke", **data},
        )
    r.raise_for_status()
    return r.json()


# ---- Smoke ----

def test_healthz(client: httpx.Client) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "diamond"


def test_healthz_detail(client: httpx.Client) -> None:
    body = client.get("/healthz/detail").json()
    assert body["service"] == "diamond"
    assert "db_loaded" in body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    for p in ("/api/blastp", "/api/blastx", "/api/cluster", "/api/msa"):
        assert p in paths


def test_422_blastp_no_reference(client: httpx.Client) -> None:
    with open(QUERY, "rb") as fq:
        r = client.post("/api/blastp", files={"query": (QUERY.name, fq, "text/plain")}, data={"name": "x"})
    assert r.status_code == 422


# ---- Inference: blastp against an inline-built subject DB ----

def test_blastp_end_to_end(client: httpx.Client, base_url: str) -> None:
    submit = _submit_blastp(client)
    job_id = submit["job_id"]
    assert submit["status"] in ("pending", "running")

    final = poll_job(client, base_url, job_id, timeout_s=TIMEOUT_S)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any(f.endswith("fc_smoke.tsv") for f in files), files

    # The exact-match subject must appear among the hits.
    r = client.get(f"/api/jobs/{job_id}/file/fc_smoke.tsv")
    r.raise_for_status()
    assert "subj_exact" in r.text

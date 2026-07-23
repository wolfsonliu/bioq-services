"""End-to-end tests against a deployed plip-server FC service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/plip-server/tests/test_fc.py
    # or
    RUN_FC_TESTS=1 pytest services/plip-server/tests/test_fc.py

Fixtures live in `tests/data/` so the suite is self-contained. URL is read from
`services.yaml` via `bioq_service.fc_testing`.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
PDB = DATA_DIR / "1vsn.pdb"

pytestmark = pytest.mark.fc

TIMEOUT_S = 1800


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("plip-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# ---- Smoke ----

def test_healthz(client: httpx.Client) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "plip"


def test_healthz_detail(client: httpx.Client) -> None:
    body = client.get("/healthz/detail").json()
    assert body["service"] == "plip"
    assert body["ready"] is True, body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/profile" in paths


def test_422_bad_mode(client: httpx.Client) -> None:
    with open(PDB, "rb") as f:
        r = client.post("/api/profile", files={"input_pdb": (PDB.name, f, "chemical/x-pdb")}, data={"mode": "nope"})
    assert r.status_code == 422


# ---- Inference: default ligand-interaction profile ----

def test_profile_end_to_end(client: httpx.Client, base_url: str) -> None:
    with open(PDB, "rb") as f:
        submit = client.post(
            "/api/profile",
            files={"input_pdb": (PDB.name, f, "chemical/x-pdb")},
            data={"name": "fc_smoke"},
        )
    submit.raise_for_status()
    job = submit.json()
    job_id = job["job_id"]
    assert job["status"] in ("pending", "running")

    final = poll_job(client, base_url, job_id, timeout_s=TIMEOUT_S)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any(f.endswith("fc_smoke.xml") for f in files), files

    # The XML report must be well-formed PLIP output.
    r = client.get(f"/api/jobs/{job_id}/file/fc_smoke.xml")
    r.raise_for_status()
    assert "<report" in r.text

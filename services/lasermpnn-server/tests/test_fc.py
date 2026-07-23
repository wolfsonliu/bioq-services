"""End-to-end tests against the deployed LASErMPNN Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    RUN_FC_TESTS=1 pytest -m fc services/lasermpnn-server/tests/test_fc.py

URL resolves via `services/services.yaml`. Test PDB (4jnj-1_prot, a protein with
a bound small molecule) ships in `tests/data/` — copied from upstream so the
suite is self-contained. Inference calls keep designs_per_input small.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

TEST_PDB = Path(__file__).resolve().parent / "data" / "4jnj-1_prot.pdb"

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("lasermpnn-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


def _upload_pdb() -> dict:
    return {"pdb": (TEST_PDB.name, open(TEST_PDB, "rb"), "chemical/x-pdb")}


def _submit_and_poll(client, base_url, endpoint, *, data, files=None, timeout_s=1200):
    r = client.post(endpoint, data=data, files=files)
    r.raise_for_status()
    body = r.json()
    assert "job_id" in body
    job_id = body["job_id"]
    final = poll_job(client, base_url, job_id, timeout_s=timeout_s, interval_s=15)
    assert final["status"] == "completed", (
        f"failed: kind={final.get('failure_kind')} summary={final.get('error_summary')!r}"
    )
    files_list = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    return job_id, final, files_list


# ---- smoke ----

def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "lasermpnn"


def test_healthz_detail_weights(client):
    body = client.get("/healthz/detail").json()
    assert body["service"] == "lasermpnn"
    assert body.get("weights_loaded") is True, f"weights not mounted: {body.get('weights_missing')}"


def test_manifest_endpoints(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert {"/api/design", "/api/design_ligandmpnn"} <= paths


def test_openapi_served(client):
    assert "paths" in client.get("/openapi.json").json()


def test_unknown_job_404(client):
    assert client.get("/api/jobs/missing").status_code == 404


def test_422_missing_pdb(client):
    r = client.post("/api/design", data={"designs_per_input": "1"})
    assert r.status_code == 422


# ---- inference ----

def test_design_minimal(client, base_url):
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/design",
        data={"designs_per_input": "1", "designs_per_batch": "1", "sequence_temp": "0.3"},
        files=_upload_pdb(),
    )
    assert any(f.endswith(".pdb") for f in files), f"no design PDB: {files}"


def test_design_ligandmpnn_minimal(client, base_url):
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/design_ligandmpnn",
        data={"designs_per_input": "1", "designs_per_batch": "1"},
        files=_upload_pdb(),
    )
    assert files, "no output files"

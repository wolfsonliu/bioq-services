"""End-to-end tests against the deployed dockq-server Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/dockq-server/tests/test_fc.py
    # or
    RUN_FC_TESTS=1 pytest services/dockq-server/tests/test_fc.py

Test fixtures live in `tests/data/`, so the suite is self-contained — no
dependency on `opensource/DockQ` (which is gitignored).

URL is read from `services/aliyun_fc_url.md` via `bioagent_service.fc_testing`.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PDB = DATA_DIR / "model.pdb"
MODEL_ALT_PDB = DATA_DIR / "model_alt.pdb"
NATIVE_PDB = DATA_DIR / "native.pdb"

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("dockq-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# ----- Smoke -----

def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "dockq"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "dockq"
    assert "version" in body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert "/api/score" in paths
    assert "/api/score_batch" in paths


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ----- Inference: minimal job per endpoint -----

def _assert_completed(client: httpx.Client, base_url: str, job_id: str) -> dict:
    final = poll_job(client, base_url, job_id, timeout_s=600)
    assert final["status"] == "completed", final
    return final


def test_score_minimal_job(client: httpx.Client, base_url: str) -> None:
    with open(MODEL_PDB, "rb") as fm, open(NATIVE_PDB, "rb") as fn:
        r = client.post(
            "/api/score",
            files={
                "model": (MODEL_PDB.name, fm, "chemical/x-pdb"),
                "native": (NATIVE_PDB.name, fn, "chemical/x-pdb"),
            },
            data={"name": "fc_smoke"},
        )
    r.raise_for_status()
    final = _assert_completed(client, base_url, r.json()["job_id"])

    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert any(f.endswith("fc_smoke.json") for f in files), files


def test_score_batch_minimal_job(client: httpx.Client, base_url: str) -> None:
    with open(NATIVE_PDB, "rb") as fn, \
         open(MODEL_PDB, "rb") as fm1, \
         open(MODEL_ALT_PDB, "rb") as fm2:
        r = client.post(
            "/api/score_batch",
            data={"sort_by": "DockQ", "name": "fc_batch"},
            files=[
                ("native", (NATIVE_PDB.name, fn, "chemical/x-pdb")),
                ("models", (MODEL_PDB.name, fm1, "chemical/x-pdb")),
                ("models", (MODEL_ALT_PDB.name, fm2, "chemical/x-pdb")),
            ],
        )
    r.raise_for_status()
    final = _assert_completed(client, base_url, r.json()["job_id"])

    files = set(client.get(f"/api/jobs/{final['job_id']}/files").json()["files"])
    assert "scores.csv" in files, files
    assert any(p.startswith("per_model/") and p.endswith(".json") for p in files), files

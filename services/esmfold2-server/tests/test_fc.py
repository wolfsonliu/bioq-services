"""End-to-end tests against the deployed ESMFold2 Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/esmfold2-server/tests/test_fc.py
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("esmfold2-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "esmfold2"
    assert "version" in body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/fold"}


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


def test_fold_minimal_protein(client: httpx.Client, base_url: str) -> None:
    """Fold ubiquitin (76 aa) — smallest meaningful test."""
    ubiquitin = (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQK"
        "ESTLHLVLRLRGG"
    )
    r = client.post(
        "/api/fold",
        data={
            "sequences": f'[{{"type":"protein","id":"A","sequence":"{ubiquitin}"}}]',
            "num_sampling_steps": "10",
            "num_loops": "1",
        },
    )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    assert final["status"] == "completed", final
    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert any("prediction_0.cif" in f for f in files)
    assert any("metrics.json" in f for f in files)

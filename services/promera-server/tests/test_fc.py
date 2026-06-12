"""End-to-end tests against the deployed Promera Function Compute service.

Marked ``@pytest.mark.fc``, skipped by default. Run with:

    pytest -m fc services/promera-server/tests/test_fc.py

Test fixtures ship in ``tests/data/``, so the suite is self-contained.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_TARGET = DATA_DIR / "test_target.json"

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("promera-server", start=Path(__file__))


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
    assert body["service"] == "promera"
    assert "version" in body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/cofold", "/api/design"}


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ----- Cofold inference -----


def test_cofold_minimal(client: httpx.Client, base_url: str) -> None:
    with open(TEST_TARGET, "rb") as fh:
        r = client.post(
            "/api/cofold",
            files={"input_schema": ("ubiquitin.json", fh, "application/json")},
            data={"num_seeds": "1", "diffusion_samples": "1", "diffusion_steps": "50"},
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    assert final["status"] == "completed", final
    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert any(f.endswith(".cif") for f in files)


# ----- Design inference -----


def test_design_minibinder_minimal(client: httpx.Client, base_url: str) -> None:
    with open(TEST_TARGET, "rb") as fh:
        r = client.post(
            "/api/design",
            files={"target_schema": ("target.json", fh, "application/json")},
            data={
                "design_type": "minibinder",
                "num_backbones": "1",
                "diffusion_steps": "50",
                "inverse_folder_type": "none",
            },
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    assert final["status"] == "completed", final
    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert any("backbone.cif" in f for f in files)

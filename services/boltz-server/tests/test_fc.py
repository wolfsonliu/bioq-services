"""End-to-end tests against the deployed Boltz Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/boltz-server/tests/test_fc.py

The base URL is read from `services/aliyun_fc_url.md` — update that file after
deploying a new tag in the FC console. Inference tests use `msa_mode=empty`
so they don't depend on ColabFold's MSA server (which can be slow/flaky from
inside FC).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("boltz-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# ----- Smoke (no GPU work) -----

def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "boltz"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "boltz"


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert "/api/predict_structure" in paths
    assert "/api/predict_affinity" in paths


def test_manifest_model_is_boltz2(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    assert body["service_specific"]["model"]["name"] == "boltz2"


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id-fc").status_code == 404


# ----- Inference: one minimal job per endpoint -----

def test_predict_structure_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Smallest possible structure prediction — single short protein, no MSA, 1 sample."""
    payload = {
        "name": "fc_smoke",
        "msa_mode": "empty",
        "diffusion_samples": "1",
        "recycling_steps": "1",
        "sampling_steps": "50",
        "sequences": json.dumps(
            [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": (
                        "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSC"
                    ),
                    "msa_uri": "empty",
                }
            ]
        ),
    }
    r = client.post("/api/predict_structure", data=payload)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    final = poll_job(client, base_url, job_id, timeout_s=1800, interval_s=20)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any(
        f.endswith(".cif") or f.endswith(".pdb")
        for f in files
    ), f"no model file in outputs: {files}"


def test_predict_affinity_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Smallest possible affinity prediction — short protein + benzene SMILES."""
    payload = {
        "name": "fc_aff_smoke",
        "binder_id": "B",
        "msa_mode": "empty",
        "diffusion_samples": "1",
        "recycling_steps": "1",
        "sampling_steps": "50",
        "diffusion_samples_affinity": "1",
        "sampling_steps_affinity": "50",
        "sequences": json.dumps(
            [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": (
                        "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSC"
                    ),
                    "msa_uri": "empty",
                },
                {"type": "ligand", "id": "B", "smiles": "c1ccccc1"},
            ]
        ),
    }
    r = client.post("/api/predict_affinity", data=payload)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    final = poll_job(client, base_url, job_id, timeout_s=1800, interval_s=20)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any("affinity_" in f and f.endswith(".json") for f in files), \
        f"no affinity json in outputs: {files}"

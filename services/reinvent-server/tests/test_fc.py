"""End-to-end tests against the deployed reinvent-server Function Compute service.

Marked ``@pytest.mark.fc``, skipped by default. Run with::

    pytest -m fc services/reinvent-server/tests/test_fc.py
    # or
    RUN_FC_TESTS=1 pytest services/reinvent-server/tests/test_fc.py

URL is read from ``services/services.yaml`` via ``bioq_service.fc_testing``
(mirrors dockq-server/tests/test_fc.py). These tests use only the ``reinvent``
generator with a small ``num_smiles`` — no file upload, tiny payload — so they
exercise the sync submit/poll path without needing large priors staged.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "reinvent-server"

DATA_DIR = Path(__file__).resolve().parent / "data"

pytestmark = pytest.mark.fc

# Sampling from the Reinvent prior is quick, but FC cold-start + model load can
# take a while on first hit; keep a generous ceiling.
SAMPLING_TIMEOUT_S = 1800


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# =====================================================================
# Helpers (mirror dockq-server/tests/test_fc.py)
# =====================================================================

def _assert_submitted(resp_json: dict) -> None:
    assert "job_id" in resp_json
    assert resp_json["status"] in ("pending", "running")
    assert resp_json["created_at"] is not None
    assert resp_json["input_params"] is not None
    assert isinstance(resp_json["input_params"], dict)


def _assert_completed(
    client: httpx.Client,
    base_url: str,
    job_id: str,
    *,
    timeout_s: int = SAMPLING_TIMEOUT_S,
) -> dict:
    final = poll_job(client, base_url, job_id, timeout_s=timeout_s)
    assert final["status"] == "completed", (
        f"failed: kind={final.get('failure_kind')} "
        f"summary={final.get('error_summary')!r}"
    )
    assert final["created_at"] is not None
    assert final["started_at"] is not None
    assert final["completed_at"] is not None
    assert final["duration_seconds"] is not None
    assert final["duration_seconds"] > 0
    assert final["output_count"] is not None
    assert final["output_count"] > 0
    assert final["output_total_bytes"] is not None
    assert final["output_total_bytes"] > 0
    return final


def _download_text(client: httpx.Client, job_id: str, file_path: str) -> str:
    r = client.get(f"/api/jobs/{job_id}/file/{file_path}")
    r.raise_for_status()
    return r.text


def _submit_sampling(client: httpx.Client, **extra_data) -> dict:
    """Submit a reinvent sampling job (no file upload) and return the JSON."""
    r = client.post(
        "/api/sampling",
        data={"generator": "reinvent", "num_smiles": "20", **extra_data},
    )
    r.raise_for_status()
    return r.json()


# =====================================================================
# Smoke
# =====================================================================

def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "reinvent"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "reinvent"
    assert "version" in body
    assert "active_jobs" in body
    assert isinstance(body["active_jobs"], int)
    assert "max_concurrent_jobs" in body
    assert body["task_endpoints_enabled"] is True
    # Shape-only assertions: priors_loaded / cuda_available depend on the NAS
    # mount + GPU on the deployed instance, so a fresh deploy may report False.
    assert "priors_loaded" in body
    assert isinstance(body["priors_loaded"], bool)
    assert "priors_missing" in body
    assert "prior_base" in body
    assert "cuda_available" in body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert "/api/sampling" in paths
    assert "/api/scoring" in paths
    assert any(p.startswith("/api/tasks/") for p in paths)


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    spec = r.json()
    assert "paths" in spec
    assert "/api/sampling" in spec["paths"]


# =====================================================================
# Inference: sampling (reinvent prior, small num_smiles, no file)
# =====================================================================

def test_sampling_minimal_job(client: httpx.Client, base_url: str) -> None:
    submit = _submit_sampling(client)
    _assert_submitted(submit)
    assert submit["input_params"]["generator"] == "reinvent"
    assert submit["input_params"]["num_smiles"] == 20

    final = _assert_completed(client, base_url, submit["job_id"])

    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert any(f.endswith("sampling.csv") for f in files), files


def test_sampling_csv_content(client: httpx.Client, base_url: str) -> None:
    """sampling.csv exists and is non-empty (has at least one SMILES row)."""
    submit = _submit_sampling(client)
    _assert_submitted(submit)

    final = _assert_completed(client, base_url, submit["job_id"])

    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    csv_path = next(f for f in files if f.endswith("sampling.csv"))

    csv_text = _download_text(client, final["job_id"], csv_path)
    assert csv_text.strip(), "sampling.csv should be non-empty"

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert len(rows) > 0, f"sampling.csv has no data rows: {csv_text[:200]!r}"
    # REINVENT sampling output carries a SMILES column (name varies by version).
    smiles_col = next(
        (c for c in (reader.fieldnames or []) if "smiles" in c.lower()), None
    )
    assert smiles_col is not None, f"no SMILES column: {reader.fieldnames}"

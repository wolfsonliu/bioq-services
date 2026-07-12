"""End-to-end tests against the deployed SemlaFlow Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/semlaflow-server/tests/test_fc.py -v

URL resolves via `services/services.yaml`.

Unconditional generation — no file uploads.  All params in form fields.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, make_retrying_client, poll_job

pytestmark = pytest.mark.fc

# qm9: n_molecules=10 / integration_steps=50 ~1-2 min incl. novelty ref build.
# geom-drugs novelty is much slower (see design §12.1); this file only uses qm9.
INFERENCE_TIMEOUT_S = 1800


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("semlaflow-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with make_retrying_client(base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# =====================================================================
# Helpers
# =====================================================================


def _assert_submitted(body: dict) -> None:
    assert "job_id" in body
    assert body["status"] in ("pending", "running")
    assert body["created_at"] is not None
    assert body["input_params"] is not None


def _assert_completed(final: dict, client: httpx.Client) -> list[str]:
    assert final["status"] == "completed", (
        f"failed: kind={final.get('failure_kind')} "
        f"summary={final.get('error_summary')!r}"
    )
    assert final["duration_seconds"] is not None and final["duration_seconds"] > 0
    assert final["output_count"] is not None and final["output_count"] > 0
    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert files, "no output files"
    return files


def _submit_and_poll(
    client: httpx.Client, base_url: str, data: dict,
    *, timeout_s: int = INFERENCE_TIMEOUT_S,
) -> tuple[str, dict, list[str]]:
    r = client.post("/api/generate", data=data)
    r.raise_for_status()
    body = r.json()
    _assert_submitted(body)
    job_id = body["job_id"]
    final = poll_job(client, base_url, job_id, timeout_s=timeout_s, interval_s=15)
    files = _assert_completed(final, client)
    return job_id, final, files


def _count_sdf_mols(sdf_bytes: bytes) -> int:
    return sdf_bytes.decode("utf-8", errors="replace").count("$$$$")


# =====================================================================
# Smoke (no GPU work)
# =====================================================================


def test_healthz(client: httpx.Client) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "semlaflow"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "semlaflow"
    assert "weights_dir" in body
    assert body["models"].get("qm9", {}).get("ready") is True, (
        f"qm9 model not ready on NAS: {body}"
    )


def test_api_models(client: httpx.Client) -> None:
    body = client.get("/api/models").json()
    names = {m["name"] for m in body["models"]}
    assert "qm9" in names


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/generate" in paths
    assert "/api/tasks/generate" in paths


def test_manifest_model_info(client: httpx.Client) -> None:
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "SemlaFlow"
    assert "generate" in extras["tool_outputs"]


def test_openapi_served(client: httpx.Client) -> None:
    spec = client.get("/openapi.json").json()
    for path in ("/api/generate", "/api/tasks/generate", "/api/models"):
        assert path in spec["paths"], f"{path} missing from OpenAPI spec"


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404
    assert client.get("/api/jobs/missing-job-id/files").status_code == 404


# =====================================================================
# 422 Error inputs (fast, no GPU)
# =====================================================================


def test_422_bad_model(client: httpx.Client) -> None:
    r = client.post("/api/generate", data={"model_name": "not_a_model"})
    assert r.status_code == 422


def test_422_n_molecules_too_low(client: httpx.Client) -> None:
    r = client.post("/api/generate", data={"n_molecules": "0"})
    assert r.status_code == 422


def test_422_integration_steps_out_of_range(client: httpx.Client) -> None:
    r = client.post("/api/generate", data={"integration_steps": "5"})
    assert r.status_code == 422


# =====================================================================
# Inference: fast smoke (qm9, 10 mols, 50 steps)
# =====================================================================


def test_generate_fast_smoke(client: httpx.Client, base_url: str) -> None:
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={
            "model_name": "qm9",
            "n_molecules": "10",
            "integration_steps": "50",
            "seed": "42",
        },
    )

    assert "predictions.smol.sdf" in files
    assert "metrics.json" in files
    assert "generation_stats.json" in files

    sdf = client.get(f"/api/jobs/{job_id}/file/predictions.smol.sdf")
    sdf.raise_for_status()
    n_mols = _count_sdf_mols(sdf.content)
    assert 1 <= n_mols <= 10, f"expected 1-10 mols, got {n_mols}"

    stats = client.get(f"/api/jobs/{job_id}/file/generation_stats.json").json()
    assert stats["n_requested"] == 10
    assert stats["n_valid"] == n_mols
    assert stats["seed"] == 42

    metrics = client.get(f"/api/jobs/{job_id}/file/metrics.json").json()
    assert "validity" in metrics


# =====================================================================
# Job lifecycle
# =====================================================================


def test_job_download_zip(client: httpx.Client, base_url: str) -> None:
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={"model_name": "qm9", "n_molecules": "5", "integration_steps": "50"},
    )
    r = client.get(f"/api/jobs/{job_id}/download")
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert any("predictions.smol.sdf" in n for n in names)


def test_job_log_endpoint(client: httpx.Client, base_url: str) -> None:
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={"model_name": "qm9", "n_molecules": "5", "integration_steps": "50"},
    )
    r = client.get(f"/api/jobs/{job_id}/log")
    r.raise_for_status()
    body = r.json()
    log_text = body.get("log") or body.get("text") or ""
    assert len(log_text) > 0

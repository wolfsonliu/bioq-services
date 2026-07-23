"""End-to-end tests against the deployed FlowMol Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/flowmol-server/tests/test_fc.py -v

URL resolves via `services.yaml`.

Unconditional generation — no file uploads.  All params in form fields.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, make_retrying_client, poll_job

pytestmark = pytest.mark.fc

# Sampling is fast on T4: n_mols=10 / n_timesteps=100 ~5-10 s; full 100 mols
# / 250 steps ~30-60 s. 600 s covers even the 1000-mol tail.
INFERENCE_TIMEOUT_S = 600


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("flowmol-server", start=Path(__file__))


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
    final = poll_job(client, base_url, job_id, timeout_s=timeout_s, interval_s=10)
    files = _assert_completed(final, client)
    return job_id, final, files


def _count_sdf_mols(sdf_bytes: bytes) -> int:
    """Count '$$$$' record separators in an SDF."""
    return sdf_bytes.decode("utf-8", errors="replace").count("$$$$")


# =====================================================================
# Smoke (no GPU work)
# =====================================================================


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "flowmol"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "flowmol"
    assert "weights_dir" in body
    assert body["weights_loaded"] is True, f"NAS weights probe failed: {body}"
    assert "flowmol3" in body["staged_variants"]
    assert isinstance(body["active_jobs"], int)
    assert isinstance(body["max_concurrent_jobs"], int)


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert "/api/generate" in paths
    assert "/api/tasks/generate" in paths


def test_manifest_model_info(client: httpx.Client) -> None:
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "FlowMol3"
    assert "flowmol3" in extras["model"]["primary_variants"]
    assert "generate" in extras["tool_outputs"]


def test_openapi_served(client: httpx.Client) -> None:
    spec = client.get("/openapi.json").json()
    for path in ("/api/generate", "/api/tasks/generate"):
        assert path in spec["paths"], f"{path} missing from OpenAPI spec"


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404
    assert client.get("/api/jobs/missing-job-id/files").status_code == 404
    assert client.get("/api/jobs/missing-job-id/download").status_code == 404


# =====================================================================
# 422 Error inputs (fast, no GPU)
# =====================================================================


def test_422_bad_variant(client: httpx.Client) -> None:
    r = client.post("/api/generate", data={"model_variant": "not_a_real_variant"})
    assert r.status_code == 422


def test_422_n_mols_too_low(client: httpx.Client) -> None:
    r = client.post("/api/generate", data={"n_mols": "0"})
    assert r.status_code == 422


def test_422_n_mols_too_high(client: httpx.Client) -> None:
    r = client.post("/api/generate", data={"n_mols": "10000"})
    assert r.status_code == 422


def test_422_hc_thresh_out_of_range(client: httpx.Client) -> None:
    r = client.post("/api/generate", data={"hc_thresh": "1.5"})
    assert r.status_code == 422


# =====================================================================
# Inference: fast smoke (10 mols, 100 steps)
# =====================================================================


def test_generate_fast_smoke(client: httpx.Client, base_url: str) -> None:
    """Small n_mols + short n_timesteps → quick CI-grade regression."""
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={"n_mols": "10", "n_timesteps": "100", "seed": "42"},
    )

    assert "molecules.sdf" in files
    assert "sampling_stats.json" in files

    sdf = client.get(f"/api/jobs/{job_id}/file/molecules.sdf")
    sdf.raise_for_status()
    n_mols = _count_sdf_mols(sdf.content)
    assert 1 <= n_mols <= 10, f"expected 1-10 mols, got {n_mols}"

    stats = client.get(f"/api/jobs/{job_id}/file/sampling_stats.json").json()
    assert stats["n_requested"] == 10
    assert stats["n_written"] == n_mols
    assert stats["seed"] == 42


# =====================================================================
# Inference: paper-scale sampling
# =====================================================================


def test_generate_default_size(client: httpx.Client, base_url: str) -> None:
    """Default params (n_mols=100, n_timesteps=250) — typical production call."""
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={"n_mols": "100", "n_timesteps": "250"},
    )

    sdf = client.get(f"/api/jobs/{job_id}/file/molecules.sdf")
    n_mols = _count_sdf_mols(sdf.content)
    assert n_mols >= 80, (
        f"expected FlowMol3 to hit ≥80% valid on default settings; got {n_mols}/100"
    )


# =====================================================================
# Inference: fixed atom count
# =====================================================================


def test_generate_fixed_n_atoms(client: httpx.Client, base_url: str) -> None:
    """n_atoms_per_mol pins every molecule's atom count."""
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={
            "n_mols": "10",
            "n_timesteps": "100",
            "n_atoms_per_mol": "25",
            "seed": "7",
        },
    )

    stats = client.get(f"/api/jobs/{job_id}/file/sampling_stats.json").json()
    assert stats["n_atoms_per_mol"] == 25


# =====================================================================
# Job lifecycle
# =====================================================================


def test_job_download_zip(client: httpx.Client, base_url: str) -> None:
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={"n_mols": "5", "n_timesteps": "100"},
    )
    r = client.get(f"/api/jobs/{job_id}/download")
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert any("molecules.sdf" in n for n in names)


def test_job_log_endpoint(client: httpx.Client, base_url: str) -> None:
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={"n_mols": "5", "n_timesteps": "100"},
    )
    r = client.get(f"/api/jobs/{job_id}/log")
    r.raise_for_status()
    body = r.json()
    log_text = body.get("log") or body.get("text") or ""
    assert len(log_text) > 0

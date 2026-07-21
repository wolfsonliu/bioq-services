"""End-to-end tests against the deployed Megalodon Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/megalodon-server/tests/test_fc.py -v

URL resolves via `services/services.yaml`.

Unconditional generation — no file uploads. All params in form fields.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, make_retrying_client, poll_job

pytestmark = pytest.mark.fc

# drugs_diffusion n=10 / timesteps=100 incl. cold start + NAS load + metrics
# (train_smiles novelty build). Generous ceiling (see design §12.1).
INFERENCE_TIMEOUT_S = 1800

# Which variant to exercise for real generation. Must be staged/ready on NAS.
SMOKE_MODEL = "drugs_diffusion"


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("megalodon-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with make_retrying_client(base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


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


# ----- Smoke (no GPU work) -----


def test_healthz(client: httpx.Client) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "megalodon"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "megalodon"
    assert "weights_dir" in body
    assert body["models"].get(SMOKE_MODEL, {}).get("ready") is True, (
        f"{SMOKE_MODEL} not ready on NAS: {body}"
    )


def test_api_models(client: httpx.Client) -> None:
    body = client.get("/api/models").json()
    names = {m["name"] for m in body["models"]}
    assert SMOKE_MODEL in names


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/generate" in paths
    assert "/api/tasks/generate" in paths


def test_manifest_model_info(client: httpx.Client) -> None:
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "Megalodon"
    assert "generate" in extras["tool_outputs"]


def test_openapi_served(client: httpx.Client) -> None:
    spec = client.get("/openapi.json").json()
    for path in ("/api/generate", "/api/tasks/generate", "/api/models"):
        assert path in spec["paths"], f"{path} missing from OpenAPI spec"


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404
    assert client.get("/api/jobs/missing-job-id/files").status_code == 404


# ----- 422 Error inputs (fast, no GPU) -----


def test_422_bad_model(client: httpx.Client) -> None:
    assert client.post("/api/generate", data={"model_name": "not_a_model"}).status_code == 422


def test_422_n_molecules_too_low(client: httpx.Client) -> None:
    assert client.post("/api/generate", data={"n_molecules": "0"}).status_code == 422


def test_422_timesteps_out_of_range(client: httpx.Client) -> None:
    assert client.post("/api/generate", data={"timesteps": "5"}).status_code == 422


# ----- Inference: fast smoke (drugs_diffusion, 10 mols, 100 steps) -----


def test_generate_fast_smoke(client: httpx.Client, base_url: str) -> None:
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={
            "model_name": SMOKE_MODEL,
            "n_molecules": "10",
            "timesteps": "100",
            "seed": "42",
        },
    )

    assert "generated_molecules.sdf" in files
    assert "generation_stats.json" in files

    sdf = client.get(f"/api/jobs/{job_id}/file/generated_molecules.sdf")
    sdf.raise_for_status()
    n_mols = _count_sdf_mols(sdf.content)
    assert 1 <= n_mols <= 10, f"expected 1-10 mols, got {n_mols}"

    stats = client.get(f"/api/jobs/{job_id}/file/generation_stats.json").json()
    assert stats["n_requested"] == 10
    assert stats["n_valid"] == n_mols
    assert stats["seed"] == 42


def test_generate_fixed_atom_count(client: httpx.Client, base_url: str) -> None:
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={
            "model_name": SMOKE_MODEL,
            "n_molecules": "5",
            "timesteps": "100",
            "n_atoms_per_mol": "20",
        },
    )
    stats = client.get(f"/api/jobs/{job_id}/file/generation_stats.json").json()
    assert stats["n_atoms_per_mol"] == 20


# ----- Job lifecycle -----


def test_job_download_zip(client: httpx.Client, base_url: str) -> None:
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={"model_name": SMOKE_MODEL, "n_molecules": "5", "timesteps": "100"},
    )
    r = client.get(f"/api/jobs/{job_id}/download")
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert any("generated_molecules.sdf" in n for n in zf.namelist())


def test_job_log_endpoint(client: httpx.Client, base_url: str) -> None:
    job_id, final, files = _submit_and_poll(
        client, base_url,
        data={"model_name": SMOKE_MODEL, "n_molecules": "5", "timesteps": "100"},
    )
    r = client.get(f"/api/jobs/{job_id}/log")
    r.raise_for_status()
    body = r.json()
    log_text = body.get("log") or body.get("text") or ""
    assert len(log_text) > 0

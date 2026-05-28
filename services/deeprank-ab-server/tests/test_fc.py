"""End-to-end tests against the deployed DeepRank-Ab Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/deeprank-ab-server/tests/test_fc.py

DeepRank-Ab has one endpoint:
  * `/api/score` — score an antibody-antigen docking complex PDB

Test PDB ships in `tests/data/` (copied from upstream example/).
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_PDB = DATA_DIR / "test.pdb"

pytestmark = pytest.mark.fc

SCORE_TIMEOUT_S = 3600


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("deeprank-ab-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_submitted(resp_json: dict) -> None:
    assert "job_id" in resp_json
    assert resp_json["status"] in ("pending", "running")
    assert resp_json["created_at"] is not None
    assert resp_json["input_params"] is not None
    assert isinstance(resp_json["input_params"], dict)


def _assert_completed(
    client: httpx.Client, base_url: str, job_id: str, *, timeout_s: int = SCORE_TIMEOUT_S,
) -> dict:
    final = poll_job(client, base_url, job_id, timeout_s=timeout_s)
    assert final["status"] == "completed", (
        f"failed: kind={final.get('failure_kind')} summary={final.get('error_summary')!r}"
    )

    assert final["created_at"] is not None
    assert final["started_at"] is not None
    assert final["completed_at"] is not None
    assert final["duration_seconds"] is not None
    assert final["duration_seconds"] > 0
    assert final["input_params"] is not None
    assert isinstance(final["input_params"], dict)
    assert final["output_count"] is not None
    assert final["output_count"] > 0
    assert final["output_total_bytes"] is not None
    assert final["output_total_bytes"] > 0
    return final


def _submit_score(client: httpx.Client, **extra_data) -> dict:
    with open(TEST_PDB, "rb") as fh:
        r = client.post(
            "/api/score",
            files={"input_pdb": (TEST_PDB.name, fh, "chemical/x-pdb")},
            data={"heavy_chain_id": "H", "light_chain_id": "L", "antigen_chain_id": "A", **extra_data},
        )
    r.raise_for_status()
    return r.json()


def _download_text(client: httpx.Client, job_id: str, file_path: str) -> str:
    r = client.get(f"/api/jobs/{job_id}/file/{file_path}")
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------------------------
# Smoke (no inference compute)
# ---------------------------------------------------------------------------


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "deeprank-ab"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "deeprank-ab"
    assert "version" in body
    assert body["jobs_base_dir_exists"] is True
    assert "disk_usage_mb" in body
    assert "disk_limit_mb" in body
    if "active_jobs" in body:
        assert isinstance(body["active_jobs"], int)
        assert "max_concurrent_jobs" in body


def test_manifest_lists_score_endpoint(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/score"}


def test_manifest_service_specific(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    extras = body["service_specific"]
    assert "tool_outputs" in extras
    assert "score" in extras["tool_outputs"]
    assert "scoring_legend" in extras
    assert "predicted_dockq" in extras["scoring_legend"]
    assert "quality_flag" in extras["scoring_legend"]
    assert "input_uri_schemes" in extras
    assert "config_tips" in extras
    assert "model_info" in extras
    assert "EGNN" in extras["model_info"]["architecture"]
    assert "ESM-2" in extras["model_info"]["sequence_encoder"]


def test_manifest_endpoint_examples(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert by_path["/api/score"]["examples"]
    assert len(by_path["/api/score"]["examples"]) >= 2


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    spec = r.json()
    assert "paths" in spec
    assert "/api/score" in spec["paths"]


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404
    assert client.get("/api/jobs/missing-job-id/files").status_code == 404
    assert client.get("/api/jobs/missing-job-id/log").status_code == 404
    assert client.get("/api/jobs/missing-job-id/download").status_code == 404
    assert client.get("/api/jobs/missing-job-id/file/foo.csv").status_code == 404


# ---------------------------------------------------------------------------
# 422 Error inputs (fast, no inference compute)
# ---------------------------------------------------------------------------


def test_422_score_missing_input(client: httpx.Client) -> None:
    """Neither upload nor URI → 422."""
    r = client.post("/api/score", data={
        "heavy_chain_id": "H", "light_chain_id": "L", "antigen_chain_id": "A",
    })
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Inference: score antibody-antigen complex
# ---------------------------------------------------------------------------


def test_score_antibody_antigen(client: httpx.Client, base_url: str) -> None:
    """Score the example antibody-antigen complex (chains H, L, A)."""
    submit = _submit_score(client)
    _assert_submitted(submit)
    assert submit["input_params"]["heavy_chain_id"] == "H"
    assert submit["input_params"]["light_chain_id"] == "L"
    assert submit["input_params"]["antigen_chain_id"] == "A"

    final = _assert_completed(client, base_url, submit["job_id"])

    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    csv_files = [f for f in files if f.endswith("_predictions.csv")]
    assert csv_files, f"no predictions CSV in output: {files}"


def test_score_output_csv_content(client: httpx.Client, base_url: str) -> None:
    """Validate predictions CSV has correct structure and plausible values."""
    submit = _submit_score(client)
    _assert_submitted(submit)

    final = _assert_completed(client, base_url, submit["job_id"])

    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    csv_files = [f for f in files if f.endswith("_predictions.csv")]
    assert csv_files, f"no predictions CSV: {files}"

    csv_text = _download_text(client, final["job_id"], csv_files[0])
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)

    assert len(rows) >= 1, "predictions CSV should have at least one row"
    assert "predicted_dockq" in reader.fieldnames, f"missing predicted_dockq column: {reader.fieldnames}"
    assert "quality_flag" in reader.fieldnames, f"missing quality_flag column: {reader.fieldnames}"

    for row in rows:
        dockq = float(row["predicted_dockq"])
        assert 0.0 <= dockq <= 1.0, f"predicted_dockq={dockq} out of [0,1] range"
        assert row["quality_flag"] in ("ok", "low_HL_contacts", "not_applicable"), (
            f"unexpected quality_flag: {row['quality_flag']!r}"
        )


def test_score_output_has_hdf5(client: httpx.Client, base_url: str) -> None:
    """Inference should also produce graph and prediction HDF5 files."""
    submit = _submit_score(client)
    final = _assert_completed(client, base_url, submit["job_id"])

    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    hdf5_files = [f for f in files if f.endswith(".hdf5")]
    assert hdf5_files, f"no HDF5 files in output: {files}"


def test_score_nanobody(client: httpx.Client, base_url: str) -> None:
    """Score with light_chain_id='-' (nanobody / VHH mode)."""
    with open(TEST_PDB, "rb") as fh:
        r = client.post(
            "/api/score",
            files={"input_pdb": (TEST_PDB.name, fh, "chemical/x-pdb")},
            data={
                "heavy_chain_id": "H",
                "light_chain_id": "-",
                "antigen_chain_id": "A",
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["light_chain_id"] == "-"

    final = _assert_completed(client, base_url, submit["job_id"])

    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    csv_files = [f for f in files if f.endswith("_predictions.csv")]
    assert csv_files, f"no predictions CSV in nanobody output: {files}"

    csv_text = _download_text(client, final["job_id"], csv_files[0])
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        assert row["quality_flag"] in ("ok", "not_applicable"), (
            f"nanobody quality_flag should be 'ok' or 'not_applicable', got {row['quality_flag']!r}"
        )


def test_score_default_chain_ids(client: httpx.Client, base_url: str) -> None:
    """Submit without explicit chain IDs — defaults should apply (H, L, A)."""
    with open(TEST_PDB, "rb") as fh:
        r = client.post(
            "/api/score",
            files={"input_pdb": (TEST_PDB.name, fh, "chemical/x-pdb")},
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["heavy_chain_id"] == "H"
    assert submit["input_params"]["light_chain_id"] == "L"
    assert submit["input_params"]["antigen_chain_id"] == "A"

    _assert_completed(client, base_url, submit["job_id"])


# ---------------------------------------------------------------------------
# Job lifecycle endpoints
# ---------------------------------------------------------------------------


def test_job_status_polling(client: httpx.Client, base_url: str) -> None:
    """Job transitions from pending/running → completed; intermediate polls return valid JobInfo."""
    submit = _submit_score(client)
    job_id = submit["job_id"]

    r = client.get(f"/api/jobs/{job_id}")
    r.raise_for_status()
    info = r.json()
    assert info["job_id"] == job_id
    assert info["status"] in ("pending", "running", "completed")

    _assert_completed(client, base_url, job_id)


def test_job_files_endpoint(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/files returns a list of output file paths."""
    submit = _submit_score(client)
    _assert_completed(client, base_url, submit["job_id"])

    r = client.get(f"/api/jobs/{submit['job_id']}/files")
    r.raise_for_status()
    body = r.json()
    assert body["job_id"] == submit["job_id"]
    assert isinstance(body["files"], list)
    assert len(body["files"]) > 0
    assert any(f.endswith("_predictions.csv") for f in body["files"])


def test_job_single_file_download(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/file/{path} returns the file content."""
    submit = _submit_score(client)
    _assert_completed(client, base_url, submit["job_id"])

    files = client.get(f"/api/jobs/{submit['job_id']}/files").json()["files"]
    csv_file = next(f for f in files if f.endswith("_predictions.csv"))

    r = client.get(f"/api/jobs/{submit['job_id']}/file/{csv_file}")
    r.raise_for_status()
    assert "predicted_dockq" in r.text


def test_job_log_endpoint(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/log returns log text."""
    submit = _submit_score(client)
    _assert_completed(client, base_url, submit["job_id"])

    r = client.get(f"/api/jobs/{submit['job_id']}/log")
    r.raise_for_status()
    body = r.json()
    assert body["job_id"] == submit["job_id"]
    assert "log" in body
    assert isinstance(body["log"], str)


def test_job_download_zip(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/download returns a valid zip archive."""
    submit = _submit_score(client)
    _assert_completed(client, base_url, submit["job_id"])

    r = client.get(f"/api/jobs/{submit['job_id']}/download")
    r.raise_for_status()
    assert "application/zip" in r.headers.get("content-type", "")

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert len(names) > 0
    assert any(n.endswith("_predictions.csv") for n in names)


def test_job_file_not_found(client: httpx.Client, base_url: str) -> None:
    """Requesting a non-existent file within a valid job → 404."""
    submit = _submit_score(client)
    _assert_completed(client, base_url, submit["job_id"])

    r = client.get(f"/api/jobs/{submit['job_id']}/file/nonexistent.xyz")
    assert r.status_code == 404


def test_job_delete(client: httpx.Client, base_url: str) -> None:
    """DELETE /api/jobs/{id} removes the job; subsequent GET returns 404."""
    submit = _submit_score(client)
    _assert_completed(client, base_url, submit["job_id"])

    r = client.delete(f"/api/jobs/{submit['job_id']}")
    r.raise_for_status()
    assert r.json()["status"] == "deleted"

    assert client.get(f"/api/jobs/{submit['job_id']}").status_code == 404

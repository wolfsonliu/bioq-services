"""End-to-end tests against the deployed dockq-server Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/dockq-server/tests/test_fc.py
    # or
    RUN_FC_TESTS=1 pytest services/dockq-server/tests/test_fc.py

Test fixtures live in `tests/data/`, so the suite is self-contained — no
dependency on `opensource/DockQ` (which is gitignored).

URL is read from `services/services.yaml` via `bioq_service.fc_testing`.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PDB = DATA_DIR / "model.pdb"
MODEL_ALT_PDB = DATA_DIR / "model_alt.pdb"
NATIVE_PDB = DATA_DIR / "native.pdb"

pytestmark = pytest.mark.fc

DOCKQ_TIMEOUT_S = 3600
BATCH_TIMEOUT_S = 7200


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("dockq-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# =====================================================================
# Helpers
# =====================================================================

def _assert_submitted(resp_json: dict, *, expect_input_params: bool = True) -> None:
    """Validate the immediate POST response has expected fields."""
    assert "job_id" in resp_json
    assert resp_json["status"] in ("pending", "running")
    assert resp_json["created_at"] is not None
    if expect_input_params:
        assert resp_json["input_params"] is not None
        assert isinstance(resp_json["input_params"], dict)


def _assert_completed(
    client: httpx.Client, base_url: str, job_id: str, *, timeout_s: int = DOCKQ_TIMEOUT_S,
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
    """Submit a single score job and return the response JSON."""
    with open(MODEL_PDB, "rb") as fm, open(NATIVE_PDB, "rb") as fn:
        r = client.post(
            "/api/score",
            files={
                "model": (MODEL_PDB.name, fm, "chemical/x-pdb"),
                "native": (NATIVE_PDB.name, fn, "chemical/x-pdb"),
            },
            data={"name": "fc_test", **extra_data},
        )
    r.raise_for_status()
    return r.json()


def _submit_batch(client: httpx.Client, **extra_data) -> dict:
    """Submit a batch score job and return the response JSON."""
    with open(NATIVE_PDB, "rb") as fn, \
         open(MODEL_PDB, "rb") as fm1, \
         open(MODEL_ALT_PDB, "rb") as fm2:
        r = client.post(
            "/api/score_batch",
            data={"sort_by": "DockQ", "name": "fc_batch_test", **extra_data},
            files=[
                ("native", (NATIVE_PDB.name, fn, "chemical/x-pdb")),
                ("models", (MODEL_PDB.name, fm1, "chemical/x-pdb")),
                ("models", (MODEL_ALT_PDB.name, fm2, "chemical/x-pdb")),
            ],
        )
    r.raise_for_status()
    return r.json()


def _download_text(client: httpx.Client, job_id: str, file_path: str) -> str:
    r = client.get(f"/api/jobs/{job_id}/file/{file_path}")
    r.raise_for_status()
    return r.text


def _download_json(client: httpx.Client, job_id: str, file_path: str) -> dict:
    r = client.get(f"/api/jobs/{job_id}/file/{file_path}")
    r.raise_for_status()
    return r.json()


# =====================================================================
# Smoke (no DockQ compute)
# =====================================================================

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
    assert "active_jobs" in body
    assert isinstance(body["active_jobs"], int)
    assert "max_concurrent_jobs" in body
    assert body["jobs_base_dir_exists"] is True
    assert "disk_usage_mb" in body
    assert "disk_limit_mb" in body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert "/api/score" in paths
    assert "/api/score_batch" in paths


def test_manifest_service_specific(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    extras = body["service_specific"]
    assert "tool_outputs" in extras
    assert "score" in extras["tool_outputs"]
    assert "score_batch" in extras["tool_outputs"]
    assert "scoring_legend" in extras
    for metric in ("DockQ", "iRMSD", "LRMSD", "fnat", "clashes"):
        assert metric in extras["scoring_legend"]
    assert "input_uri_schemes" in extras
    assert "config_tips" in extras


def test_manifest_endpoint_examples(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert by_path["/api/score"]["examples"]
    assert by_path["/api/score_batch"]["examples"]


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    spec = r.json()
    assert "paths" in spec
    assert "/api/score" in spec["paths"]
    assert "/api/score_batch" in spec["paths"]


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404
    assert client.get("/api/jobs/missing-job-id/files").status_code == 404
    assert client.get("/api/jobs/missing-job-id/log").status_code == 404
    assert client.get("/api/jobs/missing-job-id/download").status_code == 404
    assert client.get("/api/jobs/missing-job-id/file/foo.json").status_code == 404


# =====================================================================
# 422 Error inputs (fast, no DockQ compute)
# =====================================================================

def test_422_score_missing_both_inputs(client: httpx.Client) -> None:
    """Neither upload nor URI for model/native → 422."""
    r = client.post("/api/score", data={"name": "fail"})
    assert r.status_code == 422


def test_422_score_missing_native(client: httpx.Client) -> None:
    """Model provided but no native → 422."""
    with open(MODEL_PDB, "rb") as fm:
        r = client.post(
            "/api/score",
            files={"model": (MODEL_PDB.name, fm, "chemical/x-pdb")},
            data={"name": "fail"},
        )
    assert r.status_code == 422


def test_422_score_missing_model(client: httpx.Client) -> None:
    """Native provided but no model → 422."""
    with open(NATIVE_PDB, "rb") as fn:
        r = client.post(
            "/api/score",
            files={"native": (NATIVE_PDB.name, fn, "chemical/x-pdb")},
            data={"name": "fail"},
        )
    assert r.status_code == 422


def test_422_batch_missing_models(client: httpx.Client) -> None:
    """Batch with native but no model files → 422."""
    with open(NATIVE_PDB, "rb") as fn:
        r = client.post(
            "/api/score_batch",
            files={"native": (NATIVE_PDB.name, fn, "chemical/x-pdb")},
        )
    assert r.status_code == 422
    assert "models" in r.json()["detail"].lower()


def test_422_score_invalid_name(client: httpx.Client) -> None:
    """Name with slashes should be rejected by pydantic validation."""
    with open(MODEL_PDB, "rb") as fm, open(NATIVE_PDB, "rb") as fn:
        r = client.post(
            "/api/score",
            files={
                "model": (MODEL_PDB.name, fm, "chemical/x-pdb"),
                "native": (NATIVE_PDB.name, fn, "chemical/x-pdb"),
            },
            data={"name": "bad/name"},
        )
    assert r.status_code == 422


# =====================================================================
# Inference: single score
# =====================================================================

def test_score_minimal_job(client: httpx.Client, base_url: str) -> None:
    submit = _submit_score(client, name="fc_smoke")
    _assert_submitted(submit)
    assert submit["input_params"]["name"] == "fc_smoke"

    final = _assert_completed(client, base_url, submit["job_id"])

    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert any(f.endswith("fc_smoke.json") for f in files), files


def test_score_output_content(client: httpx.Client, base_url: str) -> None:
    """Validate DockQ JSON output contains expected scoring fields."""
    submit = _submit_score(client, name="fc_content")
    _assert_submitted(submit)

    final = _assert_completed(client, base_url, submit["job_id"])

    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    json_files = [f for f in files if f.endswith(".json")]
    assert json_files, f"no JSON output: {files}"

    data = _download_json(client, final["job_id"], json_files[0])
    has_headline = any(k in data for k in ("GlobalDockQ", "total_DockQ", "DockQ"))
    assert has_headline, f"no headline DockQ key in output: {list(data.keys())}"
    assert "best_result" in data
    assert len(data["best_result"]) >= 1

    for _iface_key, iface_vals in data["best_result"].items():
        assert "DockQ" in iface_vals
        assert isinstance(iface_vals["DockQ"], (int, float))
        assert 0.0 <= iface_vals["DockQ"] <= 1.0
        assert "iRMSD" in iface_vals
        assert isinstance(iface_vals["iRMSD"], (int, float))


def test_score_with_no_align(client: httpx.Client, base_url: str) -> None:
    """Score with --no_align flag — should still produce valid output."""
    submit = _submit_score(client, name="fc_noalign", no_align="true")
    _assert_submitted(submit)
    assert submit["input_params"]["no_align"] is True

    final = _assert_completed(client, base_url, submit["job_id"])
    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert any(f.endswith("fc_noalign.json") for f in files), files


# =====================================================================
# Inference: batch score
# =====================================================================

def test_score_batch_minimal_job(client: httpx.Client, base_url: str) -> None:
    submit = _submit_batch(client, name="fc_batch")
    _assert_submitted(submit)
    assert submit["input_params"]["name"] == "fc_batch"
    assert submit["input_params"]["num_models"] == 2

    final = _assert_completed(client, base_url, submit["job_id"], timeout_s=BATCH_TIMEOUT_S)

    files = set(client.get(f"/api/jobs/{final['job_id']}/files").json()["files"])
    assert "scores.csv" in files, files
    assert any(p.startswith("per_model/") and p.endswith(".json") for p in files), files


def test_batch_scores_csv_content(client: httpx.Client, base_url: str) -> None:
    """Validate scores.csv has correct structure and values."""
    submit = _submit_batch(client, name="fc_csv")
    _assert_submitted(submit)

    final = _assert_completed(client, base_url, submit["job_id"], timeout_s=BATCH_TIMEOUT_S)

    csv_text = _download_text(client, final["job_id"], "scores.csv")
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)

    assert len(rows) == 2, f"expected 2 model rows, got {len(rows)}"
    assert "model" in reader.fieldnames
    assert "DockQ" in reader.fieldnames
    assert "iRMSD" in reader.fieldnames
    assert "n_interfaces" in reader.fieldnames

    for row in rows:
        assert row["model"], "model name should not be empty"
        dockq = float(row["DockQ"])
        assert 0.0 <= dockq <= 1.0, f"DockQ={dockq} out of range"

    dockq_values = [float(r["DockQ"]) for r in rows]
    assert dockq_values == sorted(dockq_values, reverse=True), \
        f"scores.csv should be sorted by DockQ descending: {dockq_values}"


def test_batch_per_model_jsons(client: httpx.Client, base_url: str) -> None:
    """Validate per-model JSON outputs exist and contain scoring data."""
    submit = _submit_batch(client, name="fc_per_model")
    _assert_submitted(submit)

    final = _assert_completed(client, base_url, submit["job_id"], timeout_s=BATCH_TIMEOUT_S)

    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    per_model_jsons = [f for f in files if f.startswith("per_model/") and f.endswith(".json")]
    assert len(per_model_jsons) == 2, f"expected 2 per-model JSONs, got {per_model_jsons}"

    for jpath in per_model_jsons:
        data = _download_json(client, final["job_id"], jpath)
        assert "best_result" in data, f"{jpath} missing best_result"
        has_headline = any(k in data for k in ("GlobalDockQ", "total_DockQ", "DockQ"))
        assert has_headline, f"{jpath} has no headline DockQ key: {list(data.keys())}"


# =====================================================================
# Job lifecycle endpoints
# =====================================================================

def test_job_status_polling(client: httpx.Client, base_url: str) -> None:
    """Job transitions from pending/running → completed; intermediate polls return valid JobInfo."""
    submit = _submit_score(client, name="fc_lifecycle")
    job_id = submit["job_id"]

    r = client.get(f"/api/jobs/{job_id}")
    r.raise_for_status()
    info = r.json()
    assert info["job_id"] == job_id
    assert info["status"] in ("pending", "running", "completed")

    _assert_completed(client, base_url, job_id)


def test_job_files_endpoint(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/files returns a list of output file paths."""
    submit = _submit_score(client, name="fc_files")
    _assert_completed(client, base_url, submit["job_id"])

    r = client.get(f"/api/jobs/{submit['job_id']}/files")
    r.raise_for_status()
    body = r.json()
    assert body["job_id"] == submit["job_id"]
    assert isinstance(body["files"], list)
    assert len(body["files"]) > 0
    assert any(f.endswith(".json") for f in body["files"])


def test_job_single_file_download(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/file/{path} returns the file content."""
    submit = _submit_score(client, name="fc_dl")
    _assert_completed(client, base_url, submit["job_id"])

    files = client.get(f"/api/jobs/{submit['job_id']}/files").json()["files"]
    json_file = next(f for f in files if f.endswith(".json"))

    r = client.get(f"/api/jobs/{submit['job_id']}/file/{json_file}")
    r.raise_for_status()
    data = r.json()
    assert isinstance(data, dict)
    assert "best_result" in data


def test_job_log_endpoint(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/log returns log text (may be empty for short jobs)."""
    submit = _submit_score(client, name="fc_log")
    _assert_completed(client, base_url, submit["job_id"])

    r = client.get(f"/api/jobs/{submit['job_id']}/log")
    r.raise_for_status()
    body = r.json()
    assert body["job_id"] == submit["job_id"]
    assert "log" in body
    assert isinstance(body["log"], str)


def test_job_download_zip(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/download returns a valid zip archive."""
    submit = _submit_score(client, name="fc_zip")
    _assert_completed(client, base_url, submit["job_id"])

    r = client.get(f"/api/jobs/{submit['job_id']}/download")
    r.raise_for_status()
    assert "application/zip" in r.headers.get("content-type", "")

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert len(names) > 0
    assert any(n.endswith(".json") for n in names)


def test_job_file_not_found(client: httpx.Client, base_url: str) -> None:
    """Requesting a non-existent file within a valid job → 404."""
    submit = _submit_score(client, name="fc_fnf")
    _assert_completed(client, base_url, submit["job_id"])

    r = client.get(f"/api/jobs/{submit['job_id']}/file/nonexistent.xyz")
    assert r.status_code == 404


def test_job_delete(client: httpx.Client, base_url: str) -> None:
    """DELETE /api/jobs/{id} removes the job; subsequent GET returns 404."""
    submit = _submit_score(client, name="fc_delete")
    _assert_completed(client, base_url, submit["job_id"])

    r = client.delete(f"/api/jobs/{submit['job_id']}")
    r.raise_for_status()
    assert r.json()["status"] == "deleted"

    assert client.get(f"/api/jobs/{submit['job_id']}").status_code == 404

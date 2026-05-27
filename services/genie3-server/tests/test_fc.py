"""End-to-end tests against the deployed genie3 Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/genie3-server/tests/test_fc.py

Motif and binder endpoints need a `dataset` zip with `problems/<name>.json` plus
referenced `motifs/` or `targets/` files. We build minimal one-problem zips on
the fly from fixtures in `tests/data/` (copied from upstream genie3 so the
suite is self-contained — no dependency on `opensource/genie3/`).

Each generation call sets `n_sample=1` / smallest viable length to keep the FC
GPU job under ~5 min.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest
import yaml

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
MOTIFBENCH = DATA_DIR / "motifbench"
BINDERTEST = DATA_DIR / "binder"

pytestmark = pytest.mark.fc

INFERENCE_TIMEOUT_S = 1800


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("genie3-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
        yield c


def _build_zip(files: dict[str, Path]) -> bytes:
    """Build an in-memory zip mapping archive paths → on-disk files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, src in files.items():
            zf.write(src, arcname=arcname)
    return buf.getvalue()


# =====================================================================
# Helpers
# =====================================================================


def _assert_submitted(resp_json: dict) -> None:
    """Validate the immediate POST response has expected fields."""
    assert "job_id" in resp_json
    assert resp_json["status"] in ("pending", "running")
    assert resp_json["created_at"] is not None
    assert resp_json["input_params"] is not None
    assert isinstance(resp_json["input_params"], dict)


def _assert_completed(job: dict, base_url: str, client: httpx.Client) -> list[str]:
    """Assert job completed successfully; return the output file list."""
    assert job["status"] == "completed", (
        f"failed: kind={job.get('failure_kind')} summary={job.get('error_summary')!r}"
    )

    assert job["created_at"] is not None
    assert job["started_at"] is not None
    assert job["completed_at"] is not None
    assert job["duration_seconds"] is not None
    assert job["duration_seconds"] > 0
    assert job["input_params"] is not None
    assert isinstance(job["input_params"], dict)
    assert job["output_count"] is not None
    assert job["output_count"] > 0
    assert job["output_total_bytes"] is not None
    assert job["output_total_bytes"] > 0

    files = client.get(f"/api/jobs/{job['job_id']}/files").json()["files"]
    assert files, "no output files"
    return files


def _submit_and_poll(
    client: httpx.Client,
    base_url: str,
    endpoint: str,
    *,
    data: dict | None = None,
    files: dict | list | None = None,
    timeout_s: int = INFERENCE_TIMEOUT_S,
) -> tuple[str, dict, list[str]]:
    """Submit a job, poll to completion, return (job_id, final_status, files)."""
    r = client.post(endpoint, data=data or {}, files=files or {})
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    job_id = submit["job_id"]

    final = poll_job(client, base_url, job_id, timeout_s=timeout_s)
    output_files = _assert_completed(final, base_url, client)
    return job_id, final, output_files


def _download_bytes(client: httpx.Client, job_id: str, file_path: str) -> bytes:
    r = client.get(f"/api/jobs/{job_id}/file/{file_path}")
    r.raise_for_status()
    return r.content


def _motif_zip() -> bytes:
    return _build_zip(
        {
            "problems/01_1LDB.json": MOTIFBENCH / "problems" / "01_1LDB.json",
            "motifs/01_1LDB.pdb": MOTIFBENCH / "motifs" / "01_1LDB.pdb",
        }
    )


def _binder_zip() -> bytes:
    return _build_zip(
        {
            "problems/01_bhrf1.json": BINDERTEST / "problems" / "01_bhrf1.json",
            "targets/pdb/01_bhrf1.pdb": BINDERTEST / "targets" / "pdb" / "01_bhrf1.pdb",
            "targets/pdb/01_bhrf1-chain_B.pdb": BINDERTEST / "targets" / "pdb" / "01_bhrf1-chain_B.pdb",
            "targets/fasta/01_bhrf1.fasta": BINDERTEST / "targets" / "fasta" / "01_bhrf1.fasta",
            "targets/fasta/01_bhrf1-chain_B.fasta": BINDERTEST / "targets" / "fasta" / "01_bhrf1-chain_B.fasta",
            "targets/msa/01_bhrf1.a3m": BINDERTEST / "targets" / "msa" / "01_bhrf1.a3m",
            "targets/msa/01_bhrf1-chain_B.a3m": BINDERTEST / "targets" / "msa" / "01_bhrf1-chain_B.a3m",
        }
    )


# =====================================================================
# Smoke (no GPU work)
# =====================================================================


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "genie3"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "genie3"
    assert "version" in body
    assert body["jobs_base_dir_exists"] is True
    assert "active_jobs" in body
    assert isinstance(body["active_jobs"], int)
    assert "max_concurrent_jobs" in body
    assert "disk_usage_mb" in body
    assert "disk_limit_mb" in body


def test_manifest_lists_four_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {
        "/api/generate/unconditional",
        "/api/generate/motif",
        "/api/generate/binder",
        "/api/generate",
    }


def test_manifest_service_specific(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    extras = body["service_specific"]
    assert "tool_outputs" in extras
    assert "*.pdb" in extras["tool_outputs"]["all_modes"]
    assert "endpoints_summary" in extras
    assert "config_tips" in extras
    assert "cond_strategy" in extras["config_tips"]
    assert "direction_scale" in extras["config_tips"]
    assert "input_uri_schemes" in extras


def test_manifest_endpoint_examples(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    for path in ("/api/generate/unconditional", "/api/generate/motif",
                 "/api/generate/binder", "/api/generate"):
        assert by_path[path]["examples"], f"no examples for {path}"


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    spec = r.json()
    assert "paths" in spec
    for path in ("/api/generate/unconditional", "/api/generate/motif",
                 "/api/generate/binder", "/api/generate"):
        assert path in spec["paths"], f"{path} missing from OpenAPI spec"


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404
    assert client.get("/api/jobs/missing-job-id/files").status_code == 404
    assert client.get("/api/jobs/missing-job-id/log").status_code == 404
    assert client.get("/api/jobs/missing-job-id/download").status_code == 404
    assert client.get("/api/jobs/missing-job-id/file/foo.pdb").status_code == 404


# =====================================================================
# 422 Error inputs (fast, no GPU)
# =====================================================================


def test_422_motif_bad_zip(client: httpx.Client) -> None:
    """Corrupt zip → 422."""
    r = client.post(
        "/api/generate/motif",
        files={"dataset": ("junk.zip", b"not a zip", "application/zip")},
    )
    assert r.status_code == 422


def test_422_motif_zip_without_problems(client: httpx.Client) -> None:
    """Valid zip but missing problems/ directory → 422."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("random/file.txt", "x")
    r = client.post(
        "/api/generate/motif",
        files={"dataset": ("noproblems.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 422
    assert "problems/" in r.json()["detail"].lower()


def test_422_binder_bad_zip(client: httpx.Client) -> None:
    """Corrupt zip for binder → 422."""
    r = client.post(
        "/api/generate/binder",
        files={"dataset": ("bad.zip", b"corrupt", "application/zip")},
    )
    assert r.status_code == 422


def test_422_custom_invalid_yaml(client: httpx.Client) -> None:
    """Non-parseable YAML → 422."""
    r = client.post(
        "/api/generate",
        data={"config_yaml": "{ invalid yaml: ["},
    )
    assert r.status_code == 422


def test_422_custom_yaml_not_a_dict(client: httpx.Client) -> None:
    """YAML that parses to a list, not a mapping → 422."""
    r = client.post(
        "/api/generate",
        data={"config_yaml": "- item1\n- item2\n"},
    )
    assert r.status_code == 422


# =====================================================================
# Inference: unconditional
# =====================================================================


def test_unconditional_minimal_job(client: httpx.Client, base_url: str) -> None:
    """No dataset upload — 1 sample, 50-residue monomer."""
    r = client.post(
        "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["n_sample"] == 1
    assert submit["input_params"]["min_length"] == 50

    final = poll_job(client, base_url, submit["job_id"], timeout_s=INFERENCE_TIMEOUT_S)
    _assert_completed(final, base_url, client)


def test_unconditional_output_has_pdb(client: httpx.Client, base_url: str) -> None:
    """Verify unconditional generation produces at least one PDB file with ATOM records."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
        },
    )

    pdb_files = [f for f in files if f.endswith(".pdb")]
    assert pdb_files, f"no PDB files in output: {files}"

    content = _download_bytes(client, job_id, pdb_files[0])
    text = content.decode("utf-8", errors="replace")
    assert "ATOM" in text, "PDB file does not contain ATOM records"


def test_unconditional_with_direction_scale(client: httpx.Client, base_url: str) -> None:
    """Explicit direction_scale=0.0 (recommended for longer proteins)."""
    r = client.post(
        "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
            "direction_scale": "0.0",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["direction_scale"] == 0.0

    final = poll_job(client, base_url, submit["job_id"], timeout_s=INFERENCE_TIMEOUT_S)
    _assert_completed(final, base_url, client)


# =====================================================================
# Inference: motif scaffolding
# =====================================================================


def test_motif_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Single-problem motif zip built on the fly from motifbench/01_1LDB."""
    r = client.post(
        "/api/generate/motif",
        files={"dataset": ("motif.zip", _motif_zip(), "application/zip")},
        data={
            "n_sample": "1",
            "batch_size": "1",
            "selections": "01_1LDB",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["selections"] == "01_1LDB"

    final = poll_job(client, base_url, submit["job_id"], timeout_s=INFERENCE_TIMEOUT_S)
    _assert_completed(final, base_url, client)


def test_motif_output_has_pdb(client: httpx.Client, base_url: str) -> None:
    """Motif scaffolding produces PDB files with ATOM records."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/generate/motif",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "selections": "01_1LDB",
        },
        files={"dataset": ("motif.zip", _motif_zip(), "application/zip")},
    )

    pdb_files = [f for f in files if f.endswith(".pdb")]
    assert pdb_files, f"no PDB files in output: {files}"

    content = _download_bytes(client, job_id, pdb_files[0])
    text = content.decode("utf-8", errors="replace")
    assert "ATOM" in text


# =====================================================================
# Inference: binder design
# =====================================================================


def test_binder_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Single-problem binder zip built from genie3/test/binder/01_bhrf1."""
    r = client.post(
        "/api/generate/binder",
        files={"dataset": ("binder.zip", _binder_zip(), "application/zip")},
        data={
            "n_sample": "1",
            "batch_size": "1",
            "selections": "01_bhrf1",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["selections"] == "01_bhrf1"

    final = poll_job(client, base_url, submit["job_id"], timeout_s=INFERENCE_TIMEOUT_S)
    _assert_completed(final, base_url, client)


def test_binder_output_has_pdb(client: httpx.Client, base_url: str) -> None:
    """Binder design produces PDB files with ATOM records."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/generate/binder",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "selections": "01_bhrf1",
        },
        files={"dataset": ("binder.zip", _binder_zip(), "application/zip")},
    )

    pdb_files = [f for f in files if f.endswith(".pdb")]
    assert pdb_files, f"no PDB files in output: {files}"

    content = _download_bytes(client, job_id, pdb_files[0])
    text = content.decode("utf-8", errors="replace")
    assert "ATOM" in text


# =====================================================================
# Inference: custom YAML
# =====================================================================


def test_custom_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Freeform `/api/generate` with a tiny unconditional YAML (no dataset)."""
    config = {
        "experiment": {"name": "fc_smoke_custom"},
        "paths": {"rootdir": "PLACEHOLDER_OVERRIDDEN_BY_SERVER"},
        "generation": {
            "dataset": {
                "source": "unconditional",
                "min_length": 50,
                "max_length": 50,
                "length_step": 50,
                "n_sample": 1,
            },
            "sampler": {"sampler": {"direction_scale": 0.8}},
        },
        "evaluation": {"version": "unconditional", "folding": {"model_name": "esmfold"}},
    }
    r = client.post(
        "/api/generate",
        data={"config_yaml": yaml.safe_dump(config)},
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)
    assert submit["input_params"]["config_yaml"] == "(user-supplied)"

    final = poll_job(client, base_url, submit["job_id"], timeout_s=INFERENCE_TIMEOUT_S)
    _assert_completed(final, base_url, client)


def test_custom_with_dataset_zip(client: httpx.Client, base_url: str) -> None:
    """Custom YAML + dataset zip — server rewrites paths.rootdir and paths.dataset."""
    config = {
        "experiment": {"name": "fc_custom_motif"},
        "paths": {"rootdir": "PLACEHOLDER", "dataset": "PLACEHOLDER"},
        "generation": {
            "dataset": {
                "source": "motif",
                "selections": "01_1LDB",
                "n_sample": 1,
                "batch_size": 1,
            },
            "sampler": {"sampler": {"direction_scale": 0.1}},
        },
    }
    r = client.post(
        "/api/generate",
        data={"config_yaml": yaml.safe_dump(config)},
        files={"dataset": ("motif.zip", _motif_zip(), "application/zip")},
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)

    final = poll_job(client, base_url, submit["job_id"], timeout_s=INFERENCE_TIMEOUT_S)
    _assert_completed(final, base_url, client)


# =====================================================================
# Job lifecycle endpoints
# =====================================================================


def test_job_status_polling(client: httpx.Client, base_url: str) -> None:
    """Job transitions from pending/running → completed; intermediate polls return valid JobInfo."""
    r = client.post(
        "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
        },
    )
    r.raise_for_status()
    job_id = r.json()["job_id"]

    info = client.get(f"/api/jobs/{job_id}").json()
    assert info["job_id"] == job_id
    assert info["status"] in ("pending", "running", "completed")

    final = poll_job(client, base_url, job_id, timeout_s=INFERENCE_TIMEOUT_S)
    _assert_completed(final, base_url, client)


def test_job_files_endpoint(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/files returns output file paths after completion."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
        },
    )

    r = client.get(f"/api/jobs/{job_id}/files")
    r.raise_for_status()
    body = r.json()
    assert body["job_id"] == job_id
    assert isinstance(body["files"], list)
    assert len(body["files"]) > 0
    assert any(f.endswith(".pdb") for f in body["files"])


def test_job_single_file_download(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/file/{path} returns the file content."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
        },
    )

    pdb_file = next(f for f in files if f.endswith(".pdb"))
    r = client.get(f"/api/jobs/{job_id}/file/{pdb_file}")
    r.raise_for_status()
    assert len(r.content) > 0
    assert b"ATOM" in r.content


def test_job_log_endpoint(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/log returns log text."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
        },
    )

    r = client.get(f"/api/jobs/{job_id}/log")
    r.raise_for_status()
    body = r.json()
    assert body["job_id"] == job_id
    assert "log" in body
    assert isinstance(body["log"], str)


def test_job_download_zip(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/download returns a valid zip archive with PDB outputs."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
        },
    )

    r = client.get(f"/api/jobs/{job_id}/download")
    r.raise_for_status()
    assert "application/zip" in r.headers.get("content-type", "")

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert len(names) > 0
    assert any(n.endswith(".pdb") for n in names)


def test_job_file_not_found(client: httpx.Client, base_url: str) -> None:
    """Requesting a non-existent file within a valid completed job → 404."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
        },
    )

    r = client.get(f"/api/jobs/{job_id}/file/nonexistent.xyz")
    assert r.status_code == 404


def test_job_delete(client: httpx.Client, base_url: str) -> None:
    """DELETE /api/jobs/{id} removes the job; subsequent GET returns 404."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
        },
    )

    r = client.delete(f"/api/jobs/{job_id}")
    r.raise_for_status()
    assert r.json()["status"] == "deleted"

    assert client.get(f"/api/jobs/{job_id}").status_code == 404

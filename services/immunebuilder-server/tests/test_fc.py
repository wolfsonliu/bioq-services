"""End-to-end tests against the deployed ImmuneBuilder Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/immunebuilder-server/tests/test_fc.py

URL resolves via `services/services.yaml`.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, make_retrying_client, poll_job

pytestmark = pytest.mark.fc

INFERENCE_TIMEOUT_S = 600

# =====================================================================
# Example sequences (real Ig domain sequences for functional testing)
# =====================================================================

HEAVY_SEQ = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYT"
    "RYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
LIGHT_SEQ = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPS"
    "RFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)
NANOBODY_SEQ = (
    "QVQLVESGGGLVQAGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAINSGGGSTYY"
    "PDSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAKDGYGGSFDYWGQGTQVTVSS"
)
ALPHA_SEQ = (
    "METLLGVSLVILWLQLARVNSQQGEEDPQALSIQEGENATMNCSYKTSINNLQWYRQNSGR"
    "GLVHLILIRSNEREKHSGRLRVTLDTSKKSSSLLITASRAADTASYFCAVNFGGGKLIFGQGTELSVKP"
)
BETA_SEQ = (
    "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVP"
    "NGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSFSTCSANYGYTFGSGTRLTVVEDL"
)


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("immunebuilder-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    # Wrap in Retry429Transport so smoke assertions survive short bursts of
    # account-level GPU quota exhaustion (FC returns 429 at the platform
    # layer when 'fc.gpu.tesla.1' is saturated by any function in the
    # account, not just immunebuilder).
    with make_retrying_client(base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# =====================================================================
# Helpers
# =====================================================================


def _assert_submitted(body: dict) -> None:
    """Validate the immediate POST response has expected fields."""
    assert "job_id" in body
    assert body["status"] in ("pending", "running")
    assert body["created_at"] is not None
    assert body["input_params"] is not None
    assert isinstance(body["input_params"], dict)


def _assert_completed(final: dict, client: httpx.Client) -> list[str]:
    """Assert job completed successfully; return the output file list."""
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

    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert files, "no output files"
    return files


def _submit_and_poll(
    client: httpx.Client,
    base_url: str,
    endpoint: str,
    *,
    data: dict,
    timeout_s: int = INFERENCE_TIMEOUT_S,
) -> tuple[str, dict, list[str]]:
    """Submit a job, poll to completion, return (job_id, final_status, files)."""
    r = client.post(endpoint, data=data)
    r.raise_for_status()
    body = r.json()
    _assert_submitted(body)
    job_id = body["job_id"]

    final = poll_job(client, base_url, job_id, timeout_s=timeout_s, interval_s=10)
    output_files = _assert_completed(final, client)
    return job_id, final, output_files


def _download_bytes(client: httpx.Client, job_id: str, file_path: str) -> bytes:
    r = client.get(f"/api/jobs/{job_id}/file/{file_path}")
    r.raise_for_status()
    return r.content


def _assert_pdb_valid(content: bytes) -> None:
    """Minimal PDB content validation."""
    text = content.decode("utf-8", errors="replace")
    assert "ATOM" in text, "PDB file does not contain ATOM records"


# =====================================================================
# Smoke (no GPU work)
# =====================================================================


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "immunebuilder"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    """immunebuilder overrides framework's /healthz/detail with a
    weights-focused probe (see app.py:healthz_detail).  The response
    schema is service-specific, not the framework default."""
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "immunebuilder"
    assert "version" in body
    assert "weights_dir" in body
    assert body["weights_loaded"] is True, (
        f"NAS weights probe failed: {body}"
    )
    assert body["files_found"] >= 12, (
        f"expected >=12 weight files, got {body['files_found']}"
    )
    assert isinstance(body["active_jobs"], int)
    assert isinstance(body["max_concurrent_jobs"], int)


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    """Manifest exposes both sync (/api/predict_*) and FC async task
    (/api/tasks/predict_*) variants for each predictor."""
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {
        "/api/predict_antibody",
        "/api/predict_nanobody",
        "/api/predict_tcr",
        "/api/tasks/predict_antibody",
        "/api/tasks/predict_nanobody",
        "/api/tasks/predict_tcr",
    }


def test_manifest_predictors(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    predictors = body["service_specific"]["predictors"]
    assert set(predictors.keys()) == {"antibody", "nanobody", "tcr"}


def test_manifest_tool_outputs(client: httpx.Client) -> None:
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "final_model" in extras["tool_outputs"]
    assert "unrefined_models" in extras["tool_outputs"]
    assert "error_estimates" in extras["tool_outputs"]


def test_manifest_numbering_schemes(client: httpx.Client) -> None:
    extras = client.get("/api/manifest").json()["service_specific"]
    schemes = extras["numbering_schemes"]
    assert "imgt" in schemes
    assert "chothia" in schemes
    assert "kabat" in schemes


def test_manifest_endpoint_examples(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    for path in ("/api/predict_antibody", "/api/predict_nanobody", "/api/predict_tcr"):
        assert by_path[path]["examples"], f"no examples for {path}"


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    spec = r.json()
    assert "paths" in spec
    for path in (
        "/api/predict_antibody",
        "/api/predict_nanobody",
        "/api/predict_tcr",
        "/api/tasks/predict_antibody",
        "/api/tasks/predict_nanobody",
        "/api/tasks/predict_tcr",
    ):
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


def test_422_antibody_bad_aa(client: httpx.Client) -> None:
    """Non-standard amino acid letter → 422."""
    r = client.post(
        "/api/predict_antibody",
        data={
            "heavy_sequence": "X" * 50,
            "light_sequence": LIGHT_SEQ,
        },
    )
    assert r.status_code == 422


def test_422_antibody_short_seq(client: httpx.Client) -> None:
    """Sequence too short → 422."""
    r = client.post(
        "/api/predict_antibody",
        data={
            "heavy_sequence": "EVQL",
            "light_sequence": LIGHT_SEQ,
        },
    )
    assert r.status_code == 422


def test_422_nanobody_missing_seq(client: httpx.Client) -> None:
    """Missing required field → 422."""
    r = client.post("/api/predict_nanobody", data={"name": "fail"})
    assert r.status_code == 422


def test_422_tcr_missing_beta(client: httpx.Client) -> None:
    """TCR with only alpha → 422."""
    r = client.post(
        "/api/predict_tcr",
        data={"alpha_sequence": ALPHA_SEQ},
    )
    assert r.status_code == 422


def test_422_tcr_missing_alpha(client: httpx.Client) -> None:
    """TCR with only beta → 422."""
    r = client.post(
        "/api/predict_tcr",
        data={"beta_sequence": BETA_SEQ},
    )
    assert r.status_code == 422


def test_422_antibody_invalid_numbering(client: httpx.Client) -> None:
    """Invalid numbering scheme → 422."""
    r = client.post(
        "/api/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "numbering_scheme": "invalid_scheme",
        },
    )
    assert r.status_code == 422


# =====================================================================
# Inference: predict_antibody
# =====================================================================


def test_predict_antibody_job(client: httpx.Client, base_url: str) -> None:
    """Full antibody structure prediction with default settings."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "name": "fc_ab",
        },
    )

    assert "final_model.pdb" in files
    pdb_content = _download_bytes(client, job_id, "final_model.pdb")
    _assert_pdb_valid(pdb_content)


def test_predict_antibody_save_all(client: httpx.Client, base_url: str) -> None:
    """Antibody with save_all_models=True produces ensemble outputs."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "name": "fc_ab_all",
            "save_all_models": "true",
        },
    )

    assert "final_model.pdb" in files
    assert "rank0_unrefined.pdb" in files
    assert "error_estimates.npy" in files

    pdb_content = _download_bytes(client, job_id, "final_model.pdb")
    _assert_pdb_valid(pdb_content)

    unrefined_content = _download_bytes(client, job_id, "rank0_unrefined.pdb")
    _assert_pdb_valid(unrefined_content)


def test_predict_antibody_save_final_only(client: httpx.Client, base_url: str) -> None:
    """Antibody with save_all_models=False produces only final_model.pdb."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "name": "fc_ab_final",
            "save_all_models": "false",
        },
    )

    assert "final_model.pdb" in files
    assert "rank0_unrefined.pdb" not in files
    assert "error_estimates.npy" not in files


def test_predict_antibody_numbering_chothia(client: httpx.Client, base_url: str) -> None:
    """Antibody with Chothia numbering scheme."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "name": "fc_ab_chothia",
            "numbering_scheme": "chothia",
        },
    )

    assert "final_model.pdb" in files
    assert final["input_params"]["numbering_scheme"] == "chothia"


def test_predict_antibody_no_sidechain_check(client: httpx.Client, base_url: str) -> None:
    """Antibody with sidechain bond check disabled (-u flag)."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "name": "fc_ab_nosc",
            "no_sidechain_bond_check": "true",
        },
    )

    assert "final_model.pdb" in files
    assert final["input_params"]["no_sidechain_bond_check"] is True


# =====================================================================
# Inference: predict_nanobody
# =====================================================================


def test_predict_nanobody_job(client: httpx.Client, base_url: str) -> None:
    """Full nanobody structure prediction."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_nb",
        },
    )

    assert "final_model.pdb" in files
    pdb_content = _download_bytes(client, job_id, "final_model.pdb")
    _assert_pdb_valid(pdb_content)


def test_predict_nanobody_save_all(client: httpx.Client, base_url: str) -> None:
    """Nanobody with save_all_models=True produces ensemble outputs."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_nb_all",
            "save_all_models": "true",
        },
    )

    assert "final_model.pdb" in files
    assert "rank0_unrefined.pdb" in files
    assert "error_estimates.npy" in files


# =====================================================================
# Inference: predict_tcr
# =====================================================================


def test_predict_tcr_job(client: httpx.Client, base_url: str) -> None:
    """Full TCR structure prediction."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_tcr",
        data={
            "alpha_sequence": ALPHA_SEQ,
            "beta_sequence": BETA_SEQ,
            "name": "fc_tcr",
        },
    )

    assert "final_model.pdb" in files
    pdb_content = _download_bytes(client, job_id, "final_model.pdb")
    _assert_pdb_valid(pdb_content)


def test_predict_tcr_save_all(client: httpx.Client, base_url: str) -> None:
    """TCR with save_all_models=True produces ensemble outputs."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_tcr",
        data={
            "alpha_sequence": ALPHA_SEQ,
            "beta_sequence": BETA_SEQ,
            "name": "fc_tcr_all",
            "save_all_models": "true",
        },
    )

    assert "final_model.pdb" in files
    assert "rank0_unrefined.pdb" in files
    assert "error_estimates.npy" in files


# =====================================================================
# input_params echo
# =====================================================================


def test_input_params_antibody(client: httpx.Client, base_url: str) -> None:
    """Verify input_params are stored and returned correctly for antibody."""
    r = client.post(
        "/api/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "name": "fc_params_ab",
            "numbering_scheme": "chothia",
            "save_all_models": "true",
            "no_sidechain_bond_check": "false",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)

    assert submit["input_params"]["name"] == "fc_params_ab"
    assert submit["input_params"]["numbering_scheme"] == "chothia"
    assert submit["input_params"]["heavy_sequence"] == HEAVY_SEQ
    assert submit["input_params"]["light_sequence"] == LIGHT_SEQ

    final = poll_job(client, base_url, submit["job_id"], timeout_s=INFERENCE_TIMEOUT_S, interval_s=10)
    _assert_completed(final, client)
    assert final["input_params"]["name"] == "fc_params_ab"
    assert final["input_params"]["numbering_scheme"] == "chothia"


def test_input_params_nanobody(client: httpx.Client, base_url: str) -> None:
    """Verify input_params round-trip for nanobody."""
    r = client.post(
        "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_params_nb",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)

    assert submit["input_params"]["heavy_sequence"] == NANOBODY_SEQ
    assert submit["input_params"]["name"] == "fc_params_nb"

    final = poll_job(client, base_url, submit["job_id"], timeout_s=INFERENCE_TIMEOUT_S, interval_s=10)
    _assert_completed(final, client)
    assert final["input_params"]["heavy_sequence"] == NANOBODY_SEQ


def test_input_params_tcr(client: httpx.Client, base_url: str) -> None:
    """Verify input_params round-trip for TCR."""
    r = client.post(
        "/api/predict_tcr",
        data={
            "alpha_sequence": ALPHA_SEQ,
            "beta_sequence": BETA_SEQ,
            "name": "fc_params_tcr",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)

    assert submit["input_params"]["alpha_sequence"] == ALPHA_SEQ
    assert submit["input_params"]["beta_sequence"] == BETA_SEQ
    assert submit["input_params"]["name"] == "fc_params_tcr"

    final = poll_job(client, base_url, submit["job_id"], timeout_s=INFERENCE_TIMEOUT_S, interval_s=10)
    _assert_completed(final, client)
    assert final["input_params"]["alpha_sequence"] == ALPHA_SEQ
    assert final["input_params"]["beta_sequence"] == BETA_SEQ


# =====================================================================
# Job lifecycle
# =====================================================================


def test_job_status_polling(client: httpx.Client, base_url: str) -> None:
    """Job transitions from pending/running → completed; intermediate polls return valid JobInfo."""
    r = client.post(
        "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_status",
        },
    )
    r.raise_for_status()
    job_id = r.json()["job_id"]

    info = client.get(f"/api/jobs/{job_id}").json()
    assert info["job_id"] == job_id
    assert info["status"] in ("pending", "running", "completed")

    final = poll_job(client, base_url, job_id, timeout_s=INFERENCE_TIMEOUT_S, interval_s=10)
    _assert_completed(final, client)


def test_job_files_endpoint(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/files returns output file paths after completion."""
    job_id, _, files = _submit_and_poll(
        client, base_url, "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_files",
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
    job_id, _, files = _submit_and_poll(
        client, base_url, "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_dl_single",
        },
    )

    pdb_file = next(f for f in files if f.endswith(".pdb"))
    content = _download_bytes(client, job_id, pdb_file)
    assert len(content) > 0
    _assert_pdb_valid(content)


def test_job_log_endpoint(client: httpx.Client, base_url: str) -> None:
    """GET /api/jobs/{id}/log returns log text."""
    job_id, _, _ = _submit_and_poll(
        client, base_url, "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_log",
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
    job_id, _, _ = _submit_and_poll(
        client, base_url, "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_zip",
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
    job_id, _, _ = _submit_and_poll(
        client, base_url, "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_fnf",
        },
    )

    r = client.get(f"/api/jobs/{job_id}/file/nonexistent.xyz")
    assert r.status_code == 404


def test_job_delete(client: httpx.Client, base_url: str) -> None:
    """DELETE /api/jobs/{id} removes the job; subsequent GET returns 404."""
    job_id, _, _ = _submit_and_poll(
        client, base_url, "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_del",
        },
    )

    r = client.delete(f"/api/jobs/{job_id}")
    r.raise_for_status()
    assert r.json()["status"] == "deleted"

    assert client.get(f"/api/jobs/{job_id}").status_code == 404

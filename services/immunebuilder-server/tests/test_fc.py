"""End-to-end tests against the deployed ImmuneBuilder Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/immunebuilder-server/tests/test_fc.py

URL resolves via `services/aliyun_fc_url.md`.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

pytestmark = pytest.mark.fc

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
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# =====================================================================
# Helpers
# =====================================================================

def _submit_and_poll(
    client: httpx.Client,
    base_url: str,
    endpoint: str,
    *,
    data: dict,
    timeout_s: int = 600,
) -> tuple[str, dict, list[str]]:
    """Submit a job, poll to completion, return (job_id, final_status, files)."""
    r = client.post(endpoint, data=data)
    r.raise_for_status()
    body = r.json()
    assert "job_id" in body
    job_id = body["job_id"]

    final = poll_job(client, base_url, job_id, timeout_s=timeout_s, interval_s=10)
    assert final["status"] == "completed", (
        f"failed: kind={final.get('failure_kind')} summary={final.get('error_summary')!r}"
    )

    files_list = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    return job_id, final, files_list


def _download_bytes(client: httpx.Client, job_id: str, file_path: str) -> bytes:
    r = client.get(f"/api/jobs/{job_id}/file/{file_path}")
    r.raise_for_status()
    return r.content


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
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "immunebuilder"
    assert body["jobs_base_dir_exists"] is True


def test_manifest_lists_three_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/predict_antibody", "/api/predict_nanobody", "/api/predict_tcr"}


def test_manifest_predictors(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    predictors = body["service_specific"]["predictors"]
    assert set(predictors.keys()) == {"antibody", "nanobody", "tcr"}


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    assert "paths" in r.json()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


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


# =====================================================================
# Inference: predict_antibody
# =====================================================================

def test_predict_antibody_job(client: httpx.Client, base_url: str) -> None:
    """Full antibody structure prediction."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "name": "fc_ab",
        },
    )

    assert "final_model.pdb" in files
    pdb_content = _download_bytes(client, job_id, "final_model.pdb").decode()
    assert "ATOM" in pdb_content


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


def test_predict_antibody_output_details(client: httpx.Client, base_url: str) -> None:
    """Verify job metadata: duration, output_count, output_total_bytes."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "name": "fc_ab_meta",
        },
    )

    assert final["duration_seconds"] > 0
    assert final["output_count"] > 0
    assert final["output_total_bytes"] > 0


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
    pdb_content = _download_bytes(client, job_id, "final_model.pdb").decode()
    assert "ATOM" in pdb_content


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
    pdb_content = _download_bytes(client, job_id, "final_model.pdb").decode()
    assert "ATOM" in pdb_content


# =====================================================================
# input_params echo
# =====================================================================

def test_input_params_round_trip(client: httpx.Client, base_url: str) -> None:
    """Verify input_params are stored and returned correctly."""
    r = client.post(
        "/api/predict_antibody",
        data={
            "heavy_sequence": HEAVY_SEQ,
            "light_sequence": LIGHT_SEQ,
            "name": "fc_params_rt",
            "numbering_scheme": "chothia",
            "save_all_models": "true",
            "no_sidechain_bond_check": "false",
        },
    )
    r.raise_for_status()
    submit = r.json()

    assert submit["input_params"]["name"] == "fc_params_rt"
    assert submit["input_params"]["numbering_scheme"] == "chothia"
    assert submit["input_params"]["heavy_sequence"] == HEAVY_SEQ
    assert submit["input_params"]["light_sequence"] == LIGHT_SEQ

    final = poll_job(client, base_url, submit["job_id"], timeout_s=600, interval_s=10)
    assert final["status"] == "completed"
    assert final["input_params"]["name"] == "fc_params_rt"


# =====================================================================
# Job lifecycle
# =====================================================================

def test_job_files_endpoint(client: httpx.Client, base_url: str) -> None:
    """Verify /api/jobs/{id}/files returns file list."""
    job_id, _, files = _submit_and_poll(
        client, base_url, "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_files",
        },
    )
    assert isinstance(files, list)
    assert len(files) > 0


def test_job_download_zip(client: httpx.Client, base_url: str) -> None:
    """Verify /api/jobs/{id}/download returns a zip archive."""
    job_id, _, _ = _submit_and_poll(
        client, base_url, "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_zip",
        },
    )
    r = client.get(f"/api/jobs/{job_id}/download")
    r.raise_for_status()
    assert r.headers["content-type"] == "application/zip"
    assert len(r.content) > 0


def test_job_delete(client: httpx.Client, base_url: str) -> None:
    """Verify DELETE /api/jobs/{id} removes the job."""
    job_id, _, _ = _submit_and_poll(
        client, base_url, "/api/predict_nanobody",
        data={
            "heavy_sequence": NANOBODY_SEQ,
            "name": "fc_del",
        },
    )
    r = client.delete(f"/api/jobs/{job_id}")
    r.raise_for_status()
    assert client.get(f"/api/jobs/{job_id}").status_code == 404

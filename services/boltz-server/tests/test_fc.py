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
import yaml

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"

pytestmark = pytest.mark.fc

SHORT_PROTEIN = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSC"
MINIMAL_INFERENCE_PARAMS = {
    "diffusion_samples": "1",
    "recycling_steps": "1",
    "sampling_steps": "50",
}


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("boltz-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# =====================================================================
# Helpers
# =====================================================================

def _submit_and_poll(
    client: httpx.Client,
    base_url: str,
    endpoint: str,
    payload: dict,
    *,
    timeout_s: int = 1800,
) -> tuple[str, dict, list[str]]:
    """Submit a job, poll to completion, return (job_id, final_status, files)."""
    r = client.post(endpoint, data=payload)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    final = poll_job(client, base_url, job_id, timeout_s=timeout_s, interval_s=20)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    return job_id, final, files


def _download_json(client: httpx.Client, job_id: str, file_path: str) -> dict:
    """Download a single file from a job's output and parse as JSON."""
    r = client.get(f"/api/jobs/{job_id}/file/{file_path}")
    r.raise_for_status()
    return r.json()


# =====================================================================
# Smoke (no GPU work)
# =====================================================================

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


# =====================================================================
# 422 Error inputs (fast, no GPU)
# =====================================================================

def test_422_missing_sequences(client: httpx.Client) -> None:
    """No sequences and no raw_yaml → 422."""
    r = client.post("/api/predict_structure", data={"name": "fail", "msa_mode": "empty"})
    assert r.status_code == 422


def test_422_binder_points_to_protein(client: httpx.Client) -> None:
    """binder_id references a protein chain → 422."""
    payload = {
        "name": "fail",
        "binder_id": "A",
        "msa_mode": "empty",
        "sequences": json.dumps([
            {"type": "protein", "id": "A", "sequence": "MKT", "msa_uri": "empty"},
            {"type": "ligand", "id": "B", "smiles": "CCO"},
        ]),
    }
    r = client.post("/api/predict_affinity", data=payload)
    assert r.status_code == 422


def test_422_invalid_raw_yaml(client: httpx.Client) -> None:
    """raw_yaml missing required `sequences` key → 422."""
    payload = {"name": "fail", "raw_yaml": "version: 1\nfoo: bar\n"}
    r = client.post("/api/predict_structure", data=payload)
    assert r.status_code == 422


def test_422_raw_yaml_not_a_dict(client: httpx.Client) -> None:
    """raw_yaml is a YAML list, not a mapping → 422."""
    payload = {"name": "fail", "raw_yaml": "- item1\n- item2\n"}
    r = client.post("/api/predict_structure", data=payload)
    assert r.status_code == 422


def test_422_binder_id_not_found(client: httpx.Client) -> None:
    """binder_id references a non-existent chain → 422."""
    payload = {
        "name": "fail",
        "binder_id": "Z",
        "msa_mode": "empty",
        "sequences": json.dumps([
            {"type": "protein", "id": "A", "sequence": "MKT", "msa_uri": "empty"},
            {"type": "ligand", "id": "B", "smiles": "CCO"},
        ]),
    }
    r = client.post("/api/predict_affinity", data=payload)
    assert r.status_code == 422


# =====================================================================
# Inference: structure prediction
# =====================================================================

def test_predict_structure_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Smallest possible structure prediction — single short protein, no MSA, 1 sample."""
    payload = {
        "name": "fc_smoke",
        "msa_mode": "empty",
        **MINIMAL_INFERENCE_PARAMS,
        "sequences": json.dumps([
            {
                "type": "protein",
                "id": "A",
                "sequence": SHORT_PROTEIN,
                "msa_uri": "empty",
            }
        ]),
    }
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_structure", payload,
    )

    model_files = [f for f in files if f.endswith(".cif") or f.endswith(".pdb")]
    assert model_files, f"no model file in outputs: {files}"

    confidence_files = [f for f in files if "confidence" in f and f.endswith(".json")]
    assert confidence_files, f"no confidence JSON in outputs: {files}"

    conf = _download_json(client, job_id, confidence_files[0])
    assert "ptm" in conf or "confidence" in conf, f"unexpected confidence JSON keys: {list(conf.keys())}"


def test_predict_structure_protein_ligand_complex(
    client: httpx.Client, base_url: str,
) -> None:
    """Two-chain complex: protein + SMILES ligand, empty MSA."""
    payload = {
        "name": "fc_complex",
        "msa_mode": "empty",
        **MINIMAL_INFERENCE_PARAMS,
        "sequences": json.dumps([
            {
                "type": "protein",
                "id": "A",
                "sequence": SHORT_PROTEIN,
                "msa_uri": "empty",
            },
            {"type": "ligand", "id": "B", "smiles": "c1ccccc1"},
        ]),
    }
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_structure", payload,
    )

    assert any(
        f.endswith(".cif") or f.endswith(".pdb") for f in files
    ), f"no model file in outputs: {files}"


def test_predict_structure_raw_yaml(client: httpx.Client, base_url: str) -> None:
    """Submit raw YAML directly instead of structured sequences."""
    raw = yaml.safe_dump(
        {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": SHORT_PROTEIN,
                        "msa": "empty",
                    }
                },
            ],
        },
        sort_keys=False,
    )
    payload = {
        "name": "fc_raw",
        "raw_yaml": raw,
        **MINIMAL_INFERENCE_PARAMS,
    }
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_structure", payload,
    )

    assert any(
        f.endswith(".cif") or f.endswith(".pdb") for f in files
    ), f"no model file in outputs: {files}"


# =====================================================================
# Inference: affinity prediction
# =====================================================================

def test_predict_affinity_smiles_ligand(client: httpx.Client, base_url: str) -> None:
    """Affinity prediction with SMILES ligand — validate affinity JSON fields."""
    payload = {
        "name": "fc_aff_smiles",
        "binder_id": "B",
        "msa_mode": "empty",
        **MINIMAL_INFERENCE_PARAMS,
        "diffusion_samples_affinity": "1",
        "sampling_steps_affinity": "50",
        "sequences": json.dumps([
            {
                "type": "protein",
                "id": "A",
                "sequence": SHORT_PROTEIN,
                "msa_uri": "empty",
            },
            {"type": "ligand", "id": "B", "smiles": "c1ccccc1"},
        ]),
    }
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_affinity", payload,
    )

    affinity_files = [f for f in files if "affinity_" in f and f.endswith(".json")]
    assert affinity_files, f"no affinity json in outputs: {files}"

    aff = _download_json(client, job_id, affinity_files[0])
    assert "affinity_pred_value" in aff, f"missing affinity_pred_value: {list(aff.keys())}"
    assert isinstance(aff["affinity_pred_value"], (int, float))


def test_predict_affinity_ccd_ligand(client: httpx.Client, base_url: str) -> None:
    """Affinity prediction with CCD ligand (ATP) instead of SMILES."""
    payload = {
        "name": "fc_aff_ccd",
        "binder_id": "B",
        "msa_mode": "empty",
        **MINIMAL_INFERENCE_PARAMS,
        "diffusion_samples_affinity": "1",
        "sampling_steps_affinity": "50",
        "sequences": json.dumps([
            {
                "type": "protein",
                "id": "A",
                "sequence": SHORT_PROTEIN,
                "msa_uri": "empty",
            },
            {"type": "ligand", "id": "B", "ccd": "ATP"},
        ]),
    }
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/predict_affinity", payload,
    )

    affinity_files = [f for f in files if "affinity_" in f and f.endswith(".json")]
    assert affinity_files, f"no affinity json in outputs: {files}"

    aff = _download_json(client, job_id, affinity_files[0])
    assert "affinity_pred_value" in aff, f"missing affinity_pred_value: {list(aff.keys())}"
    assert isinstance(aff["affinity_pred_value"], (int, float))


# =====================================================================
# Cross-job reference: job:// URI
# =====================================================================

def test_job_uri_cross_reference(client: httpx.Client, base_url: str) -> None:
    """Run a structure prediction, then reference its output CIF as a template
    in a second prediction via the job:// URI scheme."""

    # --- Job A: produce a structure ---
    payload_a = {
        "name": "fc_ref_src",
        "msa_mode": "empty",
        **MINIMAL_INFERENCE_PARAMS,
        "output_format": "mmcif",
        "sequences": json.dumps([
            {
                "type": "protein",
                "id": "A",
                "sequence": SHORT_PROTEIN,
                "msa_uri": "empty",
            }
        ]),
    }
    job_id_a, _, files_a = _submit_and_poll(
        client, base_url, "/api/predict_structure", payload_a,
    )

    cif_files = [f for f in files_a if f.endswith(".cif")]
    assert cif_files, f"no CIF output from job A: {files_a}"
    cif_path = cif_files[0]

    # --- Job B: use job A's CIF as a template via job:// URI ---
    payload_b = {
        "name": "fc_ref_dst",
        "msa_mode": "empty",
        **MINIMAL_INFERENCE_PARAMS,
        "sequences": json.dumps([
            {
                "type": "protein",
                "id": "A",
                "sequence": SHORT_PROTEIN,
                "msa_uri": "empty",
            }
        ]),
        "templates": json.dumps([
            {"cif_uri": f"job://{job_id_a}/{cif_path}"},
        ]),
    }
    job_id_b, _, files_b = _submit_and_poll(
        client, base_url, "/api/predict_structure", payload_b,
    )

    assert any(
        f.endswith(".cif") or f.endswith(".pdb") for f in files_b
    ), f"no model file in job B outputs: {files_b}"

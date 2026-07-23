"""End-to-end tests against the deployed Boltz Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/boltz-server/tests/test_fc.py

The base URL is read from `services.yaml` — update that file after
deploying a new tag in the FC console. Inference tests use `msa_mode=empty`
so they don't depend on ColabFold's MSA server (which can be slow/flaky from
inside FC).

The "Task endpoints (async task mode)" section verifies FC async task mode
end-to-end: HTTP 202 on submit, X-Bioagent-Job-Id alignment, and idempotency.
These tests require the FC console to have async task mode enabled for the
function (see engineering/decisions/2026-06-17-fc-async-task-mode.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from bioq_service.fc_testing import fc_url, poll_job

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
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(300.0)) as c:
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


def _async_submit_and_poll(
    client: httpx.Client,
    base_url: str,
    endpoint: str,
    payload: dict,
    *,
    task_id: str,
    timeout_s: int = 1800,
) -> tuple[str, dict, list[str]]:
    """Submit via FC async task mode, poll JobInfo to completion.

    Sends with `X-Fc-Invocation-Type: Async` + `X-Bioagent-Job-Id=<task_id>`.
    Expects 202.  The server-side task endpoint blocks synchronously inside
    the FC instance; we poll `/api/jobs/{task_id}` because JobInfo records
    every state transition.

    Returns (task_id, final_status, files).  Asserts terminal status is
    `completed`.  Raises if FC returned non-202 (which would mean async task
    mode is not enabled or the endpoint is wrong).
    """
    r = client.post(
        endpoint,
        data=payload,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    assert r.status_code == 202, (
        f"expected 202 from async invocation; got {r.status_code} body={r.text!r}.  "
        f"Check that FC console has async task mode enabled for this function."
    )

    final = poll_job(client, base_url, task_id, timeout_s=timeout_s, interval_s=20)
    assert final["status"] == "completed", final

    files = client.get(f"/api/jobs/{task_id}/files").json()["files"]
    return task_id, final, files


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


# =====================================================================
# Task endpoints (async task mode) — synchronous-blocking endpoint
# invoked via FC Async Task Mode (X-Fc-Invocation-Type: Async).
# =====================================================================

def test_async_task_predict_structure_minimal(client: httpx.Client, base_url: str) -> None:
    """Async invoke /api/tasks/predict_structure, poll JobInfo to completion.

    Validates the full FC async task mode pipeline:
      - HTTP 202 on submit (proves async task mode is enabled in FC console)
      - task_id from X-Bioagent-Job-Id is used as the JobInfo.job_id
      - server runs the pipeline synchronously to completion inside the FC instance
      - JobInfo lifecycle (pending → running → completed) is persisted to NAS
    """
    import time
    task_id = f"fc-async-min-{int(time.time())}"
    payload = {
        "name": "fc_async_min",
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
    job_id, final, files = _async_submit_and_poll(
        client, base_url, "/api/tasks/predict_structure", payload,
        task_id=task_id,
    )

    assert job_id == task_id, "task endpoint must echo X-Bioagent-Job-Id as JobInfo.job_id"
    assert final["completed_at"] is not None
    assert final["started_at"] is not None
    assert final["duration_seconds"] is not None and final["duration_seconds"] > 0

    model_files = [f for f in files if f.endswith(".cif") or f.endswith(".pdb")]
    assert model_files, f"no model file in outputs: {files}"


def test_async_task_predict_structure_honors_bioagent_job_id(
    client: httpx.Client, base_url: str,
) -> None:
    """Verify X-Bioagent-Job-Id flows through to JobInfo.job_id end-to-end."""
    import time
    task_id = f"fc-async-id-{int(time.time())}"
    payload = {
        "name": "fc_async_id",
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
    r = client.post(
        "/api/tasks/predict_structure",
        data=payload,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    assert r.status_code == 202

    final = poll_job(client, base_url, task_id, timeout_s=1800, interval_s=20)
    assert final["status"] == "completed", final
    assert final["job_id"] == task_id, "JobInfo.job_id must equal X-Bioagent-Job-Id"


def test_async_task_duplicate_rejected_at_fc_platform_layer(
    client: httpx.Client, base_url: str,
) -> None:
    """Same X-Fc-Async-Task-Id twice → FC rejects the second at platform layer.

    Important discovery from the first end-to-end run (2026-06-19): FC's async
    task mode itself dedups by X-Fc-Async-Task-Id; the second invocation never
    reaches our function (returns HTTP 409 Conflict at the FC layer).  This is
    *better* than the framework-layer dedup we built into `execute_task` —
    duplicates don't even cost a cold-start.  Our framework dedup remains a
    defense-in-depth fallback for invocation paths that bypass FC (LocalDispatcher,
    direct curl, future K8s backend).

    Test verifies:
      1. First submit succeeds (HTTP 202) and runs to completion.
      2. Second submit with SAME task_id returns HTTP 409 (FC platform dedup).
      3. The original JobInfo is unchanged (created_at, input_params).
    """
    import time
    task_id = f"fc-async-dup-{int(time.time())}"
    payload_first = {
        "name": "fc_async_first",
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
    payload_second = {**payload_first, "name": "fc_async_second_should_not_apply"}

    # First submit
    r1 = client.post(
        "/api/tasks/predict_structure",
        data=payload_first,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    assert r1.status_code == 202

    # Wait for first to finish — FC's async dedup window covers in-flight tasks too,
    # but waiting for completion gives us a clean state to inspect.
    final = poll_job(client, base_url, task_id, timeout_s=1800, interval_s=20)
    assert final["status"] == "completed", final
    first_created_at = final["created_at"]
    first_name = final["input_params"].get("name")

    # Second submit with the SAME task_id → FC platform rejects with 409.
    # (If the platform behavior ever changes to accept and let the function
    # dedup, accept 202 too — the framework's execute_task duplicate check
    # would then catch it.  See engineering/decisions/2026-06-17-fc-async-task-mode.md.)
    r2 = client.post(
        "/api/tasks/predict_structure",
        data=payload_second,
        headers={
            "X-Fc-Invocation-Type": "Async",
            "X-Bioagent-Job-Id": task_id,
            "X-Fc-Async-Task-Id": task_id,
        },
    )
    assert r2.status_code in (202, 409), (
        f"expected 409 (FC dedup) or 202 (FC accepts → server dedups); got {r2.status_code}"
    )

    # Either way, JobInfo must not change.
    if r2.status_code == 202:
        # Function was invoked; give server-side dedup a moment.
        time.sleep(30)
    re_query = client.get(f"/api/jobs/{task_id}").json()
    assert re_query["status"] == "completed"
    assert re_query["created_at"] == first_created_at, (
        "duplicate async invoke must not reset created_at"
    )
    assert re_query["input_params"].get("name") == first_name, (
        f"input_params must reflect FIRST submit's name; got {re_query['input_params']!r}"
    )

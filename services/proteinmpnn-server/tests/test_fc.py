"""End-to-end tests against the deployed ProteinMPNN Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/proteinmpnn-server/tests/test_fc.py

URL resolves via `services/services.yaml`. Test PDB ships in `tests/data/`
(monomer example 5L33, ~180 residues — copied from upstream ProteinMPNN so the
suite is self-contained). Each inference call generates 2 sequences max.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

TEST_PDB = Path(__file__).resolve().parent / "data" / "5L33.pdb"

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("proteinmpnn-server", start=Path(__file__))


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
    files: dict | None = None,
    timeout_s: int = 600,
) -> tuple[str, dict, list[str]]:
    """Submit a job, poll to completion, return (job_id, final_status, files)."""
    r = client.post(endpoint, data=data, files=files)
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


def _upload_pdb() -> dict:
    """Return files dict for multipart PDB upload."""
    return {"pdb": (TEST_PDB.name, open(TEST_PDB, "rb"), "chemical/x-pdb")}


def _download_bytes(client: httpx.Client, job_id: str, file_path: str) -> bytes:
    """Download a single file from a job's output."""
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
    assert body["service"] == "proteinmpnn"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "proteinmpnn"
    assert body["jobs_base_dir_exists"] is True


def test_manifest_lists_three_endpoints(client: httpx.Client) -> None:
    """Manifest must list the 3 core endpoints.  The deployed service may
    additionally expose `/api/tasks/<name>` async-task variants (FC async
    task mode, enabled per-service in settings) — those are accepted but
    not required."""
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    sync_endpoints = {"/api/design", "/api/score", "/api/probs"}
    assert sync_endpoints <= paths, (
        f"expected sync endpoints {sync_endpoints} ⊆ paths, got {paths}"
    )
    extras = paths - sync_endpoints
    assert extras <= {"/api/tasks/design", "/api/tasks/score", "/api/tasks/probs"}, (
        f"unexpected non-task endpoints: {extras}"
    )


def test_manifest_model_variants(client: httpx.Client) -> None:
    """Verify service_specific model_variants match expected set."""
    body = client.get("/api/manifest").json()
    variants = body["service_specific"]["model_variants"]
    assert set(variants.keys()) == {"vanilla", "soluble", "ca_only", "abmpnn"}
    assert "v_48_020" in variants["vanilla"]["model_names"]
    assert "v_48_030" in variants["vanilla"]["model_names"]
    assert "v_48_030" not in variants["ca_only"]["model_names"]
    assert variants["abmpnn"]["model_names"] == ["abmpnn"]


def test_manifest_service_specific_keys(client: httpx.Client) -> None:
    """Verify service_specific has all expected top-level keys."""
    body = client.get("/api/manifest").json()
    extras = body["service_specific"]
    for key in ("tool_outputs", "model_variants", "input_uri_schemes", "config_tips"):
        assert key in extras, f"missing service_specific key: {key}"


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    schema = r.json()
    assert "paths" in schema


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# =====================================================================
# 422 Error inputs (fast, no GPU)
# =====================================================================

def test_422_missing_pdb(client: httpx.Client) -> None:
    """No PDB file and no pdb_uri → 422."""
    r = client.post("/api/design", data={"name": "fail"})
    assert r.status_code == 422


def test_422_invalid_model_combo(client: httpx.Client) -> None:
    """ca_only + v_48_030 is rejected by model validator."""
    r = client.post(
        "/api/design",
        data={"model_variant": "ca_only", "model_name": "v_48_030"},
        files=_upload_pdb(),
    )
    assert r.status_code == 422


def test_422_fixed_without_chains(client: httpx.Client) -> None:
    """fixed_positions without chains_to_design → 422."""
    r = client.post(
        "/api/design",
        data={"name": "fail", "fixed_positions": "1 2 3"},
        files=_upload_pdb(),
    )
    assert r.status_code == 422


def test_422_bias_AA_bad_key(client: httpx.Client) -> None:
    """bias_AA with multi-letter key → 422."""
    r = client.post(
        "/api/design",
        data={"name": "fail", "bias_AA": json.dumps({"DE": 1.0})},
        files=_upload_pdb(),
    )
    assert r.status_code == 422


def test_422_fixed_segments_mismatch(client: httpx.Client) -> None:
    """fixed_positions has 3 segments but chains_to_design has 1 chain → 422."""
    r = client.post(
        "/api/design",
        data={
            "name": "fail",
            "chains_to_design": "A",
            "fixed_positions": "1 2, 3 4, 5 6",
        },
        files=_upload_pdb(),
    )
    assert r.status_code == 422


# =====================================================================
# Inference: design
# =====================================================================

def test_design_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Smallest possible design — 2 sequences, default vanilla v_48_020."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/design",
        data={
            "name": "fc_smoke_design",
            "num_seq_per_target": "2",
            "batch_size": "1",
        },
        files=_upload_pdb(),
    )

    fa_files = [f for f in files if f.endswith(".fa")]
    assert fa_files, f"no FASTA in outputs: {files}"

    content = _download_bytes(client, job_id, fa_files[0])
    text = content.decode()
    assert text.startswith(">"), f"FASTA should start with '>': {text[:100]}"
    lines = [l for l in text.strip().splitlines() if not l.startswith(">")]
    assert len(lines) >= 2, f"expected ≥2 sequences, got {len(lines)}"


def test_design_with_chains_to_design(client: httpx.Client, base_url: str) -> None:
    """Design only chain A (the only chain in 5L33)."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/design",
        data={
            "name": "fc_chain_a",
            "num_seq_per_target": "2",
            "batch_size": "1",
            "chains_to_design": "A",
        },
        files=_upload_pdb(),
    )
    assert any(f.endswith(".fa") for f in files), f"no FASTA: {files}"


def test_design_with_fixed_positions(client: httpx.Client, base_url: str) -> None:
    """Design chain A with first 5 positions fixed."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/design",
        data={
            "name": "fc_fixed",
            "num_seq_per_target": "2",
            "batch_size": "1",
            "chains_to_design": "A",
            "fixed_positions": "1 2 3 4 5",
        },
        files=_upload_pdb(),
    )
    assert any(f.endswith(".fa") for f in files), f"no FASTA: {files}"


def test_design_with_backbone_noise(client: httpx.Client, base_url: str) -> None:
    """Design with backbone_noise=0.2 to test noise injection path."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/design",
        data={
            "name": "fc_noise",
            "num_seq_per_target": "2",
            "batch_size": "1",
            "backbone_noise": "0.2",
        },
        files=_upload_pdb(),
    )
    assert any(f.endswith(".fa") for f in files), f"no FASTA: {files}"


def test_design_ca_only_variant(client: httpx.Client, base_url: str) -> None:
    """Design with ca_only model variant + v_48_020."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/design",
        data={
            "name": "fc_ca_only",
            "num_seq_per_target": "2",
            "batch_size": "1",
            "model_variant": "ca_only",
            "model_name": "v_48_020",
        },
        files=_upload_pdb(),
    )
    assert any(f.endswith(".fa") for f in files), f"no FASTA: {files}"


def test_design_multi_temp(client: httpx.Client, base_url: str) -> None:
    """Design with multiple sampling temperatures."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/design",
        data={
            "name": "fc_multi_temp",
            "num_seq_per_target": "1",
            "batch_size": "1",
            "sampling_temp": "0.1 0.3",
        },
        files=_upload_pdb(),
    )

    fa_files = [f for f in files if f.endswith(".fa")]
    assert fa_files, f"no FASTA: {files}"

    content = _download_bytes(client, job_id, fa_files[0]).decode()
    headers = [l for l in content.strip().splitlines() if l.startswith(">")]
    assert len(headers) >= 2, f"multi-temp should produce ≥2 headers, got {len(headers)}"


# =====================================================================
# Inference: score
# =====================================================================

def test_score_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Score-only path: writes per-position scores for 2 random-sampled seqs."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/score",
        data={"name": "fc_smoke_score", "num_seq_per_target": "2"},
        files=_upload_pdb(),
    )

    score_files = [f for f in files if "score_only" in f]
    assert score_files, f"no score_only files: {files}"


def test_score_job_output_details(client: httpx.Client, base_url: str) -> None:
    """Verify score job metadata: duration, output_count, output_total_bytes."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/score",
        data={"name": "fc_score_meta", "num_seq_per_target": "2"},
        files=_upload_pdb(),
    )

    assert final["duration_seconds"] > 0
    assert final["output_count"] > 0
    assert final["output_total_bytes"] > 0


# =====================================================================
# Inference: probs
# =====================================================================

def test_probs_conditional(client: httpx.Client, base_url: str) -> None:
    """Conditional probability mode (default)."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/probs",
        data={"name": "fc_probs_cond", "kind": "conditional"},
        files=_upload_pdb(),
    )

    probs_files = [f for f in files if "conditional_probs_only" in f]
    assert probs_files, f"no conditional_probs_only files: {files}"


def test_probs_unconditional(client: httpx.Client, base_url: str) -> None:
    """Unconditional probability mode."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/probs",
        data={"name": "fc_probs_uncond", "kind": "unconditional"},
        files=_upload_pdb(),
    )

    probs_files = [f for f in files if "unconditional_probs_only" in f]
    assert probs_files, f"no unconditional_probs_only files: {files}"


def test_probs_conditional_backbone(client: httpx.Client, base_url: str) -> None:
    """Conditional-backbone probability mode."""
    job_id, final, files = _submit_and_poll(
        client, base_url, "/api/probs",
        data={"name": "fc_probs_bb", "kind": "conditional_backbone"},
        files=_upload_pdb(),
    )

    assert files, "no output files"


# =====================================================================
# input_params echo
# =====================================================================

def test_input_params_round_trip(client: httpx.Client, base_url: str) -> None:
    """Verify input_params are stored and returned correctly."""
    with open(TEST_PDB, "rb") as fh:
        r = client.post(
            "/api/design",
            files={"pdb": (TEST_PDB.name, fh, "chemical/x-pdb")},
            data={
                "name": "fc_params_rt",
                "num_seq_per_target": "3",
                "batch_size": "1",
                "model_variant": "vanilla",
                "model_name": "v_48_020",
                "backbone_noise": "0.1",
            },
        )
    r.raise_for_status()
    submit = r.json()

    assert submit["input_params"]["name"] == "fc_params_rt"
    assert submit["input_params"]["num_seq_per_target"] == 3
    assert submit["input_params"]["model_variant"] == "vanilla"
    assert submit["input_params"]["model_name"] == "v_48_020"
    assert submit["input_params"]["backbone_noise"] == 0.1

    final = poll_job(client, base_url, submit["job_id"], timeout_s=600, interval_s=10)
    assert final["status"] == "completed"
    assert final["input_params"]["name"] == "fc_params_rt"
    assert final["input_params"]["num_seq_per_target"] == 3


# =====================================================================
# Cross-job reference: job:// URI
# =====================================================================

def test_job_uri_cross_reference(client: httpx.Client, base_url: str) -> None:
    """Run a design job, then use its output PDB as input to a score job
    via the job:// URI scheme."""

    # --- Job A: design sequences ---
    job_id_a, _, files_a = _submit_and_poll(
        client, base_url, "/api/design",
        data={
            "name": "fc_ref_src",
            "num_seq_per_target": "2",
            "batch_size": "1",
        },
        files=_upload_pdb(),
    )

    pdb_files = [f for f in files_a if f.endswith(".pdb")]
    if not pdb_files:
        pytest.skip("design job produced no PDB output to reference")

    pdb_path = pdb_files[0]

    # --- Job B: score using job A's output via job:// URI ---
    r = client.post(
        "/api/score",
        data={
            "name": "fc_ref_dst",
            "num_seq_per_target": "2",
            "pdb_uri": f"job://{job_id_a}/{pdb_path}",
        },
    )
    r.raise_for_status()
    job_id_b = r.json()["job_id"]

    final = poll_job(client, base_url, job_id_b, timeout_s=600, interval_s=10)
    assert final["status"] == "completed", (
        f"failed: kind={final.get('failure_kind')} summary={final.get('error_summary')!r}"
    )

"""End-to-end tests against the deployed RFantibody Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/rfantibody-server/tests/test_fc.py

RFantibody is a pipeline: rfdiffusion → proteinmpnn → rf2. We exercise each
endpoint at least once, but chain them via module-scoped fixtures so we don't
re-run the expensive rfdiffusion step three times — the downstream tests pull
their inputs via `input_uri=job://<id>/...` URIs.

If `test_rfdiffusion_minimal_job` fails, the proteinmpnn/rf2 tests will be
marked as errored automatically (fixture dependency).

Test PDBs ship in `tests/data/` (copied from upstream RFantibody examples) so
the suite is self-contained — no dependency on `opensource/RFantibody/`.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"

TARGET_PDB = DATA_DIR / "rsv_site3.pdb"
ANTIBODY_FRAMEWORK_PDB = DATA_DIR / "hu-4D5-8_Fv.pdb"

pytestmark = pytest.mark.fc


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("rfantibody-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
        yield c


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


def _assert_completed(job: dict, client: httpx.Client) -> list[str]:
    """Assert job completed and return its file list."""
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


def _file_names(files: list) -> set[str]:
    """Normalize file list to a set of path strings."""
    if files and isinstance(files[0], dict):
        return {f["path"] for f in files}
    return set(files)


def _upload_pdbs() -> dict:
    """Return files dict for multipart PDB uploads (target + framework)."""
    return {
        "target": (TARGET_PDB.name, open(TARGET_PDB, "rb"), "chemical/x-pdb"),
        "framework": (ANTIBODY_FRAMEWORK_PDB.name, open(ANTIBODY_FRAMEWORK_PDB, "rb"), "chemical/x-pdb"),
    }


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
    assert body["service"] == "rfantibody"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "rfantibody"
    assert body["jobs_base_dir_exists"] is True


def test_manifest_lists_three_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/rfdiffusion", "/api/proteinmpnn", "/api/rf2"}


def test_manifest_service_specific_keys(client: httpx.Client) -> None:
    """Verify service_specific has all expected top-level keys."""
    body = client.get("/api/manifest").json()
    extras = body["service_specific"]
    for key in ("tool_outputs", "input_uri_schemes", "chaining_tip", "weights"):
        assert key in extras, f"missing service_specific key: {key}"


def test_manifest_tool_outputs(client: httpx.Client) -> None:
    """Verify tool_outputs maps each step to its conventional filename."""
    body = client.get("/api/manifest").json()
    outputs = body["service_specific"]["tool_outputs"]
    assert outputs["rfdiffusion"].endswith("1_rfdiffusion.qv")
    assert outputs["proteinmpnn"].endswith("2_proteinmpnn.qv")
    assert outputs["rf2"].endswith("3_rf2.qv")


def test_manifest_input_uri_schemes(client: httpx.Client) -> None:
    """Verify all supported URI schemes are declared."""
    body = client.get("/api/manifest").json()
    schemes = body["service_specific"]["input_uri_schemes"]
    assert "job://<job_id>/<filename>" in schemes
    assert "oss://<bucket>/<key>" in schemes


def test_manifest_chaining_tip_mentions_job_uri(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    assert "job://" in body["service_specific"]["chaining_tip"]


def test_manifest_weights_lists_all_three(client: httpx.Client) -> None:
    body = client.get("/api/manifest").json()
    weights = body["service_specific"]["weights"]
    assert "rfdiffusion" in weights
    assert "proteinmpnn" in weights
    assert "rf2" in weights


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    schema = r.json()
    assert "paths" in schema
    assert "/api/rfdiffusion" in schema["paths"]
    assert "/api/proteinmpnn" in schema["paths"]
    assert "/api/rf2" in schema["paths"]


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# =====================================================================
# 422 Error inputs (fast, no GPU)
# =====================================================================

def test_422_rfdiffusion_missing_target(client: httpx.Client) -> None:
    """rfdiffusion without target PDB → 422."""
    with open(ANTIBODY_FRAMEWORK_PDB, "rb") as f:
        r = client.post(
            "/api/rfdiffusion",
            files={"framework": (ANTIBODY_FRAMEWORK_PDB.name, f, "chemical/x-pdb")},
            data={"num_designs": "1"},
        )
    assert r.status_code == 422


def test_422_rfdiffusion_missing_framework(client: httpx.Client) -> None:
    """rfdiffusion without framework PDB → 422."""
    with open(TARGET_PDB, "rb") as t:
        r = client.post(
            "/api/rfdiffusion",
            files={"target": (TARGET_PDB.name, t, "chemical/x-pdb")},
            data={"num_designs": "1"},
        )
    assert r.status_code == 422


def test_422_proteinmpnn_missing_input(client: httpx.Client) -> None:
    """proteinmpnn without input_quiver or input_uri → 422."""
    r = client.post("/api/proteinmpnn", data={"seqs_per_struct": "1"})
    assert r.status_code == 422


def test_422_rf2_missing_input(client: httpx.Client) -> None:
    """rf2 without input_quiver or input_uri → 422."""
    r = client.post("/api/rf2", data={"num_recycles": "2"})
    assert r.status_code == 422


def test_422_rfdiffusion_diffuser_t_out_of_range(client: httpx.Client) -> None:
    """diffuser_t=0 is below ge=1 → 422."""
    r = client.post(
        "/api/rfdiffusion",
        files=_upload_pdbs(),
        data={"num_designs": "1", "diffuser_t": "0"},
    )
    assert r.status_code == 422


def test_422_proteinmpnn_temperature_out_of_range(client: httpx.Client) -> None:
    """temperature=0 is below ge=0.01 → 422."""
    r = client.post(
        "/api/proteinmpnn",
        data={"input_uri": "file:///nonexistent.qv", "temperature": "0"},
    )
    assert r.status_code == 422


def test_422_rf2_num_recycles_out_of_range(client: httpx.Client) -> None:
    """num_recycles=100 is above le=50 → 422."""
    r = client.post(
        "/api/rf2",
        data={"input_uri": "file:///nonexistent.qv", "num_recycles": "100"},
    )
    assert r.status_code == 422


# =====================================================================
# Chained inference — module-scoped fixtures so rfdiffusion runs once.
# =====================================================================

@pytest.fixture(scope="module")
def rfdiffusion_result(client: httpx.Client, base_url: str) -> tuple[dict, list[str]]:
    """Run rfdiffusion once, return (completed job dict, file list)."""
    with open(TARGET_PDB, "rb") as t, open(ANTIBODY_FRAMEWORK_PDB, "rb") as f:
        r = client.post(
            "/api/rfdiffusion",
            files={
                "target": (TARGET_PDB.name, t, "chemical/x-pdb"),
                "framework": (ANTIBODY_FRAMEWORK_PDB.name, f, "chemical/x-pdb"),
            },
            data={
                "num_designs": "1",
                "diffuser_t": "25",
                "design_loops": "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13",
                "hotspots": "T305,T456",
                "deterministic": "true",
            },
        )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)

    final = poll_job(client, base_url, submit["job_id"], timeout_s=1800, interval_s=20)
    files = _assert_completed(final, client)
    return final, files


@pytest.fixture(scope="module")
def proteinmpnn_result(
    client: httpx.Client, base_url: str, rfdiffusion_result: tuple[dict, list[str]]
) -> tuple[dict, list[str]]:
    """Run proteinmpnn off the rfdiffusion output (via job:// URI)."""
    rfd_job = rfdiffusion_result[0]
    r = client.post(
        "/api/proteinmpnn",
        data={
            "input_uri": f"job://{rfd_job['job_id']}/1_rfdiffusion.qv",
            "seqs_per_struct": "1",
            "deterministic": "true",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)

    final = poll_job(client, base_url, submit["job_id"], timeout_s=1800, interval_s=20)
    files = _assert_completed(final, client)
    return final, files


@pytest.fixture(scope="module")
def rf2_result(
    client: httpx.Client, base_url: str, proteinmpnn_result: tuple[dict, list[str]]
) -> tuple[dict, list[str]]:
    """Run rf2 off the proteinmpnn output."""
    mpnn_job = proteinmpnn_result[0]
    r = client.post(
        "/api/rf2",
        data={
            "input_uri": f"job://{mpnn_job['job_id']}/2_proteinmpnn.qv",
            "num_recycles": "2",
        },
    )
    r.raise_for_status()
    submit = r.json()
    _assert_submitted(submit)

    final = poll_job(client, base_url, submit["job_id"], timeout_s=1800, interval_s=20)
    files = _assert_completed(final, client)
    return final, files


# =====================================================================
# Inference: rfdiffusion
# =====================================================================

def test_rfdiffusion_produces_qv(
    client: httpx.Client, rfdiffusion_result: tuple[dict, list[str]]
) -> None:
    """Verify the rfdiffusion job produced a .qv output."""
    _, files = rfdiffusion_result
    names = _file_names(files)
    assert any("1_rfdiffusion.qv" in n for n in names), f"missing .qv: {names}"


def test_rfdiffusion_qv_is_nonempty(
    client: httpx.Client, rfdiffusion_result: tuple[dict, list[str]]
) -> None:
    """Download the .qv and verify it has content."""
    job, files = rfdiffusion_result
    names = _file_names(files)
    qv = next(n for n in names if "1_rfdiffusion.qv" in n)
    data = _download_bytes(client, job["job_id"], qv)
    assert len(data) > 100, f".qv file unexpectedly small: {len(data)} bytes"


def test_rfdiffusion_input_params_echo(
    rfdiffusion_result: tuple[dict, list[str]]
) -> None:
    """Verify input_params are stored and returned correctly."""
    job, _ = rfdiffusion_result
    params = job["input_params"]
    assert params["num_designs"] == 1
    assert params["diffuser_t"] == 25
    assert params["design_loops"] == "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13"
    assert params["hotspots"] == "T305,T456"
    assert params["deterministic"] is True


def test_rfdiffusion_job_metadata(
    rfdiffusion_result: tuple[dict, list[str]]
) -> None:
    """Verify job metadata fields are populated after completion."""
    job, _ = rfdiffusion_result
    assert job["duration_seconds"] > 0
    assert job["output_count"] > 0
    assert job["output_total_bytes"] > 0


# =====================================================================
# Inference: proteinmpnn
# =====================================================================

def test_proteinmpnn_produces_qv(
    client: httpx.Client, proteinmpnn_result: tuple[dict, list[str]]
) -> None:
    _, files = proteinmpnn_result
    names = _file_names(files)
    assert any("2_proteinmpnn.qv" in n for n in names), f"missing .qv: {names}"


def test_proteinmpnn_qv_is_nonempty(
    client: httpx.Client, proteinmpnn_result: tuple[dict, list[str]]
) -> None:
    job, files = proteinmpnn_result
    names = _file_names(files)
    qv = next(n for n in names if "2_proteinmpnn.qv" in n)
    data = _download_bytes(client, job["job_id"], qv)
    assert len(data) > 100, f".qv file unexpectedly small: {len(data)} bytes"


def test_proteinmpnn_input_params_echo(
    proteinmpnn_result: tuple[dict, list[str]]
) -> None:
    job, _ = proteinmpnn_result
    params = job["input_params"]
    assert params["seqs_per_struct"] == 1
    assert params["deterministic"] is True
    assert params["loops"] == "H1,H2,H3"
    assert params["temperature"] == 0.2
    assert params["omit_aas"] == "CX"


def test_proteinmpnn_job_metadata(
    proteinmpnn_result: tuple[dict, list[str]]
) -> None:
    job, _ = proteinmpnn_result
    assert job["duration_seconds"] > 0
    assert job["output_count"] > 0
    assert job["output_total_bytes"] > 0


# =====================================================================
# Inference: rf2
# =====================================================================

def test_rf2_produces_qv(
    client: httpx.Client, rf2_result: tuple[dict, list[str]]
) -> None:
    _, files = rf2_result
    names = _file_names(files)
    assert any("3_rf2.qv" in n for n in names), f"missing .qv: {names}"


def test_rf2_qv_is_nonempty(
    client: httpx.Client, rf2_result: tuple[dict, list[str]]
) -> None:
    job, files = rf2_result
    names = _file_names(files)
    qv = next(n for n in names if "3_rf2.qv" in n)
    data = _download_bytes(client, job["job_id"], qv)
    assert len(data) > 100, f".qv file unexpectedly small: {len(data)} bytes"


def test_rf2_input_params_echo(
    rf2_result: tuple[dict, list[str]]
) -> None:
    job, _ = rf2_result
    params = job["input_params"]
    assert params["num_recycles"] == 2
    assert params["hotspot_show_prop"] == 0.1


def test_rf2_job_metadata(
    rf2_result: tuple[dict, list[str]]
) -> None:
    job, _ = rf2_result
    assert job["duration_seconds"] > 0
    assert job["output_count"] > 0
    assert job["output_total_bytes"] > 0


# =====================================================================
# Cross-job: job status retrieval after completion
# =====================================================================

def test_completed_job_status_endpoint(
    client: httpx.Client, rfdiffusion_result: tuple[dict, list[str]]
) -> None:
    """GET /api/jobs/{id} should return the completed job with all metadata."""
    job, _ = rfdiffusion_result
    r = client.get(f"/api/jobs/{job['job_id']}")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "completed"
    assert body["job_id"] == job["job_id"]
    assert body["input_params"] is not None


def test_completed_job_files_endpoint(
    client: httpx.Client, rfdiffusion_result: tuple[dict, list[str]]
) -> None:
    """GET /api/jobs/{id}/files should list output files."""
    job, _ = rfdiffusion_result
    r = client.get(f"/api/jobs/{job['job_id']}/files")
    r.raise_for_status()
    body = r.json()
    assert "files" in body
    assert len(body["files"]) > 0

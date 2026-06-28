"""FC integration tests for diffusion-hopping-server (opt-in).

Marked `@pytest.mark.fc`, skipped by default. Run with:

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/diffusion-hopping-server/tests/test_fc.py -v

Fixtures (1a0q_protein.pdb + 1a0q_ligand.sdf) ship in tests/data/ — copied
from upstream's tests_data/complexes/1a0q/.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_PROTEIN = DATA_DIR / "1a0q_protein.pdb"
TEST_LIGAND = DATA_DIR / "1a0q_ligand.sdf"

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("diffusion-hopping-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
        yield c


def _save_job_outputs(
    client: httpx.Client, job_id: str, job_info: dict, dst_dir: Path,
) -> None:
    """Persist JobInfo / log / zip / extracted SDFs to dst_dir for inspection."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "jobinfo.json").write_text(json.dumps(job_info, indent=2))
    try:
        r = client.get(f"/api/jobs/{job_id}/log")
        if r.status_code == 200:
            body = r.json()
            (dst_dir / "subprocess.log").write_text(
                body.get("log") or body.get("text") or ""
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[fc_outputs] log download failed: {exc!r}")
    try:
        r = client.get(f"/api/jobs/{job_id}/download")
        if r.status_code == 200 and r.content:
            (dst_dir / f"{job_id}.zip").write_bytes(r.content)
            extract_to = dst_dir / "extracted"
            extract_to.mkdir(exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                zf.extractall(extract_to)
    except Exception as exc:  # noqa: BLE001
        print(f"[fc_outputs] zip download failed: {exc!r}")


# ----- Smoke -----


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "diffusion-hopping"


def test_healthz_detail_weights_loaded(client: httpx.Client) -> None:
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["weights_loaded"] is True, (
        f"NAS weights missing: {body.get('weights_missing')}"
    )


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    sync_endpoints = {"/api/generate"}
    assert sync_endpoints <= paths


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ----- Inference -----


def test_generate_minimal_job(
    client: httpx.Client, base_url: str, local_output_dir: Path,
) -> None:
    """End-to-end: 1a0q protein + ref ligand → 3 scaffold candidates."""
    with open(TEST_PROTEIN, "rb") as fh_p, open(TEST_LIGAND, "rb") as fh_l:
        r = client.post(
            "/api/generate",
            files={
                "protein": ("1a0q_protein.pdb", fh_p.read(), "chemical/x-pdb"),
                "reference_ligand": ("1a0q_ligand.sdf", fh_l.read(),
                                     "chemical/x-mdl-sdfile"),
            },
            data={"num_samples": "3", "model_variant": "gvp_conditional"},
        )
    r.raise_for_status()
    job_id = r.json()["job_id"]
    final = poll_job(client, base_url, job_id, timeout_s=900, interval_s=15)
    _save_job_outputs(client, job_id, final, local_output_dir / "generate_gvp")
    assert final["status"] == "completed", final
    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    sdf_files = [f for f in files if f.endswith(".sdf")]
    assert sdf_files, f"no SDFs in outputs: {files}"


def test_generate_egnn_variant(
    client: httpx.Client, base_url: str, local_output_dir: Path,
) -> None:
    """Run with EGNN backbone instead of GVP — verifies all 4 ckpts load."""
    with open(TEST_PROTEIN, "rb") as fh_p, open(TEST_LIGAND, "rb") as fh_l:
        r = client.post(
            "/api/generate",
            files={
                "protein": ("1a0q_protein.pdb", fh_p.read(), "chemical/x-pdb"),
                "reference_ligand": ("1a0q_ligand.sdf", fh_l.read(),
                                     "chemical/x-mdl-sdfile"),
            },
            data={"num_samples": "3", "model_variant": "egnn_conditional"},
        )
    r.raise_for_status()
    job_id = r.json()["job_id"]
    final = poll_job(client, base_url, job_id, timeout_s=900, interval_s=15)
    _save_job_outputs(client, job_id, final, local_output_dir / "generate_egnn")
    assert final["status"] == "completed", final
    assert final["input_params"]["model_variant"] == "egnn_conditional"

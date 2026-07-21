"""End-to-end smoke tests against the deployed diffdock-server FC function.

Marked `@pytest.mark.fc`, skipped by default.  Run with::

    RUN_FC_TESTS=1 uv run python -m pytest -m fc \\
        services/diffdock-server/tests/test_fc.py -v

URL resolves via ``services/services.yaml``.

Inference-heavy tests live in ``test_fc_task.py`` (async task mode) —
this file only covers cheap health + manifest smoke, plus 422 input
validation.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, make_retrying_client

pytestmark = pytest.mark.fc

SERVICE = "diffdock-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
PROTEIN_PDB = DATA_DIR / "1a0q_protein.pdb"
LIGAND_SDF = DATA_DIR / "1a0q_ligand.sdf"


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with make_retrying_client(base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# ---- Smoke ---------------------------------------------------------------


def test_healthz(client: httpx.Client) -> None:
    body = client.get("/healthz").raise_for_status().json()
    assert body["status"] == "ok"
    assert body["service"] == "diffdock"
    assert "version" in body


def test_healthz_detail_weights_and_lut(client: httpx.Client) -> None:
    body = client.get("/healthz/detail").raise_for_status().json()
    assert body["status"] == "ok"
    assert body["service"] == "diffdock"
    assert body["weights_loaded"] is True, (
        f"weights not present on NAS: weights_dir={body.get('weights_dir')}, "
        f"missing={body.get('weights_missing')}"
    )
    assert body["so3_cache_ok"] is True, (
        f"SO(3) LUT cache missing at {body.get('weights_dir')} — precompute "
        f"step may have failed at build time"
    )
    assert body["torus_cache_ok"] is True
    # ESMFold is optional (only needed for protein_sequence branch) — report only
    assert "esmfold_available" in body
    assert isinstance(body["active_jobs"], int)
    assert isinstance(body["max_concurrent_jobs"], int)


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/dock" in paths
    assert "/api/tasks/dock" in paths


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    spec = r.json()
    expected = {"/api/dock", "/api/tasks/dock"}
    missing = expected - set(spec.get("paths", {}))
    assert not missing, f"missing from OpenAPI: {missing}"


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    r = client.get("/api/jobs/does-not-exist-abc123")
    assert r.status_code == 404


# ---- 422 input validation ------------------------------------------------


def test_dock_missing_inputs_returns_422(client: httpx.Client) -> None:
    """No protein AND no ligand → 422."""
    r = client.post("/api/dock", data={"complex_name": "empty"})
    assert r.status_code == 422


def test_dock_missing_ligand_returns_422(client: httpx.Client) -> None:
    """Protein present but no ligand → 422."""
    with open(PROTEIN_PDB, "rb") as fp:
        r = client.post(
            "/api/dock",
            files={
                "protein": (PROTEIN_PDB.name, fp.read(), "chemical/x-pdb"),
            },
            data={"complex_name": "no_ligand"},
        )
    assert r.status_code == 422


def test_dock_two_protein_inputs_returns_422(client: httpx.Client) -> None:
    """Both protein file and protein_sequence → 422."""
    with open(PROTEIN_PDB, "rb") as fp, open(LIGAND_SDF, "rb") as fl:
        r = client.post(
            "/api/dock",
            files={
                "protein": (PROTEIN_PDB.name, fp.read(), "chemical/x-pdb"),
                "ligand": (LIGAND_SDF.name, fl.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "protein_sequence": "MKW" * 40,
                "complex_name": "conflict",
            },
        )
    # 422 direct, or the pydantic model validator catches it first
    assert r.status_code == 422, r.text

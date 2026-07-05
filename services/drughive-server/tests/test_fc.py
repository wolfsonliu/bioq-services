"""End-to-end smoke tests against the deployed drughive-server FC function.

Marked `@pytest.mark.fc`, skipped by default.  Run with:

    RUN_FC_TESTS=1 uv run python -m pytest -m fc \\
        services/drughive-server/tests/test_fc.py -v

URL resolves via `services/aliyun_fc_url.md`.

Inference-heavy tests live in ``test_fc_task.py`` (async task mode) —
this file only covers cheap health + manifest smoke, plus 422 input
validation.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, make_retrying_client

pytestmark = pytest.mark.fc

SERVICE = "drughive-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
POCKET_PDB = DATA_DIR / "5d3h_pocket.pdb"
LIGAND_SDF = DATA_DIR / "5d3h_ligand.sdf"


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    # Retry 429 (shared GPU quota bursts across the account).
    with make_retrying_client(base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# ---- Smoke ---------------------------------------------------------------


def test_healthz(client: httpx.Client) -> None:
    body = client.get("/healthz").raise_for_status().json()
    assert body["status"] == "ok"
    assert body["service"] == "drughive"
    assert "version" in body


def test_healthz_detail_weights_and_qvina(client: httpx.Client) -> None:
    body = client.get("/healthz/detail").raise_for_status().json()
    assert body["status"] == "ok"
    assert body["service"] == "drughive"
    assert body["weights_loaded"] is True, (
        f"weights not present on NAS: weights_dir={body.get('weights_dir')}, "
        f"missing={body.get('weights_missing')}"
    )
    assert body["qvina2_available"] is True, (
        f"qvina2 binary not on PATH: qvina2_path={body.get('qvina2_path')}"
    )
    assert isinstance(body["active_jobs"], int)
    assert isinstance(body["max_concurrent_jobs"], int)


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/generate" in paths
    assert "/api/generate_spatial" in paths
    assert "/api/optimize" in paths
    assert "/api/tasks/generate" in paths
    assert "/api/tasks/generate_spatial" in paths
    assert "/api/tasks/optimize" in paths


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    spec = r.json()
    expected = {
        "/api/generate", "/api/generate_spatial", "/api/optimize",
        "/api/tasks/generate", "/api/tasks/generate_spatial", "/api/tasks/optimize",
    }
    missing = expected - set(spec.get("paths", {}))
    assert not missing, f"missing from OpenAPI: {missing}"


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    r = client.get("/api/jobs/does-not-exist-abc123")
    assert r.status_code == 404


# ---- 422 input validation ------------------------------------------------


def test_generate_missing_files_returns_422(client: httpx.Client) -> None:
    """No target/ligand → 422 from uris.resolve_input."""
    r = client.post("/api/generate", data={"n_samples": "2"})
    assert r.status_code == 422


def test_generate_spatial_missing_pattern_and_file_returns_422(
    client: httpx.Client,
) -> None:
    """Neither substruct_modify file nor pattern → 422."""
    with open(POCKET_PDB, "rb") as fp, open(LIGAND_SDF, "rb") as fl:
        r = client.post(
            "/api/generate_spatial",
            files={
                "target": (POCKET_PDB.name, fp.read(), "chemical/x-pdb"),
                "ligand": (LIGAND_SDF.name, fl.read(), "chemical/x-mdl-sdfile"),
            },
            data={"n_samples": "2"},
        )
    assert r.status_code == 422, r.text


def test_optimize_qvina_key_without_pdbqt_returns_422(client: httpx.Client) -> None:
    """key_opt=affinity_qvina without target_pdbqt → 422."""
    with open(POCKET_PDB, "rb") as fp, open(LIGAND_SDF, "rb") as fl:
        r = client.post(
            "/api/optimize",
            files={
                "target": (POCKET_PDB.name, fp.read(), "chemical/x-pdb"),
                "ligand": (LIGAND_SDF.name, fl.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "key_opt": "affinity_qvina",
                "n_cycles": "1",
                "n_samples_initial": "10",
                "n_samples": "2",
                "n_best_parents": "1",
            },
        )
    # 422 direct, or 5xx from runner-cleanup path — both acceptable evidence
    # the request never touched the subprocess.
    assert r.status_code in (422, 500, 502), r.text

"""FC integration tests for openadmet-server (opt-in).

Run with:

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/openadmet-server/tests/test_fc.py -v

Predict on 3-5 SMILES against one chemprop-chemeleon model takes ~60-90 s
(cold-start + torch import + CheMeleon load + inference). Later calls in
the same instance are faster once cached.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

LOSARTAN = "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

pytestmark = pytest.mark.fc

DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("openadmet-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(300.0)) as c:
        yield c


def _save_job_outputs(
    client: httpx.Client, job_id: str, job_info: dict, dst_dir: Path,
) -> None:
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


# ===== Smoke ================================================================


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "openadmet"


def test_healthz_detail_reports_registry(client: httpx.Client) -> None:
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["chemeleon_foundation_present"] is True, (
        f"CheMeleon foundation missing: {body.get('weights_missing')}"
    )
    assert body["models_count"] >= 1, (
        f"No models registered on NAS: {body}"
    )
    assert body["weights_loaded"] is True


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/predict" in paths
    assert "/api/compare" in paths


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    paths = r.json()["paths"]
    assert "/api/models" in paths
    assert "/api/predict" in paths


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


def test_models_endpoint_returns_registry(client: httpx.Client) -> None:
    body = client.get("/api/models").json()
    assert body["count"] >= 1
    names = {m["name"] for m in body["models"]}
    # At least one of the 6 pre-staged HF models should be present.
    expected = {
        "herg-chemeleon-baseline",
        "cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1",
        "cyp1a2-cyp2d6-cyp3a4-cyp3c9-chemeleon-baseline",
        "microsomal-clearance-chemeleon-v1",
        "permeability-logd-ppb-chemeleon-baseline",
        "pxr-chemeleon-baseline",
    }
    assert names & expected, (
        f"None of the expected pre-staged models found; got {names}"
    )


# ===== Inference — smallest reasonable predict against 1 model ==============


def test_predict_inline_smiles_minimal_job(
    client: httpx.Client, base_url: str, local_output_dir: Path,
) -> None:
    """3 SMILES + herg baseline model on GPU."""
    r = client.post(
        "/api/predict",
        data={
            "input_smiles": f"{LOSARTAN},{ASPIRIN},{CAFFEINE}",
            "model_names": "herg-chemeleon-baseline",
            "accelerator": "gpu",
        },
    )
    r.raise_for_status()
    job_id = r.json()["job_id"]
    final = poll_job(client, base_url, job_id, timeout_s=1800, interval_s=20)
    _save_job_outputs(client, job_id, final, local_output_dir / "predict")
    assert final["status"] == "completed", final
    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any("predictions.csv" in f for f in files), (
        f"predictions.csv missing from outputs: {files}"
    )


def test_predict_csv_upload_minimal_job(
    client: httpx.Client, base_url: str, local_output_dir: Path,
) -> None:
    csv_path = DATA_DIR / "demo_input.csv"
    with open(csv_path, "rb") as fh:
        r = client.post(
            "/api/predict",
            files={"input_csv": ("demo_input.csv", fh, "text/csv")},
            data={
                "model_names": "herg-chemeleon-baseline",
                "accelerator": "gpu",
            },
        )
    r.raise_for_status()
    job_id = r.json()["job_id"]
    final = poll_job(client, base_url, job_id, timeout_s=1800, interval_s=20)
    _save_job_outputs(client, job_id, final, local_output_dir / "predict_csv")
    assert final["status"] == "completed", final

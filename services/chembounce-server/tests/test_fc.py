"""FC integration tests for chembounce-server (opt-in).

Run with:

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/chembounce-server/tests/test_fc.py -v

ChemBounce on the simplest input takes minutes; the long-tail "complex
molecule" cases can take 30+ min, so test_fc.py only does smoke + one
small inference.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

LOSARTAN = "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("chembounce-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
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


# ----- Smoke -----


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "chembounce"


def test_healthz_detail_db_present(client: httpx.Client) -> None:
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    # Service is usable iff 250mw DB is present.
    assert body["database_status"]["250mw"] is True, (
        f"250mw DB missing: {body.get('weights_missing')}"
    )


def test_manifest_lists_scaffold_hop(client: httpx.Client) -> None:
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/scaffold_hop" in paths


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ----- Inference -----


def test_scaffold_hop_minimal_job(
    client: httpx.Client, base_url: str, local_output_dir: Path,
) -> None:
    """Smallest viable call: losartan + frag_max_n=10 + 250mw DB."""
    r = client.post(
        "/api/scaffold_hop",
        data={
            "input_smiles": LOSARTAN,
            "frag_max_n": "10",
            "tanimoto_threshold": "0.5",
            "database": "250mw",
        },
    )
    r.raise_for_status()
    job_id = r.json()["job_id"]
    final = poll_job(client, base_url, job_id, timeout_s=1800, interval_s=20)
    _save_job_outputs(client, job_id, final, local_output_dir / "scaffold_hop")
    assert final["status"] == "completed", final
    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert any("overall_result.txt" in f for f in files), (
        f"overall_result.txt missing from outputs: {files}"
    )

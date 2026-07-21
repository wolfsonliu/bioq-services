"""End-to-end tests against the deployed Promera Function Compute service.

Marked ``@pytest.mark.fc``, skipped by default. Run with:

    pytest -m fc services/promera-server/tests/test_fc.py

Test fixtures ship in ``tests/data/``, so the suite is self-contained.

After each long-running test the JobInfo JSON + log + raw zip + extracted
output files are downloaded to ``tests/fc_outputs/run-<timestamp>/<label>/``
(see ``local_output_dir`` fixture in conftest.py) so a human can inspect
the actual predicted structures after the run.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_TARGET = DATA_DIR / "test_target.json"

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("promera-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


def _save_job_outputs(
    client: httpx.Client,
    job_id: str,
    job_info: dict,
    dst_dir: Path,
) -> None:
    """Download JobInfo / log / zip / extracted output into ``dst_dir``.

    Best-effort: any individual download failure is logged but does NOT
    raise — the test's own assertions remain the source of truth for
    pass/fail.  Call this *before* assertions so the artifacts are
    available even when the subprocess failed.
    """
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
        print(f"[fc_outputs] log download failed for {job_id}: {exc!r}")

    try:
        r = client.get(f"/api/jobs/{job_id}/download")
        if r.status_code == 200 and r.content:
            (dst_dir / f"{job_id}.zip").write_bytes(r.content)
            extract_to = dst_dir / "extracted"
            extract_to.mkdir(exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                zf.extractall(extract_to)
    except Exception as exc:  # noqa: BLE001
        print(f"[fc_outputs] zip download failed for {job_id}: {exc!r}")

    print(f"[fc_outputs] saved {job_id} → {dst_dir}")


# ----- Smoke -----


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "promera"
    assert "version" in body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/cofold", "/api/design"}


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ----- Cofold inference -----


def test_cofold_minimal(
    client: httpx.Client, base_url: str, local_output_dir: Path
) -> None:
    with open(TEST_TARGET, "rb") as fh:
        r = client.post(
            "/api/cofold",
            files={"input_schema": ("ubiquitin.json", fh, "application/json")},
            data={"num_seeds": "1", "diffusion_samples": "1", "diffusion_steps": "50"},
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _save_job_outputs(client, final["job_id"], final, local_output_dir / "cofold")
    assert final["status"] == "completed", final
    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert any(f.endswith(".cif") for f in files)


# ----- Design inference -----


def test_design_minibinder_minimal(
    client: httpx.Client, base_url: str, local_output_dir: Path
) -> None:
    with open(TEST_TARGET, "rb") as fh:
        r = client.post(
            "/api/design",
            files={"target_schema": ("target.json", fh, "application/json")},
            data={
                "design_type": "minibinder",
                "num_backbones": "1",
                "diffusion_steps": "50",
                "inverse_folder_type": "none",
            },
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _save_job_outputs(client, final["job_id"], final, local_output_dir / "design")
    assert final["status"] == "completed", final
    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert any("backbone.cif" in f for f in files)

"""End-to-end tests against the deployed genie3 Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/genie3-server/tests/test_fc.py

Motif and binder endpoints need a `dataset` zip with `problems/<name>.json` plus
referenced `motifs/` or `targets/` files. We build minimal one-problem zips on
the fly from fixtures in `tests/data/` (copied from upstream genie3 so the
suite is self-contained — no dependency on `opensource/genie3/`).

Each generation call sets `n_sample=1` / smallest viable length to keep the FC
GPU job under ~5 min.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest
import yaml

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
MOTIFBENCH = DATA_DIR / "motifbench"
BINDERTEST = DATA_DIR / "binder"

pytestmark = pytest.mark.fc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("genie3-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
        yield c


def _build_zip(files: dict[str, Path]) -> bytes:
    """Build an in-memory zip mapping archive paths → on-disk files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, src in files.items():
            zf.write(src, arcname=arcname)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "genie3"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["service"] == "genie3"
    assert body["jobs_base_dir_exists"] is True


def test_manifest_lists_four_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {
        "/api/generate/unconditional",
        "/api/generate/motif",
        "/api/generate/binder",
        "/api/generate",
    }


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ---------------------------------------------------------------------------
# Inference — one minimal job per endpoint.
# ---------------------------------------------------------------------------


def _assert_completed(job: dict, base_url: str, client: httpx.Client) -> None:
    assert job["status"] == "completed", (
        f"failed: kind={job.get('failure_kind')} summary={job.get('error_summary')!r}"
    )
    files = client.get(f"/api/jobs/{job['job_id']}/files").json()["files"]
    assert files, "no output files"


def test_unconditional_minimal_job(client: httpx.Client, base_url: str) -> None:
    """No dataset upload — 1 sample, 50-residue monomer."""
    r = client.post(
        "/api/generate/unconditional",
        data={
            "n_sample": "1",
            "batch_size": "1",
            "min_length": "50",
            "max_length": "50",
            "length_step": "50",
        },
    )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _assert_completed(final, base_url, client)


def test_motif_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Single-problem motif zip built on the fly from motifbench/01_1LDB."""
    zip_bytes = _build_zip(
        {
            "problems/01_1LDB.json": MOTIFBENCH / "problems" / "01_1LDB.json",
            "motifs/01_1LDB.pdb": MOTIFBENCH / "motifs" / "01_1LDB.pdb",
        }
    )
    r = client.post(
        "/api/generate/motif",
        files={"dataset": ("motif.zip", zip_bytes, "application/zip")},
        data={
            "n_sample": "1",
            "batch_size": "1",
            "selections": "01_1LDB",
        },
    )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _assert_completed(final, base_url, client)


def test_binder_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Single-problem binder zip built from genie3/test/binder/01_bhrf1."""
    problem_json = BINDERTEST / "problems" / "01_bhrf1.json"
    zip_bytes = _build_zip(
        {
            "problems/01_bhrf1.json": problem_json,
            "targets/pdb/01_bhrf1.pdb": BINDERTEST / "targets" / "pdb" / "01_bhrf1.pdb",
            "targets/pdb/01_bhrf1-chain_B.pdb": BINDERTEST / "targets" / "pdb" / "01_bhrf1-chain_B.pdb",
            "targets/fasta/01_bhrf1.fasta": BINDERTEST / "targets" / "fasta" / "01_bhrf1.fasta",
            "targets/fasta/01_bhrf1-chain_B.fasta": BINDERTEST / "targets" / "fasta" / "01_bhrf1-chain_B.fasta",
            "targets/msa/01_bhrf1.a3m": BINDERTEST / "targets" / "msa" / "01_bhrf1.a3m",
            "targets/msa/01_bhrf1-chain_B.a3m": BINDERTEST / "targets" / "msa" / "01_bhrf1-chain_B.a3m",
        }
    )
    r = client.post(
        "/api/generate/binder",
        files={"dataset": ("binder.zip", zip_bytes, "application/zip")},
        data={
            "n_sample": "1",
            "batch_size": "1",
            "selections": "01_bhrf1",
        },
    )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _assert_completed(final, base_url, client)


def test_custom_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Freeform `/api/generate` with a tiny unconditional YAML (no dataset)."""
    config = {
        "experiment": {"name": "fc_smoke_custom"},
        "paths": {"rootdir": "PLACEHOLDER_OVERRIDDEN_BY_SERVER"},
        "generation": {
            "dataset": {
                "source": "unconditional",
                "min_length": 50,
                "max_length": 50,
                "length_step": 50,
                "n_sample": 1,
            },
            "sampler": {"sampler": {"direction_scale": 0.8}},
        },
        "evaluation": {"version": "unconditional", "folding": {"model_name": "esmfold"}},
    }
    r = client.post(
        "/api/generate",
        data={"config_yaml": yaml.safe_dump(config)},
    )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _assert_completed(final, base_url, client)

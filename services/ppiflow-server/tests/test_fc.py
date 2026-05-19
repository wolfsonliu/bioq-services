"""End-to-end tests against the deployed PPIFlow Function Compute service.

Marked `@pytest.mark.fc` and skipped by default. To run:

    pytest -m fc services/ppiflow-server/tests/test_fc.py
    # or
    RUN_FC_TESTS=1 pytest services/ppiflow-server/tests/test_fc.py

URL is read from `services/aliyun_fc_url.md` (the single source of truth — the
helper raises if the service isn't listed). Inference tests use the example
PDBs in `tests/data/` (copied from upstream PPIFlow so the test suite is
self-contained — no dependency on `opensource/PPIFlow/`).

Each inference test submits one minimal job (smallest possible `samples_per_target`,
shortest viable length range), polls until terminal, and asserts on the JobInfo
contract: `status=completed`, at least one output file. Total runtime per
endpoint is ~5-15 min of FC GPU time, dominated by PPIFlow's per-job overhead.

The scaffolding endpoint is currently SKIPPED because its CSV format references
motif PDB paths that must exist on the FC NAS, which we cannot stage from the
test runner.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"

ANTIGEN_PDB = DATA_DIR / "1IJZ_IL13.pdb"
SCFV_FRAMEWORK_PDB = DATA_DIR / "6nou_scfv_framework.pdb"
NANOBODY_FRAMEWORK_PDB = DATA_DIR / "7eow_nanobody_framework.pdb"

pytestmark = pytest.mark.fc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("ppiflow-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    # `timeout` is for individual HTTP calls; the long polling timeout lives
    # inside `poll_job`. Multipart uploads of small PDBs (~MB) fit well under
    # the 120s connect+read budget.
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "ppiflow"
    assert "version" in body


def test_healthz_detail(client: httpx.Client) -> None:
    r = client.get("/healthz/detail")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "ppiflow"
    assert body["jobs_base_dir_exists"] is True
    assert body["disk_usage_mb"] >= 0
    assert body["disk_limit_mb"] >= 100


def test_manifest_lists_all_five_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    body = r.json()
    paths = {e["path"] for e in body["endpoints"]}
    assert paths == {
        "/api/sample/binder",
        "/api/sample/antibody",
        "/api/sample/nanobody",
        "/api/sample/monomer",
        "/api/sample/scaffolding",
    }


def test_openapi_served(client: httpx.Client) -> None:
    r = client.get("/openapi.json")
    r.raise_for_status()
    schema = r.json()
    assert schema["info"]["title"]  # FastAPI fills this from create_app(title=...)


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    r = client.get("/api/jobs/this-id-does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Inference — one minimal job per endpoint.
# ---------------------------------------------------------------------------


def _assert_completed(job: dict, base_url: str, client: httpx.Client) -> None:
    """Common terminal-state asserts: completed + at least one output file."""
    assert job["status"] == "completed", (
        f"job failed: kind={job.get('failure_kind')} "
        f"summary={job.get('error_summary')!r}"
    )
    files_resp = client.get(f"/api/jobs/{job['job_id']}/files")
    files_resp.raise_for_status()
    files = files_resp.json()["files"]
    assert len(files) >= 1, f"no output files for job {job['job_id']!r}: {files}"


def test_monomer_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Smallest possible monomer call: one length, 2 samples."""
    r = client.post(
        "/api/sample/monomer",
        data={
            "length_subset": "[60]",
            "samples_per_target": "2",
            "name": "fc_smoke_monomer",
        },
    )
    r.raise_for_status()
    job_id = r.json()["job_id"]
    final = poll_job(client, base_url, job_id)
    _assert_completed(final, base_url, client)


def test_binder_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Smallest viable binder job — 2 samples in a tight length window."""
    with open(ANTIGEN_PDB, "rb") as fh:
        r = client.post(
            "/api/sample/binder",
            files={"target": (ANTIGEN_PDB.name, fh, "chemical/x-pdb")},
            data={
                "target_chain": "C",
                "binder_chain": "A",
                "specified_hotspots": "C11,C14,C15",
                "samples_min_length": "60",
                "samples_max_length": "70",
                "samples_per_target": "2",
                "name": "fc_smoke_binder",
            },
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _assert_completed(final, base_url, client)


def test_antibody_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Tightest CDR length spec (single-length per CDR), 2 samples."""
    with open(ANTIGEN_PDB, "rb") as ag, open(SCFV_FRAMEWORK_PDB, "rb") as fw:
        r = client.post(
            "/api/sample/antibody",
            files={
                "antigen": (ANTIGEN_PDB.name, ag, "chemical/x-pdb"),
                "framework": (SCFV_FRAMEWORK_PDB.name, fw, "chemical/x-pdb"),
            },
            data={
                "antigen_chain": "C",
                "heavy_chain": "A",
                "light_chain": "B",
                "specified_hotspots": "C11,C14,C15",
                "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,10-10,CDRL1,7-7,CDRL2,3-3,CDRL3,9-9",
                "samples_per_target": "2",
                "name": "fc_smoke_antibody",
            },
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _assert_completed(final, base_url, client)


def test_nanobody_minimal_job(client: httpx.Client, base_url: str) -> None:
    """Heavy-only CDR design; tight length spec + 2 samples."""
    with open(ANTIGEN_PDB, "rb") as ag, open(NANOBODY_FRAMEWORK_PDB, "rb") as fw:
        r = client.post(
            "/api/sample/nanobody",
            files={
                "antigen": (ANTIGEN_PDB.name, ag, "chemical/x-pdb"),
                "framework": (NANOBODY_FRAMEWORK_PDB.name, fw, "chemical/x-pdb"),
            },
            data={
                "antigen_chain": "C",
                "heavy_chain": "A",
                "specified_hotspots": "C11,C14,C15",
                "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,10-10",
                "samples_per_target": "2",
                "name": "fc_smoke_nanobody",
            },
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    _assert_completed(final, base_url, client)


@pytest.mark.skip(
    reason=(
        "Scaffolding requires the motif CSV's `motif_path` column to point at "
        "PDB files reachable on the FC NAS — we can't stage those from the test "
        "runner. Enable once we have a fixture for shipping motif PDB(s) via "
        "OSS / file:// URIs."
    )
)
def test_scaffolding_minimal_job(client: httpx.Client, base_url: str) -> None:
    raise NotImplementedError

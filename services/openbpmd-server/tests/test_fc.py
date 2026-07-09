"""FC integration tests for openbpmd-server (opt-in, sync submit/poll).

Marked ``@pytest.mark.fc``, skipped by default.  Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/openbpmd-server/tests/test_fc.py -v

Fixtures
--------
The Amber system (solvated.rst7 + solvated.prm7, ~10 MB) is NOT committed —
it is resolved from the local ``opensource/OpenBPMD/tests/files/`` clone, or
override with ``OPENBPMD_TEST_STRUCTURE`` / ``OPENBPMD_TEST_PARAMETERS``.
The bundled system's ligand residue name is ``UNK``.

Runtime
-------
A full 10 rep x 10 ns run is hours.  This regression uses the advanced
``sim_ns`` / ``equil_steps`` knobs to run a SHORT trajectory (nreps=1,
sim_ns=0.02, equil_steps=500) so the whole pipeline validates in a few
minutes on a GPU.  The resulting scores are NOT physically meaningful — the
test only asserts the pipeline is wired up and results.csv is produced.
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "openbpmd-server"
SESSION_HEADER = "bioagent-session-id"

_DEFAULT_CLONE = (
    Path(__file__).resolve().parents[3]
    / "opensource" / "OpenBPMD" / "tests" / "files"
)
TEST_STRUCTURE = Path(
    os.environ.get("OPENBPMD_TEST_STRUCTURE", _DEFAULT_CLONE / "solvated.rst7")
)
TEST_PARAMETERS = Path(
    os.environ.get("OPENBPMD_TEST_PARAMETERS", _DEFAULT_CLONE / "solvated.prm7")
)
LIG_RESNAME = os.environ.get("OPENBPMD_TEST_LIG_RESNAME", "UNK")

TIMEOUT = httpx.Timeout(connect=30, read=600, write=600, pool=30)
POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 20

_fixtures_present = TEST_STRUCTURE.exists() and TEST_PARAMETERS.exists()
_needs_fixtures = pytest.mark.skipif(
    not _fixtures_present,
    reason=f"fixtures missing: {TEST_STRUCTURE} / {TEST_PARAMETERS}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture(scope="module")
def session_headers() -> dict[str, str]:
    return {SESSION_HEADER: f"test-{uuid.uuid4().hex[:12]}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_with_retry(
    call: Callable[[], httpx.Response], *, max_attempts: int = 20, backoff_s: int = 30,
) -> httpx.Response:
    """Retry FC's 429 ResourceExhausted (max_concurrent_jobs=1)."""
    last: httpx.Response | None = None
    for _attempt in range(max_attempts):
        last = call()
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


def _retry_get(client: httpx.Client, path: str, **kw: Any) -> httpx.Response:
    return _http_with_retry(lambda: client.get(path, **kw))


def _retry_post(client: httpx.Client, path: str, **kw: Any) -> httpx.Response:
    return _http_with_retry(lambda: client.post(path, **kw))


def _save_job_outputs(client, job_id, job_info, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "jobinfo.json").write_text(json.dumps(job_info, indent=2))
    try:
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        if r.status_code == 200 and r.content:
            (dst_dir / f"{job_id}.zip").write_bytes(r.content)
    except Exception as exc:  # noqa: BLE001
        print(f"[fc_outputs] zip download failed: {exc!r}")


# ---------------------------------------------------------------------------
# Module-scoped inference fixture — one short scoring run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def score_job(client, session_headers, local_output_dir: Path) -> dict:
    with open(TEST_STRUCTURE, "rb") as fh_s, open(TEST_PARAMETERS, "rb") as fh_p:
        r = _retry_post(
            client, "/api/score",
            files={
                "structure": ("solvated.rst7", fh_s.read(), "application/octet-stream"),
                "parameters": ("solvated.prm7", fh_p.read(), "application/octet-stream"),
            },
            data={
                "lig_resname": LIG_RESNAME,
                "nreps": "1",
                "hill_height": "0.3",
                # Short trajectory so the FC regression is fast.
                "sim_ns": "0.02",
                "equil_steps": "500",
            },
            headers=session_headers,
        )
    assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
    job_id = r.json()["job_id"]

    final = poll_job(
        client, "", job_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        max_transient_errors=60, extra_headers=session_headers,
    )
    _save_job_outputs(client, job_id, final, local_output_dir / "score")
    assert final["status"] == "completed", (
        f"failed: kind={final.get('failure_kind')} "
        f"summary={final.get('error_summary')!r}"
    )
    return final


# ===================================================================
# Section 1: Smoke (no compute)
# ===================================================================


@pytest.mark.fc
class TestSmoke:
    def test_healthz(self, client):
        body = _retry_get(client, "/healthz").json()
        assert body["status"] == "ok"
        assert body["service"] == "openbpmd"

    def test_healthz_detail(self, client):
        body = _retry_get(client, "/healthz/detail").json()
        assert body["status"] == "ok"
        assert body["cuda_available"] is True, (
            f"CUDA platform not available on FC GPU: {body.get('platforms')}"
        )
        assert body["openmm_version"] is not None

    def test_openapi_served(self, client):
        paths = _retry_get(client, "/openapi.json").json()["paths"]
        assert "/api/score" in paths
        assert "/api/tasks/score" in paths


# ===================================================================
# Section 2: Errors (fast, no GPU)
# ===================================================================


@pytest.mark.fc
class TestErrors:
    def test_unknown_job_status_404(self, client):
        assert _retry_get(client, "/api/jobs/missing-id").status_code == 404

    def test_score_rejects_missing_inputs(self, client):
        r = _retry_post(client, "/api/score", data={"lig_resname": "MOL"})
        assert r.status_code in (400, 422)


# ===================================================================
# Section 3: Sync inference (short trajectory)
# ===================================================================


@pytest.mark.fc
@_needs_fixtures
class TestSyncScore:
    def test_job_completed(self, score_job):
        assert score_job["status"] == "completed"
        assert score_job.get("duration_seconds", 0) > 0

    def test_input_params_echo(self, score_job):
        params = score_job.get("input_params") or {}
        assert params.get("nreps") == 1
        assert params.get("lig_resname") == LIG_RESNAME

    def test_results_files_present(self, client, score_job):
        job_id = score_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        assert any(f == "results.csv" for f in files), f"got: {files}"
        assert any(f == "scoring_stats.json" for f in files)

    def test_scoring_stats_valid(self, client, score_job):
        job_id = score_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/file/scoring_stats.json")
        assert r.status_code == 200
        stats = json.loads(r.content)
        assert stats["nreps_done"] == 1
        assert stats["results_written"] is True
        assert "comp_score" in stats


# ===================================================================
# Section 4: Job lifecycle
# ===================================================================


@pytest.mark.fc
@_needs_fixtures
class TestJobLifecycle:
    def test_status_endpoint(self, client, score_job):
        job_id = score_job["job_id"]
        assert _retry_get(client, f"/api/jobs/{job_id}").json()["status"] == "completed"

    def test_download_zip(self, client, score_job):
        job_id = score_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any("results.csv" in n for n in zf.namelist())

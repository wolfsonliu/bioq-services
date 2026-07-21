"""FC integration tests for diffdock-pp-server (opt-in, sync submit/poll).

Marked ``@pytest.mark.fc``, skipped by default.  Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/diffdock-pp-server/tests/test_fc.py -v

Fixtures (1a2k_receptor.pdb + 1a2k_ligand.pdb) ship in tests/data/ —
copied from upstream ``datasets/single_pair_dataset/structures/`` (MIT).

Uses ``num_samples=4, top_k=2`` to keep the FC regression fast (~2-3 min
on H20 / T4). Higher counts follow the same code path — the smoke test
just validates the pipeline is wired up.
"""

from __future__ import annotations

import io
import json
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "diffdock-pp-server"
SESSION_HEADER = "bioagent-session-id"

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_RECEPTOR = DATA_DIR / "1a2k_receptor.pdb"
TEST_LIGAND = DATA_DIR / "1a2k_ligand.pdb"

TIMEOUT = httpx.Timeout(connect=30, read=600, write=600, pool=30)

# 40 diffusion steps × ~1-3s per step × N samples + ESM embed load.
# Small counts (4) finish in ~2-3 min on H20. Buffer for cold start.
POLL_TIMEOUT_S = 1200
POLL_INTERVAL_S = 20


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
    """Session affinity header so all polls hit the same FC instance."""
    return {SESSION_HEADER: f"test-{uuid.uuid4().hex[:12]}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_with_retry(
    call: Callable[[], httpx.Response],
    *,
    max_attempts: int = 20,
    backoff_s: int = 30,
) -> httpx.Response:
    """Retry FC's 429 ResourceExhausted.

    diffdock-pp-server runs with ``max_concurrent_jobs=1`` so even
    sequential GETs occasionally trip the 429 gateway throttle.
    """
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


def _assert_submitted(body: dict) -> str:
    assert "job_id" in body
    assert body["status"] in ("pending", "running")
    return body["job_id"]


def _assert_completed(body: dict) -> None:
    assert body["status"] == "completed", (
        f"failed: kind={body.get('failure_kind')} "
        f"summary={body.get('error_summary')!r}"
    )
    assert body.get("duration_seconds") is not None
    assert body["duration_seconds"] > 0
    assert body.get("output_count", 0) > 0


def _save_job_outputs(
    client: httpx.Client, job_id: str, job_info: dict, dst_dir: Path,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "jobinfo.json").write_text(json.dumps(job_info, indent=2))
    try:
        r = _retry_get(client, f"/api/jobs/{job_id}/log")
        if r.status_code == 200:
            body = r.json()
            (dst_dir / "subprocess.log").write_text(
                body.get("log") or body.get("text") or ""
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[fc_outputs] log download failed: {exc!r}")
    try:
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        if r.status_code == 200 and r.content:
            (dst_dir / f"{job_id}.zip").write_bytes(r.content)
            extract_to = dst_dir / "extracted"
            extract_to.mkdir(exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                zf.extractall(extract_to)
    except Exception as exc:  # noqa: BLE001
        print(f"[fc_outputs] zip download failed: {exc!r}")


# ---------------------------------------------------------------------------
# Module-scoped inference fixture — one dock run per session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dock_job(
    client: httpx.Client,
    session_headers: dict[str, str],
    local_output_dir: Path,
) -> dict:
    """One full docking pass on 1A2K — 4 samples, top-2, with confidence."""
    with open(TEST_RECEPTOR, "rb") as fh_r, open(TEST_LIGAND, "rb") as fh_l:
        r = _retry_post(
            client, "/api/dock",
            files={
                "receptor": ("1a2k_receptor.pdb", fh_r.read(), "chemical/x-pdb"),
                "ligand": ("1a2k_ligand.pdb", fh_l.read(), "chemical/x-pdb"),
            },
            data={
                "num_samples": "4",
                "top_k": "2",
                "use_confidence_model": "true",
                "seed": "42",
            },
            headers=session_headers,
        )
    assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
    job_id = _assert_submitted(r.json())

    final = poll_job(
        client, "", job_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        max_transient_errors=60,
        extra_headers=session_headers,
    )
    _save_job_outputs(client, job_id, final, local_output_dir / "dock")
    _assert_completed(final)
    return final


# ===================================================================
# Section 1: Smoke (no inference compute)
# ===================================================================


@pytest.mark.fc
class TestSmoke:
    def test_healthz(self, client):
        r = _retry_get(client, "/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "diffdock-pp"
        assert "version" in body

    def test_healthz_detail(self, client):
        r = _retry_get(client, "/healthz/detail")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "diffdock-pp"
        assert body["weights_dir"] == "/data/models/diffdock-pp"
        assert body["weights_loaded"] is True, (
            f"NAS weights missing: {body.get('weights_missing')}. "
            f"Rsync score + confidence checkpoints + esm_cache under "
            f"{body['weights_dir']}/."
        )
        assert body.get("weights_missing") == {}

    def test_openapi_served(self, client):
        r = _retry_get(client, "/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        assert "/api/dock" in paths
        assert "/api/tasks/dock" in paths


# ===================================================================
# Section 2: Errors (fast, no GPU)
# ===================================================================


@pytest.mark.fc
class TestErrors:
    def test_unknown_job_status_404(self, client):
        assert _retry_get(client, "/api/jobs/missing-job-id").status_code == 404

    def test_dock_rejects_missing_inputs(self, client):
        r = _retry_post(client, "/api/dock", data={"num_samples": "3"})
        assert r.status_code in (400, 422)

    def test_dock_rejects_num_samples_out_of_range(self, client):
        r = _retry_post(
            client, "/api/dock",
            data={
                "receptor_uri": "file:///nonexistent.pdb",
                "ligand_uri": "file:///nonexistent.pdb",
                "num_samples": "0",
            },
        )
        assert r.status_code == 422


# ===================================================================
# Section 3: Sync inference
# ===================================================================


@pytest.mark.fc
class TestSyncDock:
    def test_job_completed(self, dock_job):
        assert dock_job["status"] == "completed"

    def test_input_params_echo(self, dock_job):
        params = dock_job.get("input_params") or {}
        assert params.get("num_samples") == 4
        assert params.get("top_k") == 2
        assert params.get("use_confidence_model") is True

    def test_dock_pose_files_present(self, client, dock_job):
        job_id = dock_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        # top_k=2 → dock_pose_1.pdb + dock_pose_2.pdb + confidence_scores.json + raw_samples.pkl
        pdb_files = [f for f in files if f.startswith("dock_pose_") and f.endswith(".pdb")]
        assert len(pdb_files) == 2, f"expected 2 dock_pose_*.pdb, got: {files}"
        assert any(f == "confidence_scores.json" for f in files)
        assert any(f == "raw_samples.pkl" for f in files)

    def test_confidence_scores_json_valid(self, client, dock_job):
        job_id = dock_job["job_id"]
        r = _retry_get(
            client, f"/api/jobs/{job_id}/file/confidence_scores.json"
        )
        assert r.status_code == 200
        scores = json.loads(r.content)
        assert isinstance(scores, list)
        assert len(scores) == 2  # top_k=2
        assert scores[0]["rank"] == 1
        assert scores[0]["sample_file"] == "dock_pose_1.pdb"

    def test_dock_pose_pdb_has_receptor_and_ligand(self, client, dock_job):
        """Sanity-check the assembled PDB actually contains both chains."""
        job_id = dock_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/file/dock_pose_1.pdb")
        assert r.status_code == 200
        text = r.content.decode()
        # Our wrapper writes receptor as chain R, ligand as chain L.
        assert " R" in text and " L" in text, (
            "dock_pose_1.pdb should contain both R (receptor) and L (ligand) chains"
        )
        assert "TER" in text
        assert "END" in text


# ===================================================================
# Section 4: Job lifecycle
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, dock_job):
        job_id = dock_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}")
        assert r.status_code == 200
        assert r.json()["status"] == "completed"

    def test_job_log_endpoint(self, client, dock_job):
        job_id = dock_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert len(log_text) > 0

    def test_job_download_zip(self, client, dock_job):
        job_id = dock_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any(n.startswith("dock_pose_") and n.endswith(".pdb") for n in zf.namelist())

"""FC integration tests for pocketxmol-server (opt-in, sync submit/poll).

Marked ``@pytest.mark.fc``, skipped by default.  Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/pocketxmol-server/tests/test_fc.py -v

Fixtures ship in tests/data/ — small excerpt of upstream data/examples/.
PocketXMol runs 100 denoising steps per sample; small batches finish in
tens of seconds on T4/A10, minutes for pep design.

max_concurrent_jobs=1 → HTTP gateway 429s under any parallel work; every
GET/POST goes through _http_with_retry to absorb them.
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

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "pocketxmol-server"
SESSION_HEADER = "bioagent-session-id"

DATA_DIR = Path(__file__).resolve().parent / "data"
TIMEOUT = httpx.Timeout(connect=30, read=600, write=600, pool=30)

# PocketXMol diffusion sampling ~ 100 steps × N mols; on T4 typical
# job = 1-5 min, cold start + weight load can push to ~10.
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
    return {SESSION_HEADER: f"test-{uuid.uuid4().hex[:12]}"}


# ---------------------------------------------------------------------------
# Retry helpers (max_concurrent_jobs=1 → 429s common)
# ---------------------------------------------------------------------------
def _http_with_retry(
    call: Callable[[], httpx.Response],
    *,
    max_attempts: int = 20,
    backoff_s: int = 30,
) -> httpx.Response:
    last: httpx.Response | None = None
    for _ in range(max_attempts):
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


def _assert_completed(body: dict) -> None:
    assert body["status"] == "completed", (
        f"failed: kind={body.get('failure_kind')} "
        f"summary={body.get('error_summary')!r}"
    )
    assert body.get("duration_seconds") is not None
    assert body["duration_seconds"] > 0
    assert body.get("output_count", 0) > 0


def _save_job_outputs(
    client: httpx.Client, job_id: str, info: dict, dst: Path,
) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "jobinfo.json").write_text(json.dumps(info, indent=2))
    try:
        r = _retry_get(client, f"/api/jobs/{job_id}/log")
        if r.status_code == 200:
            body = r.json()
            (dst / "subprocess.log").write_text(
                body.get("log") or body.get("text") or ""
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[fc_outputs] log download failed: {exc!r}")
    try:
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        if r.status_code == 200 and r.content:
            (dst / f"{job_id}.zip").write_bytes(r.content)
            (dst / "extracted").mkdir(exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                zf.extractall(dst / "extracted")
    except Exception as exc:  # noqa: BLE001
        print(f"[fc_outputs] zip download failed: {exc!r}")


def _submit_and_wait(
    client: httpx.Client, session_headers: dict[str, str],
    local_output_dir: Path, endpoint: str, label: str,
    *, files: dict | None = None, data: dict | None = None,
) -> dict:
    r = _retry_post(
        client, endpoint, files=files, data=data, headers=session_headers,
    )
    assert r.status_code == 200, f"submit {endpoint}: {r.status_code} {r.text!r}"
    body = r.json()
    job_id = body["job_id"]
    final = poll_job(
        client, "", job_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        max_transient_errors=60, extra_headers=session_headers,
    )
    _save_job_outputs(client, job_id, final, local_output_dir / label)
    _assert_completed(final)
    return final


pytestmark = pytest.mark.fc


# ===========================================================================
# Section 1: Smoke
# ===========================================================================
class TestSmoke:
    def test_healthz(self, client):
        r = _retry_get(client, "/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "pocketxmol"

    def test_healthz_detail_weights_loaded(self, client):
        r = _retry_get(client, "/healthz/detail")
        assert r.status_code == 200
        body = r.json()
        assert body["weights_loaded"] is True, (
            f"NAS weights missing: {body.get('weights_missing')}"
        )

    def test_manifest_has_six_endpoints(self, client):
        r = _retry_get(client, "/api/manifest")
        assert r.status_code == 200
        paths = {e["path"] for e in r.json()["endpoints"]}
        for p in ("/api/dock", "/api/sbdd", "/api/linking", "/api/optimize",
                  "/api/pepdesign", "/api/confidence"):
            assert p in paths, f"manifest missing {p}"

    def test_openapi_served(self, client):
        r = _retry_get(client, "/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        for p in ("/api/dock", "/api/tasks/dock"):
            assert p in paths, f"openapi missing {p}"

    def test_unknown_job_404(self, client):
        assert _retry_get(client, "/api/jobs/nonexistent-job-id").status_code == 404


# ===========================================================================
# Section 2: Endpoint inference — minimal jobs per family
# ===========================================================================
@pytest.fixture(scope="module")
def dock_job(client, session_headers, local_output_dir):
    """Dock a small mol against 8C7Y — 3 samples."""
    with open(DATA_DIR / "8C7Y_TXV_protein.pdb", "rb") as fp, \
            open(DATA_DIR / "8C7Y_TXV_ligand_start_conf.sdf", "rb") as fl:
        return _submit_and_wait(
            client, session_headers, local_output_dir,
            "/api/dock", "dock",
            files={
                "protein": ("protein.pdb", fp.read(), "chemical/x-pdb"),
                "ligand": ("ligand.sdf", fl.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "num_samples": "3",
                "batch_size": "3",
                "pocket_coord": "[-8.257, 85.181, 19.050]",
                "pocket_radius": "15",
            },
        )


class TestDock:
    def test_completed_with_sdf(self, dock_job, client):
        job_id = dock_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/files")
        assert r.status_code == 200
        files = r.json()["files"]
        assert any(f.endswith(".sdf") for f in files), files

    def test_input_params_echoed(self, dock_job):
        p = dock_job.get("input_params") or {}
        assert p.get("num_samples") == 3


@pytest.fixture(scope="module")
def sbdd_job(client, session_headers, local_output_dir):
    """De novo SBDD on 2ar9 pocket — 5 samples, simple mode (fastest)."""
    with open(DATA_DIR / "2ar9_A.pdb", "rb") as fp:
        return _submit_and_wait(
            client, session_headers, local_output_dir,
            "/api/sbdd", "sbdd",
            files={"protein": ("protein.pdb", fp.read(), "chemical/x-pdb")},
            data={
                "num_samples": "5",
                "batch_size": "5",
                "pocket_coord": "[-8.1603, 36.6972, 38.7714]",
                "pocket_radius": "15",
                "mode": "simple",
            },
        )


class TestSbdd:
    def test_completed_with_sdf(self, sbdd_job, client):
        job_id = sbdd_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        assert any(f.endswith(".sdf") for f in files), files


@pytest.fixture(scope="module")
def linking_job(client, session_headers, local_output_dir):
    """Fragment growing on 2ar9 with 1-group fragment."""
    with open(DATA_DIR / "2ar9_A.pdb", "rb") as fp, \
            open(DATA_DIR / "fragment.sdf", "rb") as fl:
        return _submit_and_wait(
            client, session_headers, local_output_dir,
            "/api/linking", "linking",
            files={
                "protein": ("protein.pdb", fp.read(), "chemical/x-pdb"),
                "input_ligand": ("frag.sdf", fl.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "num_samples": "3", "batch_size": "3",
                "fragments": "[[0,1,2,3,4,5,6]]",
                "mol_size_mean": "28",
            },
        )


class TestLinking:
    def test_completed_with_sdf(self, linking_job, client):
        job_id = linking_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        assert any(f.endswith(".sdf") for f in files), files


@pytest.fixture(scope="module")
def pepdesign_job(client, session_headers, local_output_dir):
    """De novo linear peptide of length 5 targeting 3bik pocket."""
    with open(DATA_DIR / "3bik_A.pdb", "rb") as fp, \
            open(DATA_DIR / "3bik_A_pocket_coord.sdf", "rb") as fr:
        return _submit_and_wait(
            client, session_headers, local_output_dir,
            "/api/pepdesign", "pepdesign",
            files={
                "protein": ("protein.pdb", fp.read(), "chemical/x-pdb"),
                "ref_ligand": ("ref.sdf", fr.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "mode": "denovo_linear",
                "pep_length": "5",
                "num_samples": "3",
                "batch_size": "3",
                "pocket_radius": "20",
            },
        )


class TestPepDesign:
    def test_completed_with_pdb_or_sdf(self, pepdesign_job, client):
        job_id = pepdesign_job["job_id"]
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        # Peptide runs emit both *_mol.sdf and *.pdb.
        assert any(f.endswith(".pdb") or f.endswith(".sdf") for f in files), files


class TestConfidenceChained:
    """Confidence scores a prior sbdd job — requires sbdd_job fixture to
    have populated /data on the FC instance.  Session affinity ensures the
    same instance serves both jobs.
    """

    def test_confidence_completes(
        self, sbdd_job, client, session_headers, local_output_dir,
    ):
        source_job_id = sbdd_job["job_id"]
        r = _retry_post(
            client, "/api/confidence",
            data={"source_job_id": source_job_id, "variant": "tuned_cfd"},
            headers=session_headers,
        )
        assert r.status_code == 200, f"confidence submit: {r.status_code} {r.text!r}"
        job_id = r.json()["job_id"]
        final = poll_job(
            client, "", job_id,
            timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
            max_transient_errors=60, extra_headers=session_headers,
        )
        _save_job_outputs(client, job_id, final, local_output_dir / "confidence")
        _assert_completed(final)


# ===========================================================================
# Section 3: Job lifecycle (uses the cheapest fixture available)
# ===========================================================================
class TestJobLifecycle:
    def test_status_endpoint(self, dock_job, client):
        job_id = dock_job["job_id"]
        body = _retry_get(client, f"/api/jobs/{job_id}").json()
        assert body["status"] == "completed"

    def test_log_endpoint(self, dock_job, client):
        job_id = dock_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/log")
        assert r.status_code == 200

    def test_download_zip(self, dock_job, client):
        job_id = dock_job["job_id"]
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        assert r.content
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any(n.endswith(".sdf") for n in zf.namelist()), zf.namelist()

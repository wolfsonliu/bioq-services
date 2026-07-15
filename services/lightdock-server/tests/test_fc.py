"""FC integration tests for lightdock-server (opt-in, sync submit/poll).

Marked ``@pytest.mark.fc``, skipped by default. Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/lightdock-server/tests/test_fc.py -v

Fixtures ship in tests/data/ (2UUY receptor/ligand from LightDock's own test
data) so the suite is self-contained. Sampling is kept tiny (swarms=2,
glowworms=5, steps=3) so a docking run finishes in minutes, not hours.
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

SERVICE = "lightdock-server"
SESSION_HEADER = "bioagent-session-id"

DATA_DIR = Path(__file__).resolve().parent / "data"
RECEPTOR = DATA_DIR / "receptor.pdb"
LIGAND = DATA_DIR / "ligand.pdb"

TIMEOUT = httpx.Timeout(connect=30, read=600, write=600, pool=30)
POLL_TIMEOUT_S = 3600
POLL_INTERVAL_S = 20

# Tiny sampling so the GSO run finishes quickly.
TINY = {"swarms": "2", "glowworms": "5", "steps": "3", "top": "3"}


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


def _http_with_retry(
    call: Callable[[], httpx.Response], *, max_attempts: int = 20, backoff_s: int = 30,
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


def _save_job_outputs(client, job_id, job_info, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "jobinfo.json").write_text(json.dumps(job_info, indent=2))
    try:
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        if r.status_code == 200 and r.content:
            (dst_dir / f"{job_id}.zip").write_bytes(r.content)
    except Exception as exc:  # noqa: BLE001
        print(f"[fc_outputs] zip download failed: {exc!r}")


# ===================================================================
# Section 1: Smoke (no compute)
# ===================================================================


@pytest.mark.fc
class TestSmoke:
    def test_healthz(self, client):
        body = _retry_get(client, "/healthz").json()
        assert body["status"] == "ok"
        assert body["service"] == "lightdock"

    def test_healthz_detail(self, client):
        body = _retry_get(client, "/healthz/detail").json()
        assert body["status"] == "ok"
        assert body["lightdock_available"] is True
        assert body["lightdock_version"]
        assert "fastdfire" in body["scoring_functions"]

    def test_openapi_served(self, client):
        paths = _retry_get(client, "/openapi.json").json()["paths"]
        assert "/api/dock" in paths
        assert "/api/tasks/dock" in paths

    def test_unknown_job_404(self, client):
        assert _retry_get(client, "/api/jobs/missing-id").status_code == 404


# ===================================================================
# Section 2: docking inference
# ===================================================================


@pytest.mark.fc
class TestDocking:
    def test_dock(self, client, session_headers, local_output_dir):
        with open(RECEPTOR, "rb") as fr, open(LIGAND, "rb") as fl:
            r = _retry_post(
                client, "/api/dock",
                files={
                    "receptor": ("receptor.pdb", fr.read(), "chemical/x-pdb"),
                    "ligand": ("ligand.pdb", fl.read(), "chemical/x-pdb"),
                },
                data=TINY, headers=session_headers,
            )
        assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
        job_id = r.json()["job_id"]
        final = poll_job(
            client, "", job_id,
            timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
            max_transient_errors=60, extra_headers=session_headers,
        )
        _save_job_outputs(client, job_id, final, local_output_dir / "dock")
        assert final["status"] == "completed", (
            f"dock failed: kind={final.get('failure_kind')} "
            f"summary={final.get('error_summary')!r}"
        )
        files = _retry_get(client, f"/api/jobs/{job_id}/files").json()["files"]
        assert any(f.endswith("top/top_1.pdb") or f.endswith("top_1.pdb") for f in files), files

    def test_download_zip(self, client, session_headers, local_output_dir):
        with open(RECEPTOR, "rb") as fr, open(LIGAND, "rb") as fl:
            r = _retry_post(
                client, "/api/dock",
                files={
                    "receptor": ("receptor.pdb", fr.read(), "chemical/x-pdb"),
                    "ligand": ("ligand.pdb", fl.read(), "chemical/x-pdb"),
                },
                data=TINY, headers=session_headers,
            )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        final = poll_job(
            client, "", job_id,
            timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
            max_transient_errors=60, extra_headers=session_headers,
        )
        assert final["status"] == "completed", final
        r = _retry_get(client, f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any("top_1.pdb" in n for n in zf.namelist()), zf.namelist()

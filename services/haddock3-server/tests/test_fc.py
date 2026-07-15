"""FC integration tests for haddock3-server (opt-in, sync submit/poll).

Marked ``@pytest.mark.fc``, skipped by default. Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/haddock3-server/tests/test_fc.py -v

Fixtures ship in tests/data/ (derived from HADDOCK3's own examples, Apache-2.0)
so the suite is self-contained.

CNS gating
----------
The restraints endpoints are CNS-free and always run. The docking + scoring
tests need a CNS binary staged at HADDOCK3_CNS_EXEC (surfaced by
/healthz/detail as ``cns_available``); they self-skip when it is absent.
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

SERVICE = "haddock3-server"
SESSION_HEADER = "bioagent-session-id"

DATA_DIR = Path(__file__).resolve().parent / "data"
COMPLEX_PDB = DATA_DIR / "complex.pdb"
MOL_A = DATA_DIR / "mol_A.pdb"
MOL_B = DATA_DIR / "mol_B.pdb"
AMBIG_TBL = DATA_DIR / "ambig.tbl"
ACTPASS_A = DATA_DIR / "a.actpass"
ACTPASS_B = DATA_DIR / "b.actpass"

TIMEOUT = httpx.Timeout(connect=30, read=600, write=600, pool=30)
POLL_TIMEOUT_S = 3600
POLL_INTERVAL_S = 20


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


@pytest.fixture(scope="module")
def cns_available(client) -> bool:
    body = _retry_get(client, "/healthz/detail").json()
    return bool(body.get("cns_available"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _run_to_completion(client, session_headers, path, files, data, label, local_output_dir):
    r = _retry_post(client, path, files=files, data=data, headers=session_headers)
    assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
    job_id = r.json()["job_id"]
    final = poll_job(
        client, "", job_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        max_transient_errors=60, extra_headers=session_headers,
    )
    _save_job_outputs(client, job_id, final, local_output_dir / label)
    assert final["status"] == "completed", (
        f"{label} failed: kind={final.get('failure_kind')} "
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
        assert body["service"] == "haddock3"

    def test_healthz_detail(self, client):
        body = _retry_get(client, "/healthz/detail").json()
        assert body["status"] == "ok"
        assert body["restraints_available"] is True
        assert "cns_available" in body
        assert body["haddock3_version"]

    def test_openapi_served(self, client):
        paths = _retry_get(client, "/openapi.json").json()["paths"]
        for p in ("/api/dock", "/api/dock/protein-protein", "/api/score",
                  "/api/restraints/restrain-bodies",
                  "/api/restraints/active-passive-to-ambig"):
            assert p in paths

    def test_unknown_job_404(self, client):
        assert _retry_get(client, "/api/jobs/missing-id").status_code == 404


# ===================================================================
# Section 2: CNS-free restraints inference (always runs)
# ===================================================================


@pytest.mark.fc
class TestRestraints:
    def test_restrain_bodies(self, client, session_headers, local_output_dir):
        with open(COMPLEX_PDB, "rb") as fh:
            final = _run_to_completion(
                client, session_headers, "/api/restraints/restrain-bodies",
                files={"structure": ("complex.pdb", fh.read(), "chemical/x-pdb")},
                data={}, label="restrain-bodies", local_output_dir=local_output_dir,
            )
        files = _retry_get(client, f"/api/jobs/{final['job_id']}/files").json()["files"]
        assert any(f.endswith("restraints.tbl") for f in files), files
        r = _retry_get(client, f"/api/jobs/{final['job_id']}/file/restraints.tbl")
        assert r.status_code == 200 and b"assign" in r.content.lower()

    def test_active_passive_to_ambig(self, client, session_headers, local_output_dir):
        with open(ACTPASS_A, "rb") as fa, open(ACTPASS_B, "rb") as fb:
            final = _run_to_completion(
                client, session_headers, "/api/restraints/active-passive-to-ambig",
                files={
                    "actpass1": ("a.actpass", fa.read(), "text/plain"),
                    "actpass2": ("b.actpass", fb.read(), "text/plain"),
                },
                data={"segid1": "A", "segid2": "B"},
                label="actpass-to-ambig", local_output_dir=local_output_dir,
            )
        files = _retry_get(client, f"/api/jobs/{final['job_id']}/files").json()["files"]
        assert any(f.endswith("ambig.tbl") for f in files), files


# ===================================================================
# Section 3: CNS-gated scoring + docking (self-skip without CNS)
# ===================================================================


@pytest.mark.fc
class TestScoring:
    def test_score(self, client, session_headers, cns_available, local_output_dir):
        if not cns_available:
            pytest.skip("CNS not staged (healthz cns_available=false)")
        with open(COMPLEX_PDB, "rb") as fh:
            final = _run_to_completion(
                client, session_headers, "/api/score",
                files={"complex": ("complex.pdb", fh.read(), "chemical/x-pdb")},
                data={"full": "true"}, label="score", local_output_dir=local_output_dir,
            )
        r = _retry_get(client, f"/api/jobs/{final['job_id']}/file/score.json")
        assert r.status_code == 200
        assert "haddock_score" in json.loads(r.content)


@pytest.mark.fc
class TestDocking:
    def test_protein_protein(self, client, session_headers, cns_available, local_output_dir):
        if not cns_available:
            pytest.skip("CNS not staged (healthz cns_available=false)")
        with open(MOL_A, "rb") as fa, open(MOL_B, "rb") as fb, open(AMBIG_TBL, "rb") as ft:
            final = _run_to_completion(
                client, session_headers, "/api/dock/protein-protein",
                files={
                    "mol1": ("mol_A.pdb", fa.read(), "chemical/x-pdb"),
                    "mol2": ("mol_B.pdb", fb.read(), "chemical/x-pdb"),
                    "ambig": ("ambig.tbl", ft.read(), "text/plain"),
                },
                # Tiny sampling so the CNS run finishes in minutes, not hours.
                data={"sampling": "4", "do_flexref": "false", "do_emref": "false",
                      "clustering": "false", "top_models": "2"},
                label="protein-protein", local_output_dir=local_output_dir,
            )
        files = _retry_get(client, f"/api/jobs/{final['job_id']}/files").json()["files"]
        assert any("caprieval" in f for f in files), files

    def test_download_zip(self, client, session_headers, local_output_dir):
        # Reuse the CNS-free restraints job so this lifecycle check always runs.
        with open(COMPLEX_PDB, "rb") as fh:
            final = _run_to_completion(
                client, session_headers, "/api/restraints/restrain-bodies",
                files={"structure": ("complex.pdb", fh.read(), "chemical/x-pdb")},
                data={}, label="restrain-bodies-zip", local_output_dir=local_output_dir,
            )
        r = _retry_get(client, f"/api/jobs/{final['job_id']}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any("restraints.tbl" in n for n in zf.namelist())

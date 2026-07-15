"""FC async task mode tests for haddock3-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/haddock3-server/tests/test_fc_task.py -v

Validates the /api/tasks/<name> endpoints end-to-end under FC async task mode
(``X-Fc-Invocation-Type: Async``). The CNS-free restraints task drives the
submit/lifecycle/dedup assertions; the CNS-gated docking task self-skips when no
CNS binary is staged.
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "haddock3-server"
DATA_DIR = Path(__file__).resolve().parent / "data"
COMPLEX_PDB = DATA_DIR / "complex.pdb"
MOL_A = DATA_DIR / "mol_A.pdb"
MOL_B = DATA_DIR / "mol_B.pdb"
AMBIG_TBL = DATA_DIR / "ambig.tbl"

POLL_TIMEOUT_S = 3600
POLL_INTERVAL_S = 20
TIMEOUT = httpx.Timeout(connect=30, read=300, write=600, pool=30)


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture(scope="module")
def restrain_task_id() -> str:
    return f"fc-async-restrain-{int(time.time())}-{uuid.uuid4().hex[:6]}"


def _async_headers(task_id: str) -> dict[str, str]:
    return {
        "X-Fc-Invocation-Type": "Async",
        "X-Bioagent-Job-Id": task_id,
        "X-Fc-Async-Task-Id": task_id,
    }


def _get_with_retry(client, path, *, max_attempts=20, backoff_s=30):
    last = None
    for _ in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    return last


def _poll(client, task_id: str) -> dict:
    final = poll_job(
        client, "", task_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S, max_transient_errors=60,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


@pytest.fixture(scope="module")
def restrain_submit(client, restrain_task_id):
    with open(COMPLEX_PDB, "rb") as fh:
        return client.post(
            "/api/tasks/restraints/restrain-bodies",
            files={"structure": ("complex.pdb", fh.read(), "chemical/x-pdb")},
            headers=_async_headers(restrain_task_id),
        )


@pytest.fixture(scope="module")
def restrain_task(client, restrain_task_id, restrain_submit) -> dict:
    assert restrain_submit.status_code == 202, (
        f"async submit returned {restrain_submit.status_code}: {restrain_submit.text!r}"
    )
    return _poll(client, restrain_task_id)


# ===================================================================
# Section 1: submit semantics + OpenAPI
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_returns_202(self, restrain_submit):
        assert restrain_submit.status_code == 202

    def test_task_endpoints_registered(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200
        expected = {
            "/api/tasks/dock",
            "/api/tasks/dock/protein-protein",
            "/api/tasks/score",
            "/api/tasks/restraints/restrain-bodies",
            "/api/tasks/restraints/active-passive-to-ambig",
        }
        missing = expected - set(r.json()["paths"])
        assert not missing, f"missing task endpoints: {missing}"


# ===================================================================
# Section 2: completion + output
# ===================================================================


@pytest.mark.fc
class TestAsyncRestrain:
    def test_completed(self, restrain_task, restrain_task_id):
        assert restrain_task["status"] == "completed"
        assert restrain_task["job_id"] == restrain_task_id
        assert restrain_task.get("output_count", 0) > 0

    def test_output_downloadable(self, client, restrain_task_id, restrain_task):
        files = _get_with_retry(
            client, f"/api/jobs/{restrain_task_id}/files"
        ).json()["files"]
        assert any(f.endswith("restraints.tbl") for f in files), files
        r = _get_with_retry(client, f"/api/jobs/{restrain_task_id}/file/restraints.tbl")
        assert r.status_code == 200 and len(r.content) > 100


# ===================================================================
# Section 3: lifecycle
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status(self, client, restrain_task_id, restrain_task):
        body = _get_with_retry(client, f"/api/jobs/{restrain_task_id}").json()
        assert body["status"] == "completed"

    def test_download_zip(self, client, restrain_task_id, restrain_task):
        r = _get_with_retry(client, f"/api/jobs/{restrain_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any("restraints.tbl" in n for n in zf.namelist())


# ===================================================================
# Section 4: platform-layer dedup
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    def test_duplicate_does_not_rerun(self, client, restrain_task_id, restrain_task):
        first_created = restrain_task["created_at"]
        first_completed = restrain_task["completed_at"]
        with open(COMPLEX_PDB, "rb") as fh:
            r2 = client.post(
                "/api/tasks/restraints/restrain-bodies",
                files={"structure": ("complex.pdb", fh.read(), "chemical/x-pdb")},
                data={"exclude": "Z"},  # deliberately different — must not take effect
                headers=_async_headers(restrain_task_id),
            )
        assert r2.status_code in (202, 409), f"got {r2.status_code} {r2.text!r}"
        if r2.status_code == 202:
            time.sleep(15)
        re_query = _get_with_retry(client, f"/api/jobs/{restrain_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created
        assert re_query["completed_at"] == first_completed


# ===================================================================
# Section 5: CNS-gated docking (self-skip without CNS)
# ===================================================================


@pytest.mark.fc
class TestAsyncDocking:
    def test_protein_protein(self, client):
        detail = _get_with_retry(client, "/healthz/detail").json()
        if not detail.get("cns_available"):
            pytest.skip("CNS not staged (healthz cns_available=false)")
        task_id = f"fc-async-pp-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        with open(MOL_A, "rb") as fa, open(MOL_B, "rb") as fb, open(AMBIG_TBL, "rb") as ft:
            r = client.post(
                "/api/tasks/dock/protein-protein",
                files={
                    "mol1": ("mol_A.pdb", fa.read(), "chemical/x-pdb"),
                    "mol2": ("mol_B.pdb", fb.read(), "chemical/x-pdb"),
                    "ambig": ("ambig.tbl", ft.read(), "text/plain"),
                },
                data={"sampling": "4", "do_flexref": "false", "do_emref": "false",
                      "clustering": "false", "top_models": "2"},
                headers=_async_headers(task_id),
            )
        assert r.status_code == 202, f"{r.status_code} {r.text!r}"
        final = _poll(client, task_id)
        files = _get_with_retry(client, f"/api/jobs/{task_id}/files").json()["files"]
        assert any("caprieval" in f for f in files), files
        assert final["status"] == "completed"

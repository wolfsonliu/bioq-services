"""FC integration tests for alphafold-server (opt-in).

Run only when FC_TEST_BASE_URL is set::

    FC_TEST_BASE_URL=https://xxx.cn-hangzhou.fcapp.run \
    pytest -m fc services/alphafold-server/tests/test_fc.py -v
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import pytest

BASE_URL = os.environ.get("FC_TEST_BASE_URL", "")
TIMEOUT = httpx.Timeout(connect=30, read=600, write=60, pool=30)

EXAMPLE_FASTA = """\
>test
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATG
"""


@pytest.fixture
def client():
    token = os.environ.get("FC_TEST_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(base_url=BASE_URL, headers=headers, timeout=TIMEOUT)


@pytest.mark.fc
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "alphafold"


@pytest.mark.fc
def test_manifest(client):
    r = client.get("/api/manifest")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "alphafold"
    assert any(e["path"] == "/api/fold" for e in body["endpoints"])


@pytest.mark.fc
def test_fold_monomer(client, tmp_path):
    fasta = tmp_path / "test.fasta"
    fasta.write_text(EXAMPLE_FASTA)

    with open(fasta, "rb") as f:
        r = client.post(
            "/api/fold",
            data={"model_preset": "monomer_ptm", "models_to_relax": "best"},
            files={"input_fasta": ("test.fasta", f, "text/plain")},
        )
    assert r.status_code == 200, r.text
    job = r.json()
    job_id = job["job_id"]

    for _ in range(360):
        time.sleep(10)
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        status = r.json()["status"]
        if status in ("completed", "failed"):
            break
    else:
        pytest.fail(f"Job {job_id} did not complete within 60 min")

    assert status == "completed", f"Job failed: {r.json()}"

    r = client.get(f"/api/jobs/{job_id}/files")
    assert r.status_code == 200
    files = r.json()["files"]
    filenames = {f["name"] for f in files}
    assert any("ranked_0.pdb" in n for n in filenames)

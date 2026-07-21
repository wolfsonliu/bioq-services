"""FC sync submit/poll integration tests for iggm-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/iggm-server/tests/test_fc.py -v

Inputs use file:// URIs pointing at the examples vendored into the image at
/opt/iggm/examples/ — this avoids uploading the ~265 KB antigen PDB and keeps
the test self-contained (no NAS fixture staging needed).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "iggm-server"

# In-image vendored example fixtures (see scripts/vendor.sh).
EX_DIR = "/opt/iggm/examples"
AB_FASTA = f"file://{EX_DIR}/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta"
COMPLEX_FASTA = f"file://{EX_DIR}/fasta.files.native/8hpu_M_N_A.fasta"
ANTIGEN_PDB = f"file://{EX_DIR}/pdb.files.native/8hpu_M_N_A.pdb"

POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 15
TIMEOUT = httpx.Timeout(connect=30, read=120, write=60, pool=30)


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


@pytest.mark.fc
class TestSmoke:
    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["service"] == "iggm"

    def test_healthz_detail_weights(self, client):
        r = client.get("/healthz/detail")
        assert r.status_code == 200
        body = r.json()
        assert body["weights_loaded"] is True, (
            f"checkpoints not all present on NAS: {body.get('weights_missing')}"
        )

    def test_manifest(self, client):
        r = client.get("/api/manifest")
        assert r.status_code == 200
        assert r.json()["service"] == "iggm"

    def test_openapi_paths(self, client):
        spec = client.get("/openapi.json").json()
        for p in ("/api/design", "/api/affinity-maturation", "/api/epitope"):
            assert p in spec["paths"], p


@pytest.mark.fc
class TestEpitope:
    """cal_ppi is CPU + fast; good sync smoke of the real pipeline."""

    def test_epitope_computes(self, client):
        r = client.post(
            "/api/epitope",
            data={"fasta_uri": COMPLEX_FASTA, "antigen_uri": ANTIGEN_PDB},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        final = poll_job(
            client, "", job_id, timeout_s=600, interval_s=10,
        )
        assert final["status"] == "completed", final
        f = client.get(f"/api/jobs/{job_id}/file/epitope.json")
        assert f.status_code == 200
        payload = f.json()
        assert isinstance(payload["epitope"], list)
        assert payload["antigen_length"] > 0


@pytest.mark.fc
class TestDesign:
    def test_design_small(self, client):
        r = client.post(
            "/api/design",
            data={
                "run_task": "design",
                "steps": "5",
                "num_samples": "1",
                "fasta_uri": AB_FASTA,
                "antigen_uri": ANTIGEN_PDB,
            },
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        final = poll_job(
            client, "", job_id,
            timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        )
        assert final["status"] == "completed", final
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        assert any(n.endswith(".pdb") for n in files), files
        assert any(n.endswith(".fasta") for n in files), files

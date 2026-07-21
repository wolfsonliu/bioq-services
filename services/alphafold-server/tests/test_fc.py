"""FC integration tests for alphafold-server (opt-in).

Run only when FC tests are enabled::

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/alphafold-server/tests/test_fc.py -v

AlphaFold jobs are long-running (MSA ~10-30 min + inference ~15-30 min).
Tests are structured to submit a single job and reuse it across all
lifecycle assertions.
"""

from __future__ import annotations

import io
import zipfile

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "alphafold-server"
SESSION_HEADER = "bioagent-session-id"

EXAMPLE_FASTA = """\
>test_ubiquitin
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
"""

# Small heterodimer (ubiquitin + a short helical peptide).  Total ~130
# residues keeps multimer MSA + inference inside FC's instance lifetime.
MULTIMER_FASTA = """\
>chainA
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
>chainB
GSHMGSAEYELDPKQIDDLEKEIATLEKERLALEKERLALEKERLALEKERLALEK
"""

TIMEOUT = httpx.Timeout(connect=30, read=600, write=60, pool=30)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url():
    return fc_url(SERVICE)


@pytest.fixture(scope="module")
def client(base_url):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture(scope="module")
def session_headers():
    import uuid

    return {SESSION_HEADER: f"test-{uuid.uuid4().hex[:12]}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_submitted(body: dict) -> str:
    assert "job_id" in body
    assert body["status"] in ("pending", "running")
    assert body.get("created_at") is not None
    return body["job_id"]


def _assert_completed(body: dict):
    assert body["status"] == "completed", f"Expected completed, got: {body}"
    assert body.get("started_at") is not None
    assert body.get("completed_at") is not None
    assert body.get("duration_seconds") is not None
    assert body["duration_seconds"] > 0
    assert body.get("output_count", 0) > 0
    assert body.get("output_total_bytes", 0) > 0


def _submit_fold(client: httpx.Client, *, fasta: str = EXAMPLE_FASTA, **params) -> dict:
    files = {"input_fasta": ("test.fasta", fasta.encode(), "text/plain")}
    r = client.post("/api/fold", data=params, files=files)
    assert r.status_code == 200, f"submit failed: {r.text}"
    return r.json()


def _download_bytes(client: httpx.Client, job_id: str, path: str) -> bytes:
    r = client.get(f"/api/jobs/{job_id}/file/{path}")
    assert r.status_code == 200, f"download {path} failed: {r.text}"
    return r.content


# ---------------------------------------------------------------------------
# Module-scoped job: submit once, reuse for all lifecycle tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fold_job(client, session_headers):
    """Submit a monomer_ptm fold job and poll until completion."""
    body = _submit_fold(
        client,
        model_preset="monomer_ptm",
        db_preset="reduced_dbs",
        models_to_relax="best",
    )
    job_id = _assert_submitted(body)

    result = poll_job(
        client,
        "",
        job_id,
        timeout_s=5400,
        interval_s=30,
        extra_headers=session_headers,
    )
    _assert_completed(result)
    return result


@pytest.fixture(scope="module")
def multimer_session_headers():
    """Distinct session id so multimer + monomer fixtures don't share an instance."""
    import uuid

    return {SESSION_HEADER: f"test-mm-{uuid.uuid4().hex[:12]}"}


@pytest.fixture(scope="module")
def multimer_fold_job(client, multimer_session_headers):
    """Submit a multimer fold job (small heterodimer) and poll until completion.

    Requires the `uniprot.fasta` Jackhmmer DB on NAS — multimer pipelines run
    an extra Jackhmmer pass against UniProt for paired MSAs that monomer
    presets skip.  The test will fail with a clear "Could not find Jackhmmer
    database" message if that DB is missing.
    """
    body = _submit_fold(
        client,
        fasta=MULTIMER_FASTA,
        model_preset="multimer",
        db_preset="reduced_dbs",
        models_to_relax="best",
        num_multimer_predictions_per_model="1",
    )
    job_id = _assert_submitted(body)

    result = poll_job(
        client,
        "",
        job_id,
        timeout_s=7200,
        interval_s=30,
        extra_headers=multimer_session_headers,
    )
    _assert_completed(result)
    return result


# ===================================================================
# Section 1: Smoke tests (no job submission needed)
# ===================================================================


@pytest.mark.fc
class TestSmoke:
    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "alphafold"
        assert "version" in body

    def test_healthz_detail(self, client):
        r = client.get("/healthz/detail")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "alphafold"
        assert "active_jobs" in body
        assert "max_concurrent_jobs" in body
        assert body["max_concurrent_jobs"] >= 1
        assert "disk_usage_mb" in body
        assert "disk_limit_mb" in body

    def test_openapi_served(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert "paths" in spec
        assert "/api/fold" in spec["paths"]


# ===================================================================
# Section 2: Manifest
# ===================================================================


@pytest.mark.fc
class TestManifest:
    def test_service_name_and_endpoint(self, client):
        r = client.get("/api/manifest")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "alphafold"
        paths = {e["path"] for e in body["endpoints"]}
        assert "/api/fold" in paths

    def test_service_specific_model(self, client):
        extras = client.get("/api/manifest").json()["service_specific"]
        model = extras["model"]
        assert "AlphaFold" in model["name"]
        assert model["output_format"] == "PDB"
        assert "monomer" in str(model["supports"]).lower()
        assert "multimer" in str(model["supports"]).lower()

    def test_service_specific_tool_outputs(self, client):
        extras = client.get("/api/manifest").json()["service_specific"]
        assert "fold" in extras["tool_outputs"]
        desc = extras["tool_outputs"]["fold"]
        assert "ranked" in desc
        assert "pdb" in desc.lower() or "PDB" in desc

    def test_service_specific_config_tips(self, client):
        extras = client.get("/api/manifest").json()["service_specific"]
        tips = extras["config_tips"]
        assert "model_preset" in tips
        assert "db_preset" in tips
        assert "models_to_relax" in tips
        assert "max_template_date" in tips

    def test_service_specific_uri_schemes(self, client):
        extras = client.get("/api/manifest").json()["service_specific"]
        schemes = extras["input_uri_schemes"]
        assert "upload" in schemes
        assert any("job://" in k for k in schemes)
        assert any("oss://" in k for k in schemes)

    def test_endpoint_examples(self, client):
        body = client.get("/api/manifest").json()
        by_path = {e["path"]: e for e in body["endpoints"]}
        examples = by_path["/api/fold"]["examples"]
        assert len(examples) >= 2
        titles = [e["title"] for e in examples]
        assert any("monomer" in t for t in titles)
        assert any("multimer" in t for t in titles)


# ===================================================================
# Section 3: Error cases (no real job, quick)
# ===================================================================


@pytest.mark.fc
class TestErrors:
    def test_unknown_job_status_404(self, client):
        r = client.get("/api/jobs/nonexistent-job-id")
        assert r.status_code == 404

    def test_unknown_job_files_404(self, client):
        r = client.get("/api/jobs/nonexistent-job-id/files")
        assert r.status_code == 404

    def test_unknown_job_log_404(self, client):
        r = client.get("/api/jobs/nonexistent-job-id/log")
        assert r.status_code == 404

    def test_unknown_job_download_404(self, client):
        r = client.get("/api/jobs/nonexistent-job-id/download")
        assert r.status_code == 404

    def test_unknown_job_file_404(self, client):
        r = client.get("/api/jobs/nonexistent-job-id/file/ranked_0.pdb")
        assert r.status_code == 404

    def test_fold_rejects_missing_fasta(self, client):
        r = client.post("/api/fold", data={"model_preset": "monomer_ptm"})
        assert r.status_code == 422


# ===================================================================
# Section 4: Fold inference (long-running, uses shared job fixture)
# ===================================================================


@pytest.mark.fc
class TestFoldMonomer:
    def test_job_completed(self, fold_job):
        assert fold_job["status"] == "completed"

    def test_input_params_echo(self, fold_job):
        params = fold_job.get("input_params", {})
        assert params.get("model_preset") == "monomer_ptm"
        assert params.get("db_preset") == "reduced_dbs"
        assert params.get("models_to_relax") == "best"

    def test_duration_reasonable(self, fold_job):
        d = fold_job["duration_seconds"]
        assert d > 30, "AlphaFold should take at least 30s"
        assert d < 7200, "AlphaFold should finish within 2h"


@pytest.mark.fc
class TestFoldMultimer:
    def test_job_completed(self, multimer_fold_job):
        assert multimer_fold_job["status"] == "completed"

    def test_input_params_echo(self, multimer_fold_job):
        params = multimer_fold_job.get("input_params", {})
        assert params.get("model_preset") == "multimer"
        assert params.get("db_preset") == "reduced_dbs"
        assert params.get("num_multimer_predictions_per_model") == 1

    def test_duration_reasonable(self, multimer_fold_job):
        d = multimer_fold_job["duration_seconds"]
        assert d > 30, "AlphaFold-Multimer should take at least 30s"
        assert d < 7200, "AlphaFold-Multimer should finish within 2h"

    def test_output_contains_ranked_pdb(self, client, multimer_fold_job):
        job_id = multimer_fold_job["job_id"]
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        assert any("ranked_0.pdb" in n for n in files), (
            f"Expected ranked_0.pdb in multimer output files: {files}"
        )

    def test_ranked_pdb_has_two_chains(self, client, multimer_fold_job):
        """A multimer ranked PDB must contain ATOM records for both chains."""
        job_id = multimer_fold_job["job_id"]
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        ranked = [f for f in files if "ranked_0.pdb" in f]
        assert ranked, "ranked_0.pdb not found"
        data = _download_bytes(client, job_id, ranked[0])
        text = data.decode("utf-8", errors="replace")
        chains = {
            line[21]
            for line in text.splitlines()
            if line.startswith("ATOM") and len(line) > 21
        }
        assert len(chains) >= 2, (
            f"Multimer PDB should contain >=2 chains, got: {chains}"
        )


# ===================================================================
# Section 5: Job lifecycle (files, download, log, delete)
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_job_files_endpoint(self, client, fold_job):
        job_id = fold_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/files")
        assert r.status_code == 200
        files = r.json()["files"]
        assert len(files) > 0
        assert any("ranked_0.pdb" in n for n in files), f"No ranked_0.pdb in {files}"

    def test_output_contains_ranking_debug(self, client, fold_job):
        job_id = fold_job["job_id"]
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        assert any("ranking_debug.json" in n for n in files), (
            f"Expected ranking_debug.json in output files: {files}"
        )

    def test_output_contains_relaxed_model(self, client, fold_job):
        job_id = fold_job["job_id"]
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        assert any("relaxed_model" in n for n in files), (
            f"Expected relaxed_model_*.pdb in output files: {files}"
        )

    def test_single_file_download_ranked_pdb(self, client, fold_job):
        job_id = fold_job["job_id"]
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        ranked = [f for f in files if "ranked_0.pdb" in f]
        assert ranked, "ranked_0.pdb not found"
        path = ranked[0]

        data = _download_bytes(client, job_id, path)
        assert len(data) > 100, "PDB file too small"
        text = data.decode("utf-8", errors="replace")
        assert "ATOM" in text, "PDB should contain ATOM records"

    def test_single_file_download_ranking_json(self, client, fold_job):
        job_id = fold_job["job_id"]
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        ranking = [f for f in files if "ranking_debug.json" in f]
        assert ranking, "ranking_debug.json not found"
        path = ranking[0]

        import json

        data = _download_bytes(client, job_id, path)
        parsed = json.loads(data)
        assert "order" in parsed, "ranking_debug.json should have 'order' key"
        assert "plddts" in parsed, "ranking_debug.json should have 'plddts' key"

    def test_job_log_endpoint(self, client, fold_job):
        job_id = fold_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/log")
        assert r.status_code == 200
        log = r.json()
        assert "log" in log or "text" in log
        log_text = log.get("log") or log.get("text", "")
        assert len(log_text) > 0, "Log should be non-empty for a completed job"

    def test_job_download_zip(self, client, fold_job):
        job_id = fold_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        assert "zip" in r.headers.get("content-type", "").lower() or len(r.content) > 100

        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert len(names) > 0, "Zip should contain files"
        assert any("ranked_0.pdb" in n for n in names), f"No ranked_0.pdb in zip: {names}"

    def test_job_file_not_found(self, client, fold_job):
        job_id = fold_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/file/nonexistent_file.xyz")
        assert r.status_code == 404

    def test_job_delete(self, client, fold_job):
        job_id = fold_job["job_id"]
        r = client.delete(f"/api/jobs/{job_id}")
        assert r.status_code == 200

        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 404

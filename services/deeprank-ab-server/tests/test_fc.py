"""FC integration tests for deeprank-ab-server (opt-in, submit/poll path).

Run only when FC tests are enabled::

    RUN_FC_TESTS=1 \
    uv run python -m pytest -m fc services/deeprank-ab-server/tests/test_fc.py -v

DeepRank-Ab has one inference endpoint:
  * ``/api/score`` — score an antibody-antigen docking complex PDB

The submit/poll path here is the legacy invocation mode.  For functional
testing prefer ``test_fc_task.py`` (async task mode) — it keeps the FC
function instance alive for the full pipeline, dedups by task id, and
spawns fewer parallel instances under the test load.

This file follows the same structure as alphafold-server/tests/test_fc.py:
a single shared antibody fixture (``score_job``) is reused across every
lifecycle assertion, and a second fixture (``nanobody_score_job``) backs the
VHH-mode class.  At most two real inference jobs run per session.
"""

from __future__ import annotations

import csv
import io
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "deeprank-ab-server"
SESSION_HEADER = "bioagent-session-id"

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_PDB = DATA_DIR / "test.pdb"

# Long read timeout for the submit POST itself (FC cold start + multipart
# upload of a few MB PDB).  poll_job uses its own short reads.
TIMEOUT = httpx.Timeout(connect=30, read=600, write=60, pool=30)

# DeepRank-Ab is ~3-10 min including ESM-2 embedding; allow 30 min total to
# absorb cold start + NAS contention.
POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 15


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
    """Session-affinity header so the antibody fixture binds to one FC instance."""
    return {SESSION_HEADER: f"test-{uuid.uuid4().hex[:12]}"}


@pytest.fixture(scope="module")
def nanobody_session_headers() -> dict[str, str]:
    """Distinct session id so nanobody + antibody fixtures get separate instances."""
    return {SESSION_HEADER: f"test-nb-{uuid.uuid4().hex[:12]}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_submitted(body: dict) -> str:
    assert "job_id" in body
    assert body["status"] in ("pending", "running")
    assert body.get("created_at") is not None
    assert isinstance(body.get("input_params"), dict)
    return body["job_id"]


def _assert_completed(body: dict) -> None:
    assert body["status"] == "completed", (
        f"failed: kind={body.get('failure_kind')} summary={body.get('error_summary')!r}"
    )
    assert body.get("started_at") is not None
    assert body.get("completed_at") is not None
    assert body.get("duration_seconds") is not None
    assert body["duration_seconds"] > 0
    assert body.get("output_count", 0) > 0
    assert body.get("output_total_bytes", 0) > 0


def _submit_score(
    client: httpx.Client,
    *,
    headers: dict[str, str] | None = None,
    pdb_path: Path = TEST_PDB,
    **form_fields: str,
) -> dict:
    with open(pdb_path, "rb") as fh:
        r = client.post(
            "/api/score",
            files={"input_pdb": (pdb_path.name, fh.read(), "chemical/x-pdb")},
            data=form_fields,
            headers=headers or {},
        )
    assert r.status_code == 200, f"submit failed: {r.status_code} {r.text!r}"
    return r.json()


def _download_bytes(client: httpx.Client, job_id: str, path: str) -> bytes:
    r = client.get(f"/api/jobs/{job_id}/file/{path}")
    assert r.status_code == 200, f"download {path} failed: {r.status_code} {r.text!r}"
    return r.content


# ---------------------------------------------------------------------------
# Module-scoped inference jobs — submit once, reuse across lifecycle tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def score_job(client: httpx.Client, session_headers: dict[str, str]) -> dict:
    """Submit the canonical H/L/A complex and poll until completion."""
    body = _submit_score(
        client,
        headers=session_headers,
        heavy_chain_id="H",
        light_chain_id="L",
        antigen_chain_id="A",
    )
    job_id = _assert_submitted(body)

    result = poll_job(
        client,
        "",
        job_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
        extra_headers=session_headers,
    )
    _assert_completed(result)
    return result


@pytest.fixture(scope="module")
def nanobody_score_job(
    client: httpx.Client, nanobody_session_headers: dict[str, str]
) -> dict:
    """Submit the same PDB in nanobody mode (light_chain_id='-') and poll."""
    body = _submit_score(
        client,
        headers=nanobody_session_headers,
        heavy_chain_id="H",
        light_chain_id="-",
        antigen_chain_id="A",
    )
    job_id = _assert_submitted(body)

    result = poll_job(
        client,
        "",
        job_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
        extra_headers=nanobody_session_headers,
    )
    _assert_completed(result)
    return result


# ===================================================================
# Section 1: Smoke (no inference compute)
# ===================================================================


@pytest.mark.fc
class TestSmoke:
    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "deeprank-ab"
        assert "version" in body

    def test_healthz_detail(self, client):
        """Custom /healthz/detail reports NAS-mounted ESM-2 weight presence."""
        r = client.get("/healthz/detail")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "deeprank-ab"
        # Weights externalized to NAS — verify both expected files.
        # See engineering/decisions/2026-06-26-service-weights-externalization.md.
        assert body["weights_loaded"] is True, (
            f"NAS ESM-2 weights missing: {body.get('weights_missing')}"
        )
        assert body["weights_dir"] == "/data/models/deeprank-ab/esm"
        assert body["weights_missing"] == {}
        assert body["max_concurrent_jobs"] >= 1
        assert isinstance(body.get("active_jobs"), int)

    def test_openapi_served(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert "/api/score" in spec["paths"]


# ===================================================================
# Section 2: Manifest
# ===================================================================


@pytest.mark.fc
class TestManifest:
    def test_score_endpoint_listed(self, client):
        body = client.get("/api/manifest").json()
        paths = {e["path"] for e in body["endpoints"]}
        # Sync endpoint must always be there; task endpoint is optional.
        assert "/api/score" in paths, (
            f"expected /api/score in manifest, got {paths}"
        )
        extras = paths - {"/api/score"}
        assert extras <= {"/api/tasks/score"}, (
            f"unexpected manifest endpoints: {extras}"
        )

    def test_service_specific_model_info(self, client):
        extras = client.get("/api/manifest").json()["service_specific"]
        info = extras["model_info"]
        assert "EGNN" in info["architecture"]
        assert "ESM-2" in info["sequence_encoder"]

    def test_service_specific_scoring_legend(self, client):
        extras = client.get("/api/manifest").json()["service_specific"]
        legend = extras["scoring_legend"]
        assert "predicted_dockq" in legend
        assert "quality_flag" in legend

    def test_service_specific_tool_outputs(self, client):
        extras = client.get("/api/manifest").json()["service_specific"]
        assert "score" in extras["tool_outputs"]

    def test_service_specific_uri_schemes(self, client):
        extras = client.get("/api/manifest").json()["service_specific"]
        schemes = extras["input_uri_schemes"]
        assert "upload" in schemes
        assert any("oss://" in k for k in schemes)

    def test_endpoint_examples(self, client):
        body = client.get("/api/manifest").json()
        by_path = {e["path"]: e for e in body["endpoints"]}
        examples = by_path["/api/score"]["examples"]
        assert len(examples) >= 2


# ===================================================================
# Section 3: Error cases (no real job, quick)
# ===================================================================


@pytest.mark.fc
class TestErrors:
    def test_unknown_job_status_404(self, client):
        assert client.get("/api/jobs/missing-job-id").status_code == 404

    def test_unknown_job_files_404(self, client):
        assert client.get("/api/jobs/missing-job-id/files").status_code == 404

    def test_unknown_job_log_404(self, client):
        assert client.get("/api/jobs/missing-job-id/log").status_code == 404

    def test_unknown_job_download_404(self, client):
        assert client.get("/api/jobs/missing-job-id/download").status_code == 404

    def test_unknown_job_file_404(self, client):
        assert client.get("/api/jobs/missing-job-id/file/foo.csv").status_code == 404

    def test_422_score_missing_input(self, client):
        """Neither upload nor URI → 422."""
        r = client.post(
            "/api/score",
            data={"heavy_chain_id": "H", "light_chain_id": "L", "antigen_chain_id": "A"},
        )
        assert r.status_code == 422


# ===================================================================
# Section 4: Score inference (long-running, uses shared job fixture)
# ===================================================================


@pytest.mark.fc
class TestScoreAntibody:
    def test_job_completed(self, score_job):
        assert score_job["status"] == "completed"

    def test_input_params_echo(self, score_job):
        params = score_job.get("input_params") or {}
        assert params.get("heavy_chain_id") == "H"
        assert params.get("light_chain_id") == "L"
        assert params.get("antigen_chain_id") == "A"

    def test_duration_reasonable(self, score_job):
        d = score_job["duration_seconds"]
        # ESM-2 embedding + EGNN > 10s; should easily finish within 30 min.
        assert d > 10, f"too fast for real DeepRank-Ab work: {d}s"
        assert d < POLL_TIMEOUT_S


@pytest.mark.fc
class TestScoreNanobody:
    """light_chain_id='-' path — quality_flag must skip 'low_HL_contacts'."""

    def test_job_completed(self, nanobody_score_job):
        assert nanobody_score_job["status"] == "completed"

    def test_input_params_echo(self, nanobody_score_job):
        params = nanobody_score_job.get("input_params") or {}
        assert params.get("light_chain_id") == "-"

    def test_quality_flag_skips_low_hl_contacts(self, client, nanobody_score_job):
        job_id = nanobody_score_job["job_id"]
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        csv_files = [f for f in files if f.endswith("_predictions.csv")]
        assert csv_files, f"no predictions CSV in nanobody output: {files}"

        text = _download_bytes(client, job_id, csv_files[0]).decode()
        for row in csv.DictReader(io.StringIO(text)):
            assert row["quality_flag"] in ("ok", "not_applicable"), (
                f"nanobody quality_flag must be 'ok' or 'not_applicable', "
                f"got {row['quality_flag']!r}"
            )


# ===================================================================
# Section 5: Job lifecycle (files, download, log, delete) — antibody only
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_files_endpoint_lists_predictions(self, client, score_job):
        job_id = score_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/files")
        assert r.status_code == 200
        files = r.json()["files"]
        assert any(f.endswith("_predictions.csv") for f in files), (
            f"_predictions.csv missing from outputs: {files}"
        )

    def test_files_endpoint_lists_hdf5(self, client, score_job):
        job_id = score_job["job_id"]
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        assert any(f.endswith(".hdf5") for f in files), (
            f"HDF5 (graph / predictions) missing from outputs: {files}"
        )

    def test_predictions_csv_schema_and_values(self, client, score_job):
        """CSV declares predicted_dockq + quality_flag; rows have plausible values."""
        job_id = score_job["job_id"]
        files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
        csv_path = next(f for f in files if f.endswith("_predictions.csv"))
        text = _download_bytes(client, job_id, csv_path).decode()
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)

        assert rows, "predictions CSV should have at least one row"
        assert "predicted_dockq" in reader.fieldnames, (
            f"missing predicted_dockq column: {reader.fieldnames}"
        )
        assert "quality_flag" in reader.fieldnames, (
            f"missing quality_flag column: {reader.fieldnames}"
        )
        for row in rows:
            dockq = float(row["predicted_dockq"])
            assert 0.0 <= dockq <= 1.0, f"predicted_dockq={dockq} out of [0,1]"
            assert row["quality_flag"] in ("ok", "low_HL_contacts", "not_applicable"), (
                f"unexpected quality_flag: {row['quality_flag']!r}"
            )

    def test_job_log_endpoint(self, client, score_job):
        job_id = score_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/log")
        assert r.status_code == 200
        body = r.json()
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)
        assert len(log_text) > 0, "log should be non-empty for a completed job"

    def test_job_download_zip(self, client, score_job):
        job_id = score_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        assert "zip" in r.headers.get("content-type", "").lower() or len(r.content) > 100
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any(n.endswith("_predictions.csv") for n in names), (
            f"predictions CSV missing from zip: {names}"
        )

    def test_job_file_not_found(self, client, score_job):
        job_id = score_job["job_id"]
        r = client.get(f"/api/jobs/{job_id}/file/nonexistent.xyz")
        assert r.status_code == 404

    def test_job_delete(self, client, score_job):
        """DELETE last — kept at end so other lifecycle tests still see the job."""
        job_id = score_job["job_id"]
        r = client.delete(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        assert r.json().get("status") == "deleted"

        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 404

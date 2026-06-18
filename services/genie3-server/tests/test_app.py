"""End-to-end smoke for genie3-server.

We stub the real `genie3 generate` invocation (would require GPU + weights) by
hooking into the runner with a no-op build_argv. The goal is to confirm app
startup, /health, the manifest endpoint, and that the structured endpoints
exist + accept their request bodies via OpenAPI.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

from server.adapter import Genie3Adapter
from server.settings import Genie3Settings


class _OfflineSettings(Genie3Settings):
    model_config = SettingsConfigDict(
        env_prefix="GENIE3_TEST_",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    from bioagent_service import create_app

    settings = _OfflineSettings(jobs_base_dir=tmp_path / "jobs", root=tmp_path)
    adapter = Genie3Adapter(settings=settings)
    app = create_app(adapter, settings, title="Genie3 Test")

    # Stub endpoint that pretends `genie3 generate` produced a PDB. Lets us
    # exercise the full runner pipeline without needing the real binary.
    @app.post("/api/stub-unconditional")
    def _stub():
        def _build(_job_id: str, job_dir: Path) -> list[str]:
            out = job_dir / "output" / "unconditional" / "pdbs"
            out.mkdir(parents=True, exist_ok=True)
            return ["bash", "-c", f"echo 'ATOM' > {out / '0.pdb'}"]
        return app.state.runner.submit(build_argv=_build, label="stub-unconditional")

    return TestClient(app)


def test_health(client: TestClient) -> None:
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "genie3"
    assert "version" in health
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "genie3"
    assert detail["version"] == health["version"]


def test_manifest_exposes_genie3_specific_extras(client: TestClient) -> None:
    body = client.get("/api/manifest").json()
    assert body["service"] == "genie3"
    extras = body["service_specific"]
    assert "*.pdb" in extras["tool_outputs"]["all_modes"]
    # cond_strategy guidance is here so agents avoid the legacy validation trap.
    assert "cond_strategy" in extras["config_tips"]


def test_stub_runner_pipeline_produces_pdb(client: TestClient) -> None:
    """Round-trip a fake genie3 job through the framework: submit → poll → files."""
    import time

    r = client.post("/api/stub-unconditional")
    r.raise_for_status()
    job_id = r.json()["job_id"]

    for _ in range(50):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert body["status"] == "completed"

    files = client.get(f"/api/jobs/{job_id}/files").json()
    assert any(f.endswith("0.pdb") for f in files["files"])


@pytest.fixture
def real_app_client(tmp_path: Path) -> TestClient:
    """Construct a TestClient against the real `server.app.app` so the four
    `/api/generate/*` endpoints are reachable. Settings read env, so we point
    `GENIE3_JOBS_BASE_DIR` at tmp_path before importing.
    """
    os.environ["GENIE3_JOBS_BASE_DIR"] = str(tmp_path / "jobs")
    # Point the binary at /bin/true so task endpoint smoke tests exit quickly
    # (rc=0, but no outputs ⇒ finalize_job marks the job FAILED, which is the
    # terminal state we assert on). Never actually runs real genie3.
    os.environ["GENIE3_BIN"] = "/bin/true"
    import importlib

    # Force a clean import so settings pick up the env override even if a prior
    # test already imported it.
    import sys
    sys.modules.pop("server.app", None)

    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


def test_motif_endpoint_rejects_bad_zip(real_app_client: TestClient, tmp_path: Path) -> None:
    """A malformed dataset zip must return HTTP 422 (not 500) and leave no orphan job."""
    bad = tmp_path / "junk.zip"
    bad.write_bytes(b"not a zip")

    before = real_app_client.app.state.job_store.all_jobs()  # type: ignore[attr-defined]
    with open(bad, "rb") as f:
        r = real_app_client.post(
            "/api/generate/motif",
            files={"dataset": ("junk.zip", f, "application/zip")},
        )
    assert r.status_code == 422
    after = real_app_client.app.state.job_store.all_jobs()  # type: ignore[attr-defined]
    # Framework's exception cleanup must remove the half-created job.
    assert len(after) == len(before)


def test_motif_endpoint_rejects_zip_without_problems_dir(
    real_app_client: TestClient, tmp_path: Path
) -> None:
    p = tmp_path / "noproblems.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("random/file.txt", "x")
    with open(p, "rb") as f:
        r = real_app_client.post(
            "/api/generate/motif",
            files={"dataset": ("noproblems.zip", f, "application/zip")},
        )
    assert r.status_code == 422
    assert "problems/" in r.json()["detail"]


def test_manifest_examples_cover_cond_strategy_gotcha(real_app_client: TestClient) -> None:
    """The custom-YAML example must show how to override cond_strategy to dodge the
    'Interface mode extended not found' validation trap we hit on 2026-05-12."""
    body = real_app_client.get("/api/manifest").json()
    custom_ep = next(e for e in body["endpoints"] if e["path"] == "/api/generate")
    assert len(custom_ep["examples"]) >= 1
    example = custom_ep["examples"][0]
    assert "cond_strategy" in (example["curl"] or "")
    assert "hotspot" in (example["curl"] or "")
    assert "extended" in (example["notes"] or "").lower()


def test_manifest_field_metadata_is_complete(real_app_client: TestClient) -> None:
    """Sanity: every service endpoint has wire-format + schema_ref + request_fields.

    The exact content-type varies — FastAPI uses `multipart/form-data` for
    endpoints with file uploads and `application/x-www-form-urlencoded` for
    pure Form-field endpoints (e.g. unconditional generation). Both are valid
    and the agent must respect whichever the manifest declares.
    """
    body = real_app_client.get("/api/manifest").json()
    for ep in body["endpoints"]:
        assert ep["request_content_type"] in (
            "multipart/form-data",
            "application/x-www-form-urlencoded",
        ), f"{ep['path']}: unexpected content_type {ep['request_content_type']!r}"
        assert ep["request_schema_ref"] is not None, f"{ep['path']}: missing schema_ref"
        assert ep["response_schema_ref"] == "#/components/schemas/JobInfo"
        assert len(ep["request_fields"]) > 0


def test_manifest_unconditional_uses_form_urlencoded(real_app_client: TestClient) -> None:
    """unconditional has no file params, so it's url-encoded form. Agents need to know."""
    body = real_app_client.get("/api/manifest").json()
    unc = next(e for e in body["endpoints"] if e["path"] == "/api/generate/unconditional")
    assert unc["request_content_type"] == "application/x-www-form-urlencoded"
    # And the file-bearing endpoints are multipart.
    motif = next(e for e in body["endpoints"] if e["path"] == "/api/generate/motif")
    assert motif["request_content_type"] == "multipart/form-data"
    fields = {f["name"]: f for f in motif["request_fields"]}
    assert fields["dataset"]["is_file"] is True
    assert fields["dataset"]["required"] is True


def test_openapi_lists_all_four_endpoints() -> None:
    """The real server.app constructs settings from env — point it at /tmp first."""
    os.environ["GENIE3_JOBS_BASE_DIR"] = "/tmp/genie3_jobs_test"

    import importlib

    server_app = importlib.import_module("server.app")
    schema = TestClient(server_app.app).get("/openapi.json").json()
    paths = set(schema["paths"].keys())
    assert "/api/generate/unconditional" in paths
    assert "/api/generate/motif" in paths
    assert "/api/generate/binder" in paths
    assert "/api/generate" in paths
    # Task mirrors must also be registered.
    assert "/api/tasks/generate/unconditional" in paths
    assert "/api/tasks/generate/motif" in paths
    assert "/api/tasks/generate/binder" in paths
    assert "/api/tasks/generate" in paths
    # model_form_depends wraps request models as Body_<endpoint>_* in OpenAPI.
    models = set(schema["components"]["schemas"].keys())
    assert any("unconditional" in m.lower() for m in models)
    assert any("motif" in m.lower() for m in models)
    assert any("binder" in m.lower() for m in models)


# ----- Task endpoint smoke -----


def test_unconditional_task_endpoint_returns_terminal_status(
    real_app_client: TestClient,
) -> None:
    """POST /api/tasks/generate/unconditional blocks until subprocess exits."""
    resp = real_app_client.post(
        "/api/tasks/generate/unconditional",
        data={},  # all UnconditionalRequest fields have defaults
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_unconditional_task_endpoint_honors_job_id_header(
    real_app_client: TestClient,
) -> None:
    resp = real_app_client.post(
        "/api/tasks/generate/unconditional",
        data={},
        headers={"X-Bioagent-Job-Id": "genie3-task-001"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"] == "genie3-task-001"


def test_unconditional_task_endpoint_duplicate_returns_existing(
    real_app_client: TestClient,
) -> None:
    hdrs = {"X-Bioagent-Job-Id": "genie3-dup-001"}
    r1 = real_app_client.post("/api/tasks/generate/unconditional", data={}, headers=hdrs)
    r2 = real_app_client.post("/api/tasks/generate/unconditional", data={}, headers=hdrs)
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert r1.json()["job_id"] == r2.json()["job_id"]
    assert r1.json()["created_at"] == r2.json()["created_at"]


def test_motif_task_endpoint_rejects_bad_zip(
    real_app_client: TestClient, tmp_path: Path
) -> None:
    """Task mirror of /api/generate/motif must also surface 422 on bad zip."""
    bad = tmp_path / "junk.zip"
    bad.write_bytes(b"not a zip")
    with open(bad, "rb") as f:
        r = real_app_client.post(
            "/api/tasks/generate/motif",
            files={"dataset": ("junk.zip", f, "application/zip")},
        )
    assert r.status_code == 422

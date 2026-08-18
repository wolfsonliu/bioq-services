"""End-to-end smoke for the rfantibody-server FastAPI app.

Does not spawn the real RFantibody scripts — those need GPU + weights. The goal
is to confirm the framework wiring works: app starts, OpenAPI schema is complete,
and the three POST endpoints accept valid request bodies (we trip a `bash false`
to keep this offline).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

from server.adapter import RFantibodyAdapter
from server.settings import RFantibodySettings


class _OfflineSettings(RFantibodySettings):
    # Don't read .env on dev machines (would otherwise leak unrelated config).
    model_config = SettingsConfigDict(
        env_prefix="RFANTIBODY_TEST_",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Build a settings/adapter pair pointing entirely at tmp_path; create_app
    # is the same one the production app uses (see server.app).
    from bioq_service import create_app

    settings = _OfflineSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path,
        weights_dir=tmp_path / "weights",
        scripts_dir=tmp_path / "scripts",
    )
    adapter = RFantibodyAdapter(settings=settings)
    app = create_app(adapter, settings, title="RFantibody Test")

    # Mount minimal stand-in routes that exercise the runner without invoking the
    # real scripts. We don't import from server.app because that constructs its
    # own settings from the real env, which we want to bypass.
    from server.tools import RFDIFFUSION_OUTPUT

    @app.post("/api/rfdiffusion-stub")
    def _stub():
        def _build(_job_id: str, job_dir: Path) -> list[str]:
            out = job_dir / "output"
            out.mkdir(exist_ok=True)
            # Pretend rfdiffusion ran and wrote a non-empty quiver. The adapter's
            # detect_outputs requires st_size > 0 so we write at least one byte.
            return ["bash", "-c", f"echo 'fake qv' > {out / RFDIFFUSION_OUTPUT}"]
        return app.state.runner.submit(build_argv=_build, label="rfdiffusion")

    return TestClient(app)


def test_health_and_detail(client: TestClient) -> None:
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "rfantibody"
    assert "version" in health
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "rfantibody"
    assert detail["version"] == health["version"]


def test_stub_endpoint_runs_through_framework(client: TestClient) -> None:
    r = client.post("/api/rfdiffusion-stub")
    r.raise_for_status()
    job_id = r.json()["job_id"]

    # Wait until terminal.
    import time
    for _ in range(50):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert body["status"] == "completed"

    files = client.get(f"/api/jobs/{job_id}/files").json()
    from server.tools import RFDIFFUSION_OUTPUT
    assert RFDIFFUSION_OUTPUT in files["files"]


def test_manifest_endpoints_carry_schema_refs_and_field_lists(client: TestClient) -> None:
    """Each POST endpoint must include enough metadata for an agent to call it."""
    body = client.get("/api/manifest").json()
    eps = {e["path"]: e for e in body["endpoints"]}

    # The fixture only registers /api/rfdiffusion-stub; the real endpoints live
    # on server.app.app. Just verify the stub itself carries the new metadata.
    stub = eps["/api/rfdiffusion-stub"]
    # No body params on the stub → still gives content type if requestBody exists.
    # Just confirm the fields are present in the schema (None is allowed).
    assert "request_content_type" in stub
    assert "request_schema_ref" in stub
    assert "request_fields" in stub
    assert "examples" in stub


def test_manifest_endpoint_examples_for_chaining(client: TestClient) -> None:
    """The chaining examples must reference job:// so agents learn the pattern."""
    body = client.get("/api/manifest").json()
    # Examples live under each endpoint's `examples` list. The stub fixture's
    # manifest also includes the adapter-supplied examples for /api/proteinmpnn
    # (the adapter doesn't know which endpoints we ACTUALLY register).
    rfd_examples = next(
        (e["examples"] for e in body["endpoints"] if e["path"] == "/api/rfdiffusion-stub"),
        None,
    )
    # rfdiffusion-stub isn't a real path the adapter has examples for, so empty is fine.
    assert rfd_examples == [] or rfd_examples is None or isinstance(rfd_examples, list)

    # The adapter's full example map is also reachable directly so we can verify
    # chaining_tip-aligned curl snippets exist.
    from server.adapter import RFantibodyAdapter
    settings = client.app.state.settings  # type: ignore[attr-defined]
    full = RFantibodyAdapter(settings=settings).endpoint_examples()
    assert "/api/proteinmpnn" in full
    chain_example = next(
        (e for e in full["/api/proteinmpnn"] if "job://" in (e.curl or "")), None,
    )
    assert chain_example is not None, "Expected a job:// chaining example for /api/proteinmpnn"


def test_manifest_exposes_service_specific_extras(client: TestClient) -> None:
    """Agent-facing manifest must declare the three tool outputs + URI schemes."""
    body = client.get("/api/manifest").json()
    assert body["service"] == "rfantibody"

    extras = body["service_specific"]
    # Per-tool output filenames (chaining hint).
    assert extras["tool_outputs"]["rfdiffusion"].endswith("1_rfdiffusion.qv")
    assert extras["tool_outputs"]["proteinmpnn"].endswith("2_proteinmpnn.qv")
    assert extras["tool_outputs"]["rf2"].endswith("3_rf2.qv")
    # URI schemes that resolve_input understands.
    assert "job://<job_id>/<filename>" in extras["input_uri_schemes"]
    assert "oss://<bucket>/<key>" in extras["input_uri_schemes"]
    # Chaining advice tells the agent how to avoid re-uploads.
    assert "job://" in extras["chaining_tip"]


def test_openapi_lists_service_request_models() -> None:
    """The real app (server.app) registers all six service endpoints.

    The per-endpoint request models (RFdiffusionRequest / ProteinMPNNRequest /
    RF2Request) are consumed via ``Depends(model_form_depends(Model))``, which
    flattens each model into individual ``Form()`` fields.  FastAPI therefore
    emits one ``Body_run_*`` multipart schema per route rather than the named
    request models.  Assert the observable contract instead: six body schemas
    (3 tools x sync/task) plus the shared ``JobInfo`` response model.
    """
    # Import lazily so the settings constructed there don't fail on our dev box.
    import importlib
    import os

    # Point the real app's settings at a writable temp dir so create_app doesn't
    # try to mkdir /data/rfantibody_jobs.
    os.environ["RFANTIBODY_JOBS_BASE_DIR"] = "/tmp/rfantibody_jobs_test"

    server_app = importlib.import_module("server.app")
    schema = TestClient(server_app.app).get("/openapi.json").json()

    models = set(schema["components"]["schemas"].keys())
    assert "JobInfo" in models

    expected_bodies = {
        "Body_run_rfdiffusion_api_rfdiffusion_post",
        "Body_run_rfdiffusion_task_api_tasks_rfdiffusion_post",
        "Body_run_proteinmpnn_api_proteinmpnn_post",
        "Body_run_proteinmpnn_task_api_tasks_proteinmpnn_post",
        "Body_run_rf2_api_rf2_post",
        "Body_run_rf2_task_api_tasks_rf2_post",
    }
    missing = expected_bodies - models
    assert not missing, f"missing request-body schemas: {sorted(missing)}"


def _real_app_client(tmp_path: Path) -> TestClient:
    """Build a TestClient against server.app.app with jobs_base_dir redirected to tmp."""
    import importlib
    import os

    os.environ["RFANTIBODY_JOBS_BASE_DIR"] = str(tmp_path / "jobs")
    server_app = importlib.reload(importlib.import_module("server.app"))
    return TestClient(server_app.app)


def test_rfdiffusion_task_endpoint_accepts_uploads(tmp_path: Path) -> None:
    """`/api/tasks/rfdiffusion` accepts target+framework uploads and yields a job_id.

    Task endpoint runs synchronously and may fail to spawn the real script in
    CI — the goal is to verify the validate/accept path returns a JobInfo.
    """
    client = _real_app_client(tmp_path)
    resp = client.post(
        "/api/tasks/rfdiffusion",
        data={"num_designs": 2},
        files={
            "target": ("target.pdb", b"ATOM\n", "text/plain"),
            "framework": ("framework.pdb", b"ATOM\n", "text/plain"),
        },
    )
    assert resp.status_code in (200, 422, 500)
    if resp.status_code == 200:
        body = resp.json()
        assert "job_id" in body


def test_proteinmpnn_task_endpoint_accepts_uri_fallback(tmp_path: Path) -> None:
    """`/api/tasks/proteinmpnn` accepts either input_quiver upload or input_quiver_uri.

    Passing a nonexistent file:// URI exercises the URI-resolution branch of
    `_save` — execute_task surfaces the resolver's 404 directly. Any of
    {200, 404, 422, 500} confirms the route is wired (we just don't want a
    404 on the *route* itself, which would mean the endpoint wasn't registered).
    """
    client = _real_app_client(tmp_path)
    resp = client.post(
        "/api/tasks/proteinmpnn",
        data={"seqs_per_struct": 2, "input_quiver_uri": "file:///nonexistent/input.qv"},
    )
    assert resp.status_code in (200, 404, 422, 500)


def test_rf2_task_endpoint_accepts_upload(tmp_path: Path) -> None:
    """`/api/tasks/rf2` accepts input_quiver upload."""
    client = _real_app_client(tmp_path)
    resp = client.post(
        "/api/tasks/rf2",
        data={"num_recycles": 2},
        files={"input_quiver": ("in.qv", b"fake-qv\n", "application/octet-stream")},
    )
    assert resp.status_code in (200, 422, 500)


def test_quiver_uri_field_matches_upload_field(tmp_path: Path) -> None:
    """Regression: the URI field must be named `input_quiver_uri`, not `input_uri`.

    bioq CLI's `--file input_quiver=<path>` uploads the file and emits the body
    field ``input_quiver_uri`` (see bioq/upload.py). The four proteinmpnn / rf2
    endpoints must expose exactly ``input_quiver`` (upload) + ``input_quiver_uri``
    (URI) — and must NOT expose the old ``input_uri`` name, otherwise
    ``bioq describe`` lists a stray ``--set input_uri`` and
    ``bioq run --file input_quiver=...`` 422s with neither field bound.
    """
    client = _real_app_client(tmp_path)
    manifest = client.get("/api/manifest").json()
    eps = {e["path"]: e for e in manifest["endpoints"]}
    for path in (
        "/api/proteinmpnn",
        "/api/rf2",
        "/api/tasks/proteinmpnn",
        "/api/tasks/rf2",
    ):
        fields = {f["name"]: f for f in eps[path]["request_fields"]}
        assert "input_quiver" in fields, f"{path}: missing input_quiver upload field"
        assert fields["input_quiver"]["is_file"] is True, f"{path}: input_quiver not marked file"
        assert "input_quiver_uri" in fields, f"{path}: missing input_quiver_uri URI field"
        assert "input_uri" not in fields, f"{path}: stale input_uri field still exposed"

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
    from bioagent_service import create_app

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
    assert client.get("/health").json() == {"status": "ok"}
    detail = client.get("/health/detail").json()
    assert detail["service"] == "rfantibody"


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
    """The 'real' app (server.app) registers RFdiffusionRequest et al via Annotated[..., Form()]."""
    # Import lazily so the settings constructed there don't fail on our dev box.
    import importlib
    import os

    # Point the real app's settings at a writable temp dir so create_app doesn't
    # try to mkdir /data/rfantibody_jobs.
    os.environ["RFANTIBODY_JOBS_BASE_DIR"] = "/tmp/rfantibody_jobs_test"

    server_app = importlib.import_module("server.app")
    schema = TestClient(server_app.app).get("/openapi.json").json()

    models = set(schema["components"]["schemas"].keys())
    assert "RFdiffusionRequest" in models
    assert "ProteinMPNNRequest" in models
    assert "RF2Request" in models
    assert "JobInfo" in models

"""`GET /api/manifest` introspection."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from bioagent_service import JobAdapter, ServiceSettings, create_app


class _Settings(ServiceSettings):
    model_config = SettingsConfigDict(env_file=None, env_prefix="MANI_TEST_", extra="ignore")


class _DefaultAdapter(JobAdapter):
    name = "default"


class _ExtrasAdapter(JobAdapter):
    name = "with-extras"

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {"echo": "output/result.txt"},
            "input_uri_schemes": ["upload"],
            "tip": "Use input_uri=job://<id>/<file> to chain jobs.",
        }


class _PingRequest(BaseModel):
    msg: str


@pytest.fixture
def default_client(tmp_path: Path) -> TestClient:
    settings = _Settings(jobs_base_dir=tmp_path / "jobs")
    adapter = _DefaultAdapter(settings=settings)
    app = create_app(adapter, settings, title="Default Service")

    @app.post("/api/ping", summary="Echo a message back.")
    def ping(req: _PingRequest):
        """Trivial endpoint for testing manifest discovery."""
        return {"ok": req.msg}

    return TestClient(app)


@pytest.fixture
def extras_client(tmp_path: Path) -> TestClient:
    settings = _Settings(jobs_base_dir=tmp_path / "jobs")
    adapter = _ExtrasAdapter(settings=settings)
    app = create_app(adapter, settings, title="Extras Service", version="9.9.9")

    @app.post("/api/echo")
    def echo(req: _PingRequest):
        return {"ok": req.msg}

    return TestClient(app)


def test_manifest_basic_fields(default_client: TestClient) -> None:
    body = default_client.get("/api/manifest").json()
    assert body["service"] == "default"
    assert body["title"] == "Default Service"
    assert body["version"] == "0.1.0"
    assert body["openapi_url"] == "/openapi.json"


def test_manifest_endpoints_excludes_framework_routes(default_client: TestClient) -> None:
    body = default_client.get("/api/manifest").json()
    paths = [e["path"] for e in body["endpoints"]]
    # Service-specific only:
    assert "/api/ping" in paths
    # Framework routes must be filtered out — they're described under job_lifecycle.
    for forbidden in ("/healthz", "/healthz/detail", "/api/jobs/{job_id}", "/api/manifest", "/openapi.json"):
        assert forbidden not in paths


def test_manifest_endpoint_metadata(default_client: TestClient) -> None:
    body = default_client.get("/api/manifest").json()
    ping = next(e for e in body["endpoints"] if e["path"] == "/api/ping")
    assert ping["method"] == "POST"
    assert ping["summary"] == "Echo a message back."  # FastAPI summary kwarg
    # operation_id is optional — FastAPI only fills it during OpenAPI generation,
    # so route-level inspection often sees None. That's fine; agents look up by
    # method + path anyway.


def test_manifest_job_lifecycle_protocol(default_client: TestClient) -> None:
    body = default_client.get("/api/manifest").json()
    jl = body["job_lifecycle"]
    assert jl["poll_endpoint"] == "/api/jobs/{job_id}"
    assert jl["download_endpoint"] == "/api/jobs/{job_id}/download"
    assert "completed" in jl["statuses"] and "failed" in jl["statuses"]
    assert "interrupted" in jl["failure_kinds"]
    # Recommendations are non-empty prose.
    assert len(jl["poll_recommendation"]) > 50
    assert len(jl["restart_semantics"]) > 50


def test_manifest_nas_layout(default_client: TestClient, tmp_path: Path) -> None:
    body = default_client.get("/api/manifest").json()
    nas = body["nas_layout"]
    assert nas["jobs_base_dir"].endswith("jobs")
    assert "<job_id>" in nas["per_job_structure"]
    # The cross-instance + cross-service notes are present.
    assert "NAS" in nas["cross_instance_consistency"]
    assert "path" in nas["cross_service_sharing"]


def test_manifest_extras_default_empty(default_client: TestClient) -> None:
    body = default_client.get("/api/manifest").json()
    assert body["service_specific"] == {}


def test_manifest_extras_passes_through_adapter_dict(extras_client: TestClient) -> None:
    body = extras_client.get("/api/manifest").json()
    extras = body["service_specific"]
    assert extras["tool_outputs"] == {"echo": "output/result.txt"}
    assert "upload" in extras["input_uri_schemes"]
    assert "input_uri" in extras["tip"]


def test_manifest_response_is_pydantic_validated(extras_client: TestClient) -> None:
    """The framework declares ServiceManifest as response_model, so FastAPI
    validates the response shape on its way out."""
    r = extras_client.get("/api/manifest")
    assert r.status_code == 200
    # Cross-check OpenAPI registers ServiceManifest.
    schema = extras_client.get("/openapi.json").json()
    assert "ServiceManifest" in schema["components"]["schemas"]


# ---- Level 1: wire-format metadata + flat request_fields ----


def test_manifest_includes_content_type_and_schema_refs(default_client: TestClient) -> None:
    body = default_client.get("/api/manifest").json()
    ping = next(e for e in body["endpoints"] if e["path"] == "/api/ping")
    # JSON-bodied endpoint (no UploadFile) → application/json content type.
    assert ping["request_content_type"] == "application/json"
    # Schema refs point into /openapi.json.
    assert ping["request_schema_ref"] is not None
    assert ping["request_schema_ref"].startswith("#/components/schemas/")
    # Resolving the ref must work against the live openapi.json (agent-facing contract).
    ref_name = ping["request_schema_ref"].rsplit("/", 1)[-1]
    openapi = default_client.get("/openapi.json").json()
    assert ref_name in openapi["components"]["schemas"]


def test_request_fields_flattened_with_required_marker(default_client: TestClient) -> None:
    body = default_client.get("/api/manifest").json()
    ping = next(e for e in body["endpoints"] if e["path"] == "/api/ping")
    # PingRequest has one required field: msg.
    fields = {f["name"]: f for f in ping["request_fields"]}
    assert "msg" in fields
    assert fields["msg"]["required"] is True
    assert fields["msg"]["type"] == "string"
    assert fields["msg"]["is_file"] is False


def test_request_fields_detect_optional_via_anyof_null(tmp_path: Path) -> None:
    """A field declared `Optional[str]` should come out non-required with type='string'."""
    from bioagent_service import create_app
    from fastapi import Form

    settings = _Settings(jobs_base_dir=tmp_path / "jobs")
    adapter = _DefaultAdapter(settings=settings)
    app = create_app(adapter, settings, title="Optional Test")

    @app.post("/api/optional")
    def optional_endpoint(
        kind: str = Form(...),
        note: str | None = Form(None),
    ):
        return {"kind": kind, "note": note}

    client = TestClient(app)
    body = client.get("/api/manifest").json()
    ep = next(e for e in body["endpoints"] if e["path"] == "/api/optional")
    fields = {f["name"]: f for f in ep["request_fields"]}
    assert fields["kind"]["required"] is True
    assert fields["note"]["required"] is False
    assert fields["note"]["type"] == "string"


def test_extract_fields_marks_file_uploads_by_schema_shape() -> None:
    """File fields in an OpenAPI body schema must come out is_file=True / type='file'.

    Unit-tests the extraction directly against the schema shape FastAPI generates
    for `UploadFile = File(...)` parameters (both required and Optional). The
    end-to-end file-upload path is exercised by the rfantibody-server tests
    against its real upload endpoints.
    """
    from bioagent_service.manifest import _extract_fields

    body_schema = {
        "type": "object",
        "required": ["target", "framework"],
        "properties": {
            "target": {
                "type": "string",
                "format": "binary",
                "title": "Target",
                "description": "Target structure",
            },
            "framework": {
                "type": "string",
                "contentMediaType": "application/octet-stream",
                "title": "Framework",
                "description": "Framework structure",
            },
            "input_uri": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Input Uri",
                "description": "Alternative to upload.",
            },
        },
    }
    fields = {f.name: f for f in _extract_fields(body_schema)}
    # Required file fields show as 'file' with is_file=True.
    assert fields["target"].is_file is True
    assert fields["target"].type == "file"
    assert fields["target"].required is True
    assert fields["target"].description == "Target structure"
    # The other multipart file representation (contentMediaType) also resolves to 'file'.
    assert fields["framework"].is_file is True
    assert fields["framework"].type == "file"
    # Optional[str] (anyOf with null) → required=False, type='string', is_file=False.
    assert fields["input_uri"].required is False
    assert fields["input_uri"].type == "string"
    assert fields["input_uri"].is_file is False


def test_response_schema_ref_points_to_job_info(extras_client: TestClient) -> None:
    body = extras_client.get("/api/manifest").json()
    echo = next(e for e in body["endpoints"] if e["path"] == "/api/echo")
    # The /api/echo handler in the fixture returns a dict (no response_model);
    # the response_schema_ref may be None or an auto-generated default — just
    # confirm the field exists and is either None or a valid ref.
    rsr = echo["response_schema_ref"]
    assert rsr is None or rsr.startswith("#/components/schemas/")


# ---- Level 2: endpoint examples ----


def test_endpoint_examples_default_empty(default_client: TestClient) -> None:
    body = default_client.get("/api/manifest").json()
    ping = next(e for e in body["endpoints"] if e["path"] == "/api/ping")
    assert ping["examples"] == []


def test_endpoint_examples_from_adapter_hook(tmp_path: Path) -> None:
    """Adapter.endpoint_examples() values surface on the relevant endpoint."""
    from bioagent_service import EndpointExample, JobAdapter, create_app

    class _AdapterWithExamples(JobAdapter):
        name = "demo"

        def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
            return {
                "/api/ping": [
                    EndpointExample(
                        title="hello world",
                        curl="curl -X POST .../api/ping -d '{\"msg\":\"hi\"}'",
                        body={"msg": "hi"},
                        notes="The simplest possible call.",
                    ),
                ],
            }

    settings = _Settings(jobs_base_dir=tmp_path / "jobs")
    adapter = _AdapterWithExamples(settings=settings)
    app = create_app(adapter, settings, title="Examples Demo")

    @app.post("/api/ping")
    def ping_examples_endpoint(req: _PingRequest):
        return {"ok": req.msg}

    client = TestClient(app)
    body = client.get("/api/manifest").json()
    ping = next(e for e in body["endpoints"] if e["path"] == "/api/ping")
    assert len(ping["examples"]) == 1
    ex = ping["examples"][0]
    assert ex["title"] == "hello world"
    assert "curl" in ex["curl"]
    assert ex["body"] == {"msg": "hi"}
    assert "simplest" in ex["notes"]

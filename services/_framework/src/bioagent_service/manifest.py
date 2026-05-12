"""Service manifest — agent-friendly introspection endpoint.

Every bioagent service exposes `GET /api/manifest` (registered by `create_app`).
The response is a curated, self-contained protocol description aimed at LLM
agents that want to *use* the service without parsing the full OpenAPI document.

For each service-defined endpoint the manifest carries:

  * Identity   — `method`, `path`, `summary`, `description`
  * **Wire format** — `request_content_type` (multipart vs JSON), and
    `request_schema_ref` / `response_schema_ref` pointing into `/openapi.json`
    for the full JSON Schema
  * **Flat fields** — `request_fields` lists each top-level body field with
    `name` / `type` / `required` / `description` / `is_file`, so an agent can
    construct a valid request without dereferencing $refs
  * **Examples** — copy-pasteable `curl` / `python` / `body` snippets supplied
    by `JobAdapter.endpoint_examples()`

Plus framework-level sections (`job_lifecycle`, `nas_layout`) and the adapter's
free-form `service_specific` extras (output filename conventions, chaining
hints, config gotchas).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, Request
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

from bioagent_service.adapter import JobAdapter
from bioagent_service.models import FailureKind, JobStatus
from bioagent_service.settings import ServiceSettings

# Routes the framework owns. The manifest excludes these from `endpoints`
# (they're described in `job_lifecycle` instead) so the agent sees a clean list
# of service-specific operations.
_FRAMEWORK_PATH_PREFIXES = (
    "/health",
    "/api/jobs",
    "/api/manifest",
    "/openapi.json",
    "/docs",
    "/redoc",
)


class FieldInfo(BaseModel):
    """One request-body field, flattened from the OpenAPI schema."""

    name: str = Field(..., description="Field name as it appears in the request body / form.")
    type: str = Field(
        ...,
        description=(
            "Human-friendly type name: 'string', 'integer', 'file' (multipart upload), "
            "'array[string]', or the name of a referenced pydantic model "
            "(in which case look it up in /openapi.json#/components/schemas)."
        ),
    )
    required: bool = Field(..., description="True iff the field must be present in the request.")
    description: str | None = Field(
        default=None, description="From Field(description=...) / docstring."
    )
    is_file: bool = Field(
        default=False,
        description="True iff this is a multipart file upload (rather than a form/json scalar).",
    )
    default: Any = Field(
        default=None,
        description="Default value if any (only meaningful for non-required scalar fields).",
    )


class EndpointExample(BaseModel):
    """A copy-pasteable example for one endpoint."""

    title: str = Field(..., description="Short label (e.g., 'chain from previous job').")
    curl: str | None = Field(default=None, description="Ready-to-run curl command.")
    python: str | None = Field(default=None, description="Equivalent Python (httpx / requests) snippet.")
    body: dict[str, Any] | None = Field(
        default=None,
        description="Structured request body, useful for JSON endpoints. Files are not represented here.",
    )
    notes: str | None = Field(
        default=None, description="When an agent should reach for this example over the others.",
    )


class EndpointInfo(BaseModel):
    """One service-defined HTTP operation, with enough metadata to construct a request."""

    method: str = Field(..., description="HTTP method, e.g. 'POST'.")
    path: str = Field(..., description="URL path, e.g. '/api/rfdiffusion'.")
    summary: str | None = Field(
        default=None,
        description="One-line purpose (FastAPI route summary or first docstring line).",
    )
    description: str | None = Field(
        default=None,
        description="Longer prose (FastAPI route description or docstring body).",
    )
    operation_id: str | None = Field(
        default=None,
        description=(
            "OpenAPI operation id; agents can fetch /openapi.json and look up "
            "/paths/{path}/{method} for the full request/response schemas."
        ),
    )
    request_content_type: str | None = Field(
        default=None,
        description=(
            "Wire format the body uses: 'application/json' for plain JSON, "
            "'multipart/form-data' when files are involved, etc."
        ),
    )
    request_schema_ref: str | None = Field(
        default=None,
        description=(
            "JSON pointer into /openapi.json (e.g. "
            "'#/components/schemas/Body_run_rfdiffusion_api_rfdiffusion_post'). "
            "Resolve there for the full schema; the flat `request_fields` list "
            "below covers the common case."
        ),
    )
    response_schema_ref: str | None = Field(
        default=None,
        description="Pointer to the 200-response schema (typically '#/components/schemas/JobInfo').",
    )
    request_fields: list[FieldInfo] = Field(
        default_factory=list,
        description="Top-level fields of the request body, flattened for agent consumption.",
    )
    examples: list[EndpointExample] = Field(
        default_factory=list,
        description="Copy-pasteable examples (from `adapter.endpoint_examples()`).",
    )


class JobLifecycle(BaseModel):
    """Framework-provided job state machine. Identical across all bioagent services."""

    poll_endpoint: str = "/api/jobs/{job_id}"
    files_endpoint: str = "/api/jobs/{job_id}/files"
    log_endpoint: str = "/api/jobs/{job_id}/log"
    download_endpoint: str = "/api/jobs/{job_id}/download"
    single_file_endpoint: str = "/api/jobs/{job_id}/file/{path}"
    delete_endpoint: str = "/api/jobs/{job_id}"
    statuses: list[str] = Field(
        default_factory=lambda: [s.value for s in JobStatus],
        description="All possible values of JobInfo.status.",
    )
    failure_kinds: list[str] = Field(
        default_factory=lambda: [k.value for k in FailureKind],
        description="All possible values of JobInfo.failure_kind (only set on failure).",
    )
    poll_recommendation: str = (
        "Poll the status endpoint every 5–120 s until status ∈ "
        "{'completed', 'failed'}. On failure, JobInfo.error_summary holds a "
        "one-line exception summary and error_tail holds the trailing ~4 KB of "
        "the subprocess log — usually enough for triage without a separate /log "
        "fetch."
    )
    restart_semantics: str = (
        "Jobs survive process and FC instance restarts via a `job.json` sidecar "
        "in each job directory. Jobs that were RUNNING at the moment of restart "
        "are downgraded to FAILED with failure_kind='interrupted'."
    )


class NasLayout(BaseModel):
    """Where each job's data lives on disk; shared across multi-instance FC."""

    jobs_base_dir: Path = Field(..., description="Root directory under which per-job dirs are created.")
    per_job_structure: str = (
        "<jobs_base_dir>/<job_id>/{input/, output/, logs/run.log, job.json}"
    )
    cross_instance_consistency: str = (
        "When multiple FC instances mount the same NAS at jobs_base_dir, all of "
        "them can see each other's jobs via read-through cache on GET. Writes "
        "(create/update) only happen on the instance that owns the subprocess."
    )
    cross_service_sharing: str = (
        "Other bioagent services mounted on the same NAS can read this "
        "service's output files directly by path, avoiding HTTP round-trips. "
        "Example: another service does `Path('/data/rfantibody_jobs/<id>/output/file.qv').read_bytes()`."
    )


class ServiceManifest(BaseModel):
    """Agent-facing protocol description for one bioagent service."""

    service: str = Field(..., description="Adapter.name; stable identifier across versions.")
    title: str = Field(..., description="Human-readable title (FastAPI app title).")
    version: str = Field(..., description="Semantic version of the service image.")
    description: str | None = Field(
        default=None,
        description="Free-form prose (FastAPI app description).",
    )
    endpoints: list[EndpointInfo] = Field(
        default_factory=list,
        description="Service-specific POST routes. Framework lifecycle routes are NOT included.",
    )
    job_lifecycle: JobLifecycle
    nas_layout: NasLayout
    service_specific: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Whatever `adapter.manifest_extras()` returned. Services document "
            "their output filename conventions, supported input URI schemes, "
            "and any tool-specific hints here."
        ),
    )
    openapi_url: str = Field(
        default="/openapi.json",
        description="Where to fetch the full JSON Schema for every request/response model.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_framework_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in _FRAMEWORK_PATH_PREFIXES)


def _peel_optional(schema: dict[str, Any]) -> dict[str, Any]:
    """If schema is `anyOf: [X, null]` (= Optional[X]), unwrap to X."""
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            return non_null[0]
    return schema


def _is_file_schema(schema: dict[str, Any]) -> bool:
    """Multipart file uploads show up as either format=binary or contentMediaType=...octet-stream."""
    if schema.get("type") != "string":
        return False
    return (
        schema.get("format") == "binary"
        or schema.get("contentMediaType") == "application/octet-stream"
    )


def _format_type(schema: dict[str, Any]) -> str:
    """Render a schema as a short human-readable type string for agents."""
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if _is_file_schema(schema):
        return "file"
    t = schema.get("type", "any")
    if t == "array":
        return f"array[{_format_type(schema.get('items', {}))}]"
    return t


def _extract_fields(body_schema: dict[str, Any]) -> list[FieldInfo]:
    """Walk a body schema's `properties` + `required` list into a flat FieldInfo[]."""
    properties = body_schema.get("properties", {}) or {}
    required_set = set(body_schema.get("required", []) or [])
    out: list[FieldInfo] = []
    for fname, fschema in properties.items():
        actual = _peel_optional(fschema)
        out.append(
            FieldInfo(
                name=fname,
                type=_format_type(actual),
                required=fname in required_set,
                description=fschema.get("description") or actual.get("description"),
                is_file=_is_file_schema(actual),
                default=fschema.get("default"),
            )
        )
    return out


def _resolve_ref(openapi_spec: dict[str, Any], ref: str) -> dict[str, Any]:
    """Dereference a `#/components/schemas/Foo`-style pointer. Returns {} on miss."""
    if not ref.startswith("#/"):
        return {}
    parts = ref[2:].split("/")
    node: Any = openapi_spec
    for p in parts:
        if not isinstance(node, dict):
            return {}
        node = node.get(p, {})
    return node if isinstance(node, dict) else {}


def _operation_metadata(
    openapi_spec: dict[str, Any], path: str, method: str
) -> tuple[Optional[str], Optional[str], Optional[str], list[FieldInfo]]:
    """For one path+method, pull (content_type, req_ref, resp_ref, request_fields) from OpenAPI."""
    operation = openapi_spec.get("paths", {}).get(path, {}).get(method.lower(), {})

    # ---- Request ----
    request_body = operation.get("requestBody") or {}
    content = request_body.get("content", {}) or {}
    content_type: Optional[str] = next(iter(content.keys()), None)
    req_ref: Optional[str] = None
    req_fields: list[FieldInfo] = []
    if content_type:
        req_schema = content[content_type].get("schema", {}) or {}
        req_ref = req_schema.get("$ref")
        body_schema = _resolve_ref(openapi_spec, req_ref) if req_ref else req_schema
        req_fields = _extract_fields(body_schema)

    # ---- Response ----
    responses = operation.get("responses", {}) or {}
    success = responses.get("200") or responses.get("default") or {}
    resp_content = success.get("content", {}) or {}
    first_resp = next(iter(resp_content.values()), {}) if resp_content else {}
    resp_ref: Optional[str] = first_resp.get("schema", {}).get("$ref")

    return content_type, req_ref, resp_ref, req_fields


def _service_endpoints(
    app: FastAPI, examples_by_path: dict[str, list[EndpointExample]]
) -> list[EndpointInfo]:
    """Walk app.routes and return only service-defined operations, enriched from OpenAPI."""
    # Generate the OpenAPI spec once (FastAPI caches it on the app object).
    openapi_spec = app.openapi()

    out: list[EndpointInfo] = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if _is_framework_path(route.path):
            continue
        for method in sorted(m for m in route.methods if m != "HEAD"):
            content_type, req_ref, resp_ref, req_fields = _operation_metadata(
                openapi_spec, route.path, method
            )
            doc = (route.endpoint.__doc__ or "").strip()
            out.append(
                EndpointInfo(
                    method=method,
                    path=route.path,
                    summary=route.summary or (doc.split("\n", 1)[0] if doc else None) or None,
                    description=route.description or None,
                    operation_id=route.operation_id,
                    request_content_type=content_type,
                    request_schema_ref=req_ref,
                    response_schema_ref=resp_ref,
                    request_fields=req_fields,
                    examples=examples_by_path.get(route.path, []),
                )
            )
    out.sort(key=lambda e: (e.path, e.method))
    return out


def build_manifest(
    app: FastAPI,
    adapter: JobAdapter,
    settings: ServiceSettings,
) -> ServiceManifest:
    """Assemble the manifest by introspecting the app + asking the adapter for extras."""
    examples_by_path = adapter.endpoint_examples()
    return ServiceManifest(
        service=adapter.name,
        title=app.title,
        version=app.version,
        description=app.description or None,
        endpoints=_service_endpoints(app, examples_by_path),
        job_lifecycle=JobLifecycle(),
        nas_layout=NasLayout(jobs_base_dir=settings.jobs_base_dir),
        service_specific=adapter.manifest_extras(),
    )


def make_manifest_router() -> APIRouter:
    """Returns a router exposing `GET /api/manifest`."""
    router = APIRouter()

    @router.get(
        "/api/manifest",
        response_model=ServiceManifest,
        summary="Agent-friendly protocol description for this service.",
    )
    def get_manifest(request: Request) -> ServiceManifest:
        return build_manifest(
            request.app,
            request.app.state.adapter,
            request.app.state.settings,
        )

    return router


__all__ = [
    "EndpointExample",
    "EndpointInfo",
    "FieldInfo",
    "JobLifecycle",
    "NasLayout",
    "ServiceManifest",
    "build_manifest",
    "make_manifest_router",
]

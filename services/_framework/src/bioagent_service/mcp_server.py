"""MCP server auto-mounted on each bioagent service when `create_app(enable_mcp=True)`.

For every service-defined POST route the framework auto-generates one MCP tool
that submits the job (returns `JobInfo`), plus four framework-wide lifecycle
tools that wrap the existing read/download endpoints:

  * `submit_<name>`         — submit a new job to one of the POST endpoints
  * `get_job_status(...)`   — wraps GET /api/jobs/{id}
  * `list_job_files(...)`   — wraps GET /api/jobs/{id}/files
  * `get_job_log(...)`      — wraps GET /api/jobs/{id}/log
  * `download_job_file(...)` — wraps GET /api/jobs/{id}/file/{path}

Dispatch is via in-process httpx ASGI — the tool body issues an HTTP request
against the same FastAPI app, so the MCP layer adds zero new code paths in the
job pipeline. The trade-off: tools that need binary file uploads aren't exposed
yet; clients must reference files via URI inputs (`job://`, `file://`, `oss://`)
that the service's adapter already understands. The plain HTTP endpoints remain
available for byte uploads, and the MCP `submit_*` tools accept all non-file
fields exactly as the HTTP form does.

Two transports are supported:
  * **Streamable HTTP** (mounted at `/mcp` in the FastAPI app) — used when the
    same image is deployed to Alibaba Cloud FC. Fits FC's request-response
    model; no long-lived SSE connection is required.
  * **stdio** — for local Claude Desktop / Cursor / IDE integrations. Started
    via the `bioagent-service-mcp-stdio` CLI entry point in `mcp_stdio.py`.
"""

# Note: NO `from __future__ import annotations` here. FastMCP runs
# `inspect.signature(fn, eval_str=True)` on registered tool functions, and
# stringified annotations referring to dynamically-built pydantic models
# can't be resolved against this module's globals at evaluation time.
# Real-object annotations on the closures sidestep the forward-ref dance.

import inspect
import logging
from typing import Any, Optional, get_args, get_origin

import httpx
from fastapi import FastAPI, UploadFile
from fastapi.routing import APIRoute
from pydantic import BaseModel, create_model

from bioagent_service.adapter import JobAdapter
from bioagent_service.settings import ServiceSettings

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover - guarded by enable_mcp
    raise ImportError(
        "MCP support requested but `mcp` is not installed. "
        "Install with `pip install 'bioagent-service-framework[mcp]'`."
    ) from e

logger = logging.getLogger(__name__)

# Routes the framework itself owns — they get explicit lifecycle tools and must
# not be auto-registered as `submit_*` tools.
_FRAMEWORK_PATH_PREFIXES = (
    "/health",
    "/api/jobs",
    "/api/manifest",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/mcp",
)


# ---------------------------------------------------------------------------
# Helpers — route signature introspection
# ---------------------------------------------------------------------------


def _is_framework_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in _FRAMEWORK_PATH_PREFIXES)


def _peel_optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap `Optional[X]` / `X | None`. Returns (inner, was_optional)."""
    origin = get_origin(annotation)
    if origin is None:
        return annotation, False
    args = get_args(annotation)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1 and len(args) >= 2:
        return non_none[0], True
    return annotation, False


def _is_file_param(annotation: Any) -> bool:
    """True iff the parameter is a (possibly-optional) `UploadFile`."""
    inner, _ = _peel_optional(annotation)
    try:
        return inspect.isclass(inner) and issubclass(inner, UploadFile)
    except TypeError:
        return False


def _is_basemodel(annotation: Any) -> bool:
    inner, _ = _peel_optional(annotation)
    try:
        return inspect.isclass(inner) and issubclass(inner, BaseModel)
    except TypeError:
        return False


def _peel_annotated(annotation: Any) -> Any:
    """If annotation is `Annotated[T, ...]`, return T; otherwise return as-is.

    Detected via `__metadata__` (set on `typing._AnnotatedAlias` instances) —
    `get_origin()` returns `typing.Annotated` itself which is not a reliable
    discriminator across Python versions.
    """
    if hasattr(annotation, "__metadata__"):
        args = get_args(annotation)
        if args:
            return _peel_annotated(args[0])
    return annotation


def _route_short_name(path: str) -> str:
    """`/api/design` -> `design`, `/api/some/thing` -> `some_thing`."""
    suffix = path.removeprefix("/api/").strip("/")
    return suffix.replace("/", "_") or "root"


# ---------------------------------------------------------------------------
# Dynamic input-model construction
# ---------------------------------------------------------------------------


def _strip_annotated(annotation: Any) -> Any:
    """Recursively unwrap `Annotated[T, ...]` to just `T`.

    FastAPI uses `Annotated[Model, Form()]` heavily, and pydantic field
    annotations can themselves carry `Annotated` metadata (e.g. `conint`).
    Carrying those over into a dynamically-created wrapper model triggers
    `class-not-fully-defined` errors because the wrapper's evaluation
    context lacks `Annotated`. The metadata isn't useful for MCP input
    validation anyway — we just want the underlying type.
    """
    if get_origin(annotation) is None:
        return annotation
    args = get_args(annotation)
    # `typing.Annotated[X, ...]` has origin `X` after `get_origin`, but the
    # canonical detection is via `__metadata__`.
    if hasattr(annotation, "__metadata__"):
        return _strip_annotated(args[0])
    return annotation


class _RouteSchema:
    """How to reconstruct a route handler's kwargs from a flat InputModel instance.

    `InputModel` is the auto-built pydantic model the MCP tool exposes — its
    fields are the union of every pydantic-BaseModel parameter's fields plus
    every scalar Form/URI parameter, with file parameters excluded.
    `param_plan` records, per handler parameter, how to rebuild the value at
    call time: `("model", ParamModelCls, [field_names])` rebuilds the pydantic
    instance from a subset of InputModel fields; `("scalar", None, [field])`
    passes one InputModel field through; `("file", None, [])` passes None
    (no binary upload via MCP).
    """

    def __init__(
        self,
        input_model: type[BaseModel],
        param_plan: list[tuple[str, str, Optional[type[BaseModel]], list[str]]],
        skipped_files: list[str],
    ) -> None:
        self.input_model = input_model
        self.param_plan = param_plan  # (param_name, kind, model_cls, [field_names])
        self.skipped_files = skipped_files

    def build_kwargs(self, body: BaseModel) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for pname, kind, model_cls, fnames in self.param_plan:
            if kind == "model":
                assert model_cls is not None
                sub = {f: getattr(body, f) for f in fnames if hasattr(body, f)}
                kwargs[pname] = model_cls(**sub)
            elif kind == "scalar":
                kwargs[pname] = getattr(body, fnames[0]) if fnames else None
            elif kind == "file":
                kwargs[pname] = None  # MCP transport doesn't stream bytes
        return kwargs


def _build_route_schema(route: APIRoute) -> Optional[_RouteSchema]:
    """Inspect `route.endpoint` and produce a `_RouteSchema` describing both the
    InputModel exposed to MCP clients and how to reconstruct kwargs at call time.
    Returns None if the route has no introspectable params.
    """
    # eval_str=True resolves PEP-563 string annotations against the endpoint's
    # module globals. Without this, every service that uses
    # `from __future__ import annotations` (which is most of them) gives us
    # raw strings like "Optional[str]" that our `_peel_*` helpers can't handle.
    try:
        sig = inspect.signature(route.endpoint, eval_str=True)
    except (NameError, TypeError):
        sig = inspect.signature(route.endpoint)

    fields: dict[str, tuple[Any, Any]] = {}
    param_plan: list[tuple[str, str, Optional[type[BaseModel]], list[str]]] = []
    skipped_files: list[str] = []
    used_field_names: set[str] = set()

    for pname, param in sig.parameters.items():
        annotation = _peel_annotated(param.annotation)

        if pname in {"request", "background_tasks"}:
            continue

        if _is_file_param(annotation):
            skipped_files.append(pname)
            param_plan.append((pname, "file", None, []))
            continue

        if _is_basemodel(annotation):
            inner, _ = _peel_optional(annotation)
            this_param_fields: list[str] = []
            for fname, finfo in inner.model_fields.items():
                if fname in used_field_names:
                    logger.warning(
                        "duplicate field %r in route %s — first occurrence wins, %s.%s skipped",
                        fname,
                        route.path,
                        pname,
                        fname,
                    )
                    continue
                clean_anno = _strip_annotated(finfo.annotation)
                if finfo.is_required():
                    default = ...
                else:
                    default = finfo.default if finfo.default is not None else None
                fields[fname] = (clean_anno, default)
                used_field_names.add(fname)
                this_param_fields.append(fname)
            param_plan.append((pname, "model", inner, this_param_fields))
            continue

        # Scalar Form/Query/string field (URI parameters etc.)
        if pname in used_field_names:
            logger.warning(
                "scalar param %s in route %s collides with an embedded model field",
                pname,
                route.path,
            )
            param_plan.append((pname, "scalar", None, []))
            continue
        clean_anno = _strip_annotated(annotation)
        default = param.default
        if default is inspect.Parameter.empty:
            fields[pname] = (clean_anno, ...)
        else:
            raw_default = getattr(default, "default", default)
            if raw_default is inspect.Parameter.empty:
                raw_default = None
            fields[pname] = (clean_anno, raw_default)
        used_field_names.add(pname)
        param_plan.append((pname, "scalar", None, [pname]))

    if not fields:
        return None

    model_name = f"{_route_short_name(route.path).title().replace('_', '')}Input"
    input_model = create_model(model_name, **fields)  # type: ignore[call-overload]
    return _RouteSchema(input_model, param_plan, skipped_files)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def _register_submit_tool(
    mcp: FastMCP, app: FastAPI, route: APIRoute, adapter: JobAdapter
) -> Optional[str]:
    """Register one `submit_<short>` tool for the given POST route. Returns tool name or None.

    The tool calls the FastAPI route handler directly with reconstructed kwargs
    rather than going through HTTP — FastAPI's multipart parsing of
    `Annotated[Model, Form()]` mixed with sibling Form params expects a nested
    `params` object that's not portable to encode from MCP-side, so we sidestep
    the wire format entirely.
    """
    schema = _build_route_schema(route)
    if schema is None:
        logger.warning("skipping %s — no introspectable params", route.path)
        return None

    short = _route_short_name(route.path)
    tool_name = f"submit_{adapter.name}_{short}".replace("-", "_")
    doc = (route.endpoint.__doc__ or "").strip() or f"Submit a {short} job to {adapter.name}."
    file_hint = (
        f" Files ({', '.join(schema.skipped_files)}) must be supplied via URI "
        f"inputs (`job://<id>/<file>` / `file://...` / `oss://...`); the MCP "
        f"transport does not stream binary bytes."
        if schema.skipped_files
        else ""
    )

    endpoint = route.endpoint
    endpoint_is_async = inspect.iscoroutinefunction(endpoint)

    async def submit(body):  # type: ignore[no-untyped-def]
        kwargs = schema.build_kwargs(body)
        if endpoint_is_async:
            result = await endpoint(**kwargs)
        else:
            # Run sync endpoint in a thread so we don't block the event loop —
            # the runner.submit it calls is fast but still does file I/O.
            import asyncio
            result = await asyncio.to_thread(endpoint, **kwargs)
        # JobInfo / pydantic models -> dict; passthrough for plain dicts.
        if isinstance(result, BaseModel):
            return result.model_dump(mode="json")
        return result

    submit.__annotations__ = {"body": schema.input_model, "return": dict[str, Any]}
    submit.__doc__ = f"{doc}{file_hint}"
    submit.__name__ = tool_name
    mcp.add_tool(submit, name=tool_name, description=submit.__doc__)
    return tool_name


def _register_lifecycle_tools(mcp: FastMCP, app: FastAPI, adapter: JobAdapter) -> list[str]:
    """Register the 4 read-side lifecycle tools. Names are prefixed by adapter so
    multi-service stdio sessions can host more than one service without colliding."""
    prefix = adapter.name.replace("-", "_")
    names: list[str] = []

    async def _get(path: str) -> dict[str, Any]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
            r = await client.get(path)
        r.raise_for_status()
        return r.json()

    async def get_job_status(job_id: str) -> dict[str, Any]:
        """Return current JobInfo for a job (status, progress, error info)."""
        return await _get(f"/api/jobs/{job_id}")

    async def list_job_files(job_id: str) -> dict[str, Any]:
        """List all files produced under the job's output directory."""
        return await _get(f"/api/jobs/{job_id}/files")

    async def get_job_log(job_id: str) -> dict[str, Any]:
        """Return the full subprocess log for the job (tee'd stdout+stderr)."""
        return await _get(f"/api/jobs/{job_id}/log")

    async def download_job_file(job_id: str, file_path: str) -> dict[str, Any]:
        """Fetch a single output file. Returns {'content': str, 'encoding': 'text'|'base64'}."""
        import base64

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
            r = await client.get(f"/api/jobs/{job_id}/file/{file_path}")
        r.raise_for_status()
        body = r.content
        try:
            return {"content": body.decode("utf-8"), "encoding": "text"}
        except UnicodeDecodeError:
            return {"content": base64.b64encode(body).decode("ascii"), "encoding": "base64"}

    for fn, label in (
        (get_job_status, "get_job_status"),
        (list_job_files, "list_job_files"),
        (get_job_log, "get_job_log"),
        (download_job_file, "download_job_file"),
    ):
        name = f"{prefix}_{label}"
        fn.__name__ = name
        mcp.add_tool(fn, name=name, description=fn.__doc__)
        names.append(name)
    return names


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def make_mcp_server(
    app: FastAPI,
    adapter: JobAdapter,
    settings: ServiceSettings,
    *,
    name: str | None = None,
) -> FastMCP:
    """Build an MCP server that mirrors the service's HTTP surface.

    The returned `FastMCP` instance can be:
      * mounted as a sub-app: `app.mount('/mcp', mcp.streamable_http_app())`
      * run over stdio: `await mcp.run_stdio_async()`
      * run over SSE (legacy): `await mcp.run_sse_async()`
    """
    server_name = name or f"{adapter.name}-server"
    instructions = (
        f"MCP interface for the bioagent {adapter.name!r} service. "
        f"Use `submit_{adapter.name.replace('-', '_')}_<endpoint>` tools to start "
        f"jobs, then poll `{adapter.name.replace('-', '_')}_get_job_status` until "
        f"`status` ∈ {{completed, failed}}. Outputs land under "
        f"`{settings.jobs_base_dir}/<job_id>/output/`; download individual files "
        f"with `{adapter.name.replace('-', '_')}_download_job_file`. Initial file "
        f"inputs must be referenced via URI (`job://`, `file://`, `oss://`) — the "
        f"MCP transport does not stream binary bytes; use the HTTP `/api/...` "
        f"endpoints for byte uploads."
    )
    mcp = FastMCP(server_name, instructions=instructions)

    # Lifecycle tools first (always present even if no POST routes are
    # introspectable — agents still need them to recover existing jobs).
    lifecycle = _register_lifecycle_tools(mcp, app, adapter)

    # Auto-register submit_* tools for each service-defined POST route.
    submitted: list[str] = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if _is_framework_path(route.path):
            continue
        if "POST" not in route.methods:
            continue
        try:
            tool_name = _register_submit_tool(mcp, app, route, adapter)
        except Exception:  # noqa: BLE001 — spike-level resilience
            logger.exception("failed to register MCP tool for %s", route.path)
            continue
        if tool_name:
            submitted.append(tool_name)

    logger.info(
        "MCP server %r registered %d submit tools (%s) + %d lifecycle tools (%s)",
        server_name,
        len(submitted),
        ", ".join(submitted) or "<none>",
        len(lifecycle),
        ", ".join(lifecycle),
    )
    return mcp


__all__ = ["make_mcp_server"]

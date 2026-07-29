"""`create_app` factory — the one-liner each service uses to assemble its FastAPI app."""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from bioq_service.adapter import JobAdapter
from bioq_service.errors import ServiceBusyError
from bioq_service.jobs import JobStore, reload_from_disk
from bioq_service.manifest import make_manifest_router
from bioq_service.routes import make_generic_router
from bioq_service.runner import JobRunner
from bioq_service.settings import ServiceSettings

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def _start_keepalive(
    runner: JobRunner, port: int, interval_s: int,
    keepalive_url: str | None = None,
) -> threading.Event:
    """Spawn a daemon thread that pings ``/healthz`` while jobs are active.

    FC determines instance idleness by HTTP request activity.  Background
    subprocesses are invisible to FC, so an instance running a long GPU job
    but receiving no poll requests will be reclaimed.  This thread prevents
    that by generating a lightweight HTTP request every *interval_s* seconds
    whenever ``runner.active_job_count > 0``.

    When *keepalive_url* is set the ping goes through FC's gateway (counted
    as real activity).  Otherwise falls back to localhost (works for local
    dev but invisible to FC).

    Returns a :class:`threading.Event` — set it to stop the thread.
    """
    stop = threading.Event()
    local_url = f"http://127.0.0.1:{port}/healthz"
    external_url = keepalive_url.rstrip("/") + "/healthz" if keepalive_url else None

    def _loop() -> None:
        while not stop.wait(interval_s):
            if runner.active_job_count > 0:
                if external_url:
                    try:
                        urllib.request.urlopen(external_url, timeout=10)
                    except Exception:
                        logger.debug("external keepalive failed, falling back to localhost")
                        try:
                            urllib.request.urlopen(local_url, timeout=5)
                        except Exception:
                            pass
                else:
                    try:
                        urllib.request.urlopen(local_url, timeout=5)
                    except Exception:
                        pass

    t = threading.Thread(target=_loop, daemon=True, name="fc-keepalive")
    t.start()
    logger.info(
        "FC keepalive thread started (interval=%ds, external=%s, local=%s)",
        interval_s, external_url or "(none)", local_url,
    )
    return stop


class _SessionAffinityMiddleware(BaseHTTPMiddleware):
    """Inject a session header into POST responses that carry a ``job_id``.

    FC HeaderField affinity reads this header to bind follow-up requests
    (poll, download) to the same instance that owns the job.  When
    ``header_name`` is *None* (local dev, Slurm), the middleware is a no-op.
    """

    def __init__(self, app: FastAPI, header_name: str) -> None:  # type: ignore[override]
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        if request.method != "POST" or response.status_code != 200:
            return response
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        try:
            job_id = json.loads(body).get("job_id")
        except (json.JSONDecodeError, AttributeError):
            job_id = None
        new_response = Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
        if job_id:
            new_response.headers[self.header_name] = str(job_id)
        return new_response


def create_app(
    adapter: JobAdapter,
    settings: ServiceSettings,
    *,
    title: str | None = None,
    version: str = "0.1.0",
    reload_jobs: bool = True,
) -> FastAPI:
    """Build a FastAPI app pre-wired with the framework's generic endpoints.

    Stores `adapter`, `settings`, `job_store`, `runner` on `app.state` so
    service-side routes can reach them via `request.app.state.*`.

    The caller is expected to:
      * Add service-specific POST routes that call `app.state.runner.submit(...)`
      * Ensure `settings.jobs_base_dir` is writable in the deployment environment
      * (Optional) Call `attach_mcp(app)` after the POST routes are registered
        to mount a Model Context Protocol server at `/mcp` mirroring the HTTP
        surface (one MCP tool per POST endpoint + 4 read-side lifecycle tools).

    If `reload_jobs=True` (default), `jobs_base_dir` is scanned at startup and
    any `<job_id>/job.json` sidecars are rehydrated into the store. Jobs that
    were RUNNING at the moment of restart are downgraded to FAILED with
    `failure_kind=INTERRUPTED`. Set False only for tests that need an empty store
    even when the tmp dir is reused.
    """
    if not title:
        title = f"{adapter.name}-server"

    app = FastAPI(title=title, version=version)

    if settings.session_header_name:
        app.add_middleware(_SessionAffinityMiddleware, header_name=settings.session_header_name)
        logger.info("Session affinity middleware enabled (header=%s)", settings.session_header_name)

    # Ensure the jobs root exists before we try to scan it (and so /healthz/detail
    # is accurate on the first request).
    settings.jobs_base_dir.mkdir(parents=True, exist_ok=True)

    # Persist sidecars under jobs_base_dir so the store survives restarts; see
    # `JobStore._persist` and `reload_from_disk`.
    store = JobStore(persist_dir=settings.jobs_base_dir)
    logger.info("instance_id=%s", store.instance_id)
    executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs)
    runner = JobRunner(store=store, executor=executor, settings=settings, adapter=adapter)

    app.state.adapter = adapter
    app.state.settings = settings
    app.state.job_store = store
    app.state.executor = executor
    app.state.runner = runner

    if settings.keepalive_interval_s > 0:
        stop_keepalive = _start_keepalive(
            runner, settings.port, settings.keepalive_interval_s,
            keepalive_url=settings.keepalive_url,
        )
        app.state.keepalive_stop = stop_keepalive

    if reload_jobs:
        n_restored = reload_from_disk(store, adapter, settings.jobs_base_dir)
        if n_restored:
            logger.info("recovered %d job(s) from %s", n_restored, settings.jobs_base_dir)

    @app.exception_handler(ServiceBusyError)
    async def _busy_handler(request: Request, exc: ServiceBusyError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "detail": str(exc),
                "active_jobs": exc.active,
                "max_concurrent_jobs": exc.limit,
            },
            headers={"Retry-After": "30"},
        )

    app.include_router(make_generic_router())
    app.include_router(make_manifest_router())

    return app


def attach_mcp(
    app: FastAPI,
    *,
    mount_path: str = "/mcp",
    server_name: str | None = None,
) -> "FastMCP | None":
    """Mount a Streamable-HTTP MCP server on `app` mirroring its HTTP surface.

    Call this AFTER all service-specific POST routes have been registered; the
    auto-discovery walks `app.routes` once at mount time.

    Produces an MCP tool per POST endpoint (`submit_<service>_<short>`) plus
    four read-side lifecycle tools (`<service>_get_job_status`,
    `<service>_list_job_files`, `<service>_get_job_log`,
    `<service>_download_job_file`). The mounted Starlette sub-app exposes the
    standard MCP Streamable-HTTP transport at `<mount_path>/mcp`.

    The MCP session manager is started/stopped via FastAPI startup/shutdown
    events; the server is stashed on `app.state.mcp` so the stdio CLI
    entrypoint (`bioq-service-mcp-stdio`) can reuse the same instance.
    """
    # MCP is an optional extra (`bioq-service-framework[mcp]`). If it isn't
    # installed — or the installed `mcp` package is an incompatible build (e.g.
    # a mirror resolved a name-colliding version lacking `mcp.server.fastmcp`) —
    # skip mounting rather than crashing the whole service: the core HTTP API
    # the gateway drives must still come up. Callers ignore the return value.
    try:
        from bioq_service.mcp_server import make_mcp_server  # late import — optional dep
    except ImportError as exc:
        logger.warning(
            "MCP unavailable (%s); serving HTTP API without the /mcp mount.", exc
        )
        return None

    mcp = make_mcp_server(
        app,
        app.state.adapter,
        app.state.settings,
        name=server_name,
    )

    # Materialize the streamable_http_app FIRST — `session_manager` is created
    # lazily by FastMCP and only after `streamable_http_app()` is called.
    mcp_asgi = mcp.streamable_http_app()

    # The streamable_http_app's lifespan IS the session manager; mounting it
    # under FastAPI does NOT auto-run that lifespan, so we hand-wire startup
    # and shutdown via FastAPI events. Cleaner than reconstructing the
    # FastAPI app with `lifespan=...`, and keeps `create_app`'s contract
    # unchanged.
    session_cm = mcp.session_manager.run()

    @app.on_event("startup")
    async def _start_mcp_session_manager() -> None:  # pragma: no cover
        await session_cm.__aenter__()

    @app.on_event("shutdown")
    async def _stop_mcp_session_manager() -> None:  # pragma: no cover
        await session_cm.__aexit__(None, None, None)

    app.mount(mount_path, mcp_asgi)
    app.state.mcp = mcp
    logger.info("MCP server mounted at %s (server=%r)", mount_path, mcp.name)
    return mcp


__all__ = ["attach_mcp", "create_app"]

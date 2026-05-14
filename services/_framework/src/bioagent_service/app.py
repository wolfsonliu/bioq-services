"""`create_app` factory — the one-liner each service uses to assemble its FastAPI app."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from fastapi import FastAPI

from bioagent_service.adapter import JobAdapter
from bioagent_service.jobs import JobStore, reload_from_disk
from bioagent_service.manifest import make_manifest_router
from bioagent_service.routes import make_generic_router
from bioagent_service.runner import JobRunner
from bioagent_service.settings import ServiceSettings

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


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

    # Ensure the jobs root exists before we try to scan it (and so /health/detail
    # is accurate on the first request).
    settings.jobs_base_dir.mkdir(parents=True, exist_ok=True)

    # Persist sidecars under jobs_base_dir so the store survives restarts; see
    # `JobStore._persist` and `reload_from_disk`.
    store = JobStore(persist_dir=settings.jobs_base_dir)
    executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs)
    runner = JobRunner(store=store, executor=executor, settings=settings, adapter=adapter)

    app.state.adapter = adapter
    app.state.settings = settings
    app.state.job_store = store
    app.state.executor = executor
    app.state.runner = runner

    if reload_jobs:
        n_restored = reload_from_disk(store, adapter, settings.jobs_base_dir)
        if n_restored:
            logger.info("recovered %d job(s) from %s", n_restored, settings.jobs_base_dir)

    app.include_router(make_generic_router())
    app.include_router(make_manifest_router())

    return app


def attach_mcp(
    app: FastAPI,
    *,
    mount_path: str = "/mcp",
    server_name: str | None = None,
) -> "FastMCP":
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
    entrypoint (`bioagent-service-mcp-stdio`) can reuse the same instance.
    """
    from bioagent_service.mcp_server import make_mcp_server  # late import — optional dep

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

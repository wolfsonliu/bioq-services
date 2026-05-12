"""`create_app` factory — the one-liner each service uses to assemble its FastAPI app."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI

from bioagent_service.adapter import JobAdapter
from bioagent_service.jobs import JobStore, reload_from_disk
from bioagent_service.manifest import make_manifest_router
from bioagent_service.routes import make_generic_router
from bioagent_service.runner import JobRunner
from bioagent_service.settings import ServiceSettings

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


__all__ = ["create_app"]

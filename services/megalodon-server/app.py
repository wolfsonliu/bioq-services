"""FastAPI app for megalodon-server.

Exposes `/api/generate` (submit/poll) + `/api/tasks/generate` (FC async task
mode) + `/api/models` (registry). Job lifecycle endpoints (/healthz,
/api/jobs/*, /api/manifest, /openapi.json) come from
`bioq_service.create_app`.

Unconditional generation — no file uploads. Requests carry only pydantic
form fields.
"""

from __future__ import annotations

import logging
from pathlib import Path

from bioq_service import (
    JobInfo,
    attach_mcp,
    create_app,
    model_form_depends,
    read_version_file,
    register_task_endpoint,
)
from fastapi import Depends, Request

from .adapter import MegalodonAdapter
from .models import GenerateRequest
from .settings import MegalodonSettings
from .tools import generate_argv

logger = logging.getLogger(__name__)

settings = MegalodonSettings()
adapter = MegalodonAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="Megalodon Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — surface NAS-mounted checkpoint + statistics.
# ---------------------------------------------------------------------------


def _strip_route(router, path: str, method: str) -> None:
    router.routes = [
        r for r in router.routes
        if not (getattr(r, "path", None) == path
                and method in getattr(r, "methods", set()))
    ]
    for r in router.routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            _strip_route(inner, path, method)


_strip_route(app.router, "/healthz/detail", "GET")


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Report per-variant checkpoint + statistics availability.

    A variant is `ready` when its checkpoint AND statistics bundle are
    present. Service starts even when weights are missing — surfaced here so
    an agent can detect a misconfigured FC mount / unbound SIF.
    """
    models = settings.list_models()
    model_report = {
        m.name: {
            "dataset": m.dataset,
            "objective": m.objective,
            "ckpt": m.ckpt_present,
            "stats": m.stats_present,
            "ready": m.ready,
        }
        for m in models
    }
    weights_loaded = any(m.ready for m in models)
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": weights_loaded,
        "models": model_report,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


# ---------------------------------------------------------------------------
# /api/models — checkpoint + statistics registry.
# ---------------------------------------------------------------------------


@app.get("/api/models")
def list_models() -> dict:
    """List the 6 variants with NAS presence status."""
    return {
        "models": [
            {
                "name": m.name,
                "dataset": m.dataset,
                "objective": m.objective,
                "ckpt": str(m.ckpt_path),
                "ckpt_present": m.ckpt_present,
                "stats_dir": str(m.stats_dir),
                "stats_present": m.stats_present,
                "ready": m.ready,
            }
            for m in settings.list_models()
        ]
    }


# ---------------------------------------------------------------------------
# /api/generate (submit/poll)
# ---------------------------------------------------------------------------


@app.post("/api/generate", response_model=JobInfo)
def post_generate(
    params: GenerateRequest = Depends(model_form_depends(GenerateRequest)),
) -> JobInfo:
    """Run unconditional generation. Returns a JobInfo; poll until completed."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return generate_argv(params, job_dir=job_dir, settings=settings)

    return app.state.runner.submit(
        build_argv=_build,
        label="generate",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# /api/tasks/generate (FC async task mode)
# ---------------------------------------------------------------------------


def _task_build(
    req: GenerateRequest, _job_id: str, job_dir: Path,
) -> list[str]:
    return generate_argv(req, job_dir=job_dir, settings=settings)


register_task_endpoint(
    app,
    path="/api/tasks/generate",
    label="generate",
    request_model=GenerateRequest,
    build_argv=_task_build,
    summary="Unconditional generation (single atomic task).",
)


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)

"""FastAPI app for flowmol-server.

Exposes `/api/generate` (submit/poll) + `/api/tasks/generate` (FC async
task mode).  Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest,
/openapi.json) come from `bioq_service.create_app`.

Unconditional generation — no file uploads.  Requests carry only pydantic
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

from .adapter import PRIMARY_VARIANTS, FlowMolAdapter
from .models import GenerateRequest
from .settings import FlowMolSettings
from .tools import generate_argv

logger = logging.getLogger(__name__)

settings = FlowMolSettings()
adapter = FlowMolAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="FlowMol Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — surface NAS-mounted weight presence.
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


def _variant_files(variant: str) -> dict[str, Path]:
    """Expected files under `<weights_dir>/trained_models/<variant>/`."""
    root = settings.weights_dir / "trained_models" / variant
    return {
        f"{variant}_ckpt": root / "checkpoints" / "last.ckpt",
        f"{variant}_config": root / "config.yaml",
    }


def _list_staged_variants() -> list[str]:
    """Scan NAS for variants that have both ckpt + config."""
    root = settings.weights_dir / "trained_models"
    if not root.is_dir():
        return []
    staged = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "checkpoints" / "last.ckpt").exists() and (d / "config.yaml").exists():
            staged.append(d.name)
    return staged


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Report primary-variant checkpoint availability + full staged list.

    Weights live at `<FLOWMOL_WEIGHTS_DIR>/trained_models/<variant>/`.
    Service starts even when weights are missing — the failure is surfaced
    here so an agent can detect a misconfigured FC mount / unbound SIF
    without crashing imports.
    """
    expected: dict[str, Path] = {}
    for v in PRIMARY_VARIANTS:
        expected.update(_variant_files(v))
    missing = {k: str(p) for k, p in expected.items() if not p.exists()}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": not missing,
        "weights_missing": missing,
        "primary_variants": list(PRIMARY_VARIANTS),
        "staged_variants": _list_staged_variants(),
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
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
)


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)

"""FastAPI app for openbpmd-server.

Exposes `/api/score` (submit/poll) + `/api/tasks/score` (FC async task mode).
Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest, /openapi.json)
come from `bioagent_service.create_app`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from bioagent_service import (
    JobInfo,
    attach_mcp,
    create_app,
    execute_task,
    model_form_depends,
    read_version_file,
    resolve_task_id,
)
from fastapi import Depends, File, Form, Header, Request, UploadFile

from .adapter import OpenBPMDAdapter
from .models import ScoreRequest
from .settings import OpenBPMDSettings
from .tools import score_argv
from bioagent_service.uris import resolve_input

logger = logging.getLogger(__name__)

settings = OpenBPMDSettings()
adapter = OpenBPMDAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="OpenBPMD Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — surface OpenMM version + CUDA platform presence.
# ---------------------------------------------------------------------------
# FastAPI >=0.115 wraps included routers in `_IncludedRouter`; recurse to drop
# the framework's generic /healthz/detail before ours runs.


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


def _probe_openmm() -> tuple[Optional[str], list[str]]:
    try:
        import openmm
        version = openmm.version.version
        platforms = [
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        ]
        return version, platforms
    except Exception:  # noqa: BLE001 — probe must never crash the endpoint
        return None, []


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Probe runtime deps.

    OpenBPMD has no NN weights; instead we surface the OpenMM version and the
    available compute platforms.  `cuda_available` is the real readiness
    signal for a GPU deployment; `weights_loaded` is retained for
    cross-service uniformity.
    """
    version, platforms = _probe_openmm()
    cuda_ok = "CUDA" in platforms
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "openmm_version": version,
        "platforms": platforms,
        "configured_platform": settings.platform,
        "cuda_available": cuda_ok,
        # Retained for cross-service uniformity — OpenBPMD has no weights.
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": True,
        "weights_missing": {},
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "task_endpoints_enabled": settings.task_endpoints_enabled,
    }


# ---------------------------------------------------------------------------
# Shared input staging
# ---------------------------------------------------------------------------


def _stage_inputs(
    input_dir: Path,
    *,
    structure: Optional[UploadFile],
    structure_uri: Optional[str],
    parameters: Optional[UploadFile],
    parameters_uri: Optional[str],
) -> tuple[Path, Path]:
    """Persist both required inputs under `<job_dir>/input/`."""
    input_dir.mkdir(parents=True, exist_ok=True)

    structure_dest = input_dir / (
        structure.filename if structure and structure.filename else "solvated.rst7"
    )
    structure_path = resolve_input(
        structure, structure_uri, structure_dest, settings, field_name="structure",
    )

    parameters_dest = input_dir / (
        parameters.filename if parameters and parameters.filename else "solvated.prm7"
    )
    parameters_path = resolve_input(
        parameters, parameters_uri, parameters_dest, settings,
        field_name="parameters",
    )
    return structure_path, parameters_path


# ---------------------------------------------------------------------------
# /api/score (submit/poll)
# ---------------------------------------------------------------------------


@app.post("/api/score", response_model=JobInfo)
def post_score(
    params: ScoreRequest = Depends(model_form_depends(ScoreRequest)),
    structure: Optional[UploadFile] = File(None),
    structure_uri: Optional[str] = Form(None),
    parameters: Optional[UploadFile] = File(None),
    parameters_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Score a binding pose via metadynamics. Returns a JobInfo; poll until done."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        structure_path, parameters_path = _stage_inputs(
            job_dir / "input",
            structure=structure, structure_uri=structure_uri,
            parameters=parameters, parameters_uri=parameters_uri,
        )
        return score_argv(
            params,
            job_dir=job_dir,
            structure=structure_path,
            parameters=parameters_path,
            settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="score",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# /api/tasks/score (FC async task mode)
# ---------------------------------------------------------------------------


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/score", response_model=JobInfo)
    def post_score_task(
        request: Request,
        params: ScoreRequest = Depends(model_form_depends(ScoreRequest)),
        structure: Optional[UploadFile] = File(None),
        structure_uri: Optional[str] = Form(None),
        parameters: Optional[UploadFile] = File(None),
        parameters_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"
        ),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"
        ),
    ) -> JobInfo:
        """FC async task mode — blocks until done, HTTP request returns 202."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req: ScoreRequest, input_dir: Path) -> None:
            structure_path, parameters_path = _stage_inputs(
                input_dir,
                structure=structure, structure_uri=structure_uri,
                parameters=parameters, parameters_uri=parameters_uri,
            )
            paths["structure"] = structure_path
            paths["parameters"] = parameters_path

        def _build(req: ScoreRequest, _job_id: str, job_dir: Path) -> list[str]:
            return score_argv(
                req,
                job_dir=job_dir,
                structure=paths["structure"],
                parameters=paths["parameters"],
                settings=settings,
            )

        return execute_task(
            request,
            job_id=job_id,
            label="score",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)

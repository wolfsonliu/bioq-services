"""FastAPI app for plip-server.

Exposes a single `/api/profile` endpoint (+ `/api/tasks/profile` async variant)
that profiles the non-covalent interactions in one PDB complex. Job lifecycle
endpoints (/healthz, /api/jobs/*, /api/manifest, /openapi.json) come from
`bioagent_service.create_app`.
"""

from __future__ import annotations

import importlib.util
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
from bioagent_service.uris import resolve_input
from fastapi import Depends, File, Form, Header, Request, UploadFile

from .adapter import PlipAdapter
from .models import ProfileRequest
from .settings import PlipSettings
from .tools import profile_argv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = PlipSettings()
adapter = PlipAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="PLIP Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---- /healthz/detail override: report upstream + openbabel/pymol readiness ----

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


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Extended health. PLIP has no model weights, so this probes readiness of
    the vendored upstream source and the algorithm dependencies (openbabel is
    required always; pymol only for -y/-p). Missing pieces do not crash the
    service; they surface as ready=false so an agent can detect a broken image.
    """
    plip_cmd = settings.upstream_dir / "plip" / "plipcmd.py"
    checks = {
        "upstream_dir": settings.upstream_dir.is_dir(),
        "plipcmd": plip_cmd.exists(),
        "openbabel": _module_available("openbabel"),
        "pymol": _module_available("pymol"),
    }
    missing = {k: str(k) for k, ok in checks.items() if not ok}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "upstream_dir": str(settings.upstream_dir),
        "checks": checks,
        # openbabel + upstream are required; pymol only gates visualization.
        "ready": checks["upstream_dir"] and checks["plipcmd"] and checks["openbabel"],
        "pymol_available": checks["pymol"],
        "missing": missing,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


# ---- profile ----

@app.post("/api/profile", response_model=JobInfo)
def post_profile(
    params: ProfileRequest = Depends(model_form_depends(ProfileRequest)),
    input_pdb: Optional[UploadFile] = File(None),
    input_pdb_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Profile non-covalent interactions in one PDB complex."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        pdb_path = resolve_input(
            input_pdb, input_pdb_uri, job_dir / "input" / "input.pdb", settings,
            field_name="input_pdb",
        )
        return profile_argv(params, job_dir=job_dir, input_pdb=pdb_path, settings=settings)

    return app.state.runner.submit(
        build_argv=_build, label="profile",
        input_params=params.model_dump(mode="json"),
    )


# ---- task endpoint (FC async task mode) ----

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/profile", response_model=JobInfo)
    def post_profile_task(
        request: Request,
        params: ProfileRequest = Depends(model_form_depends(ProfileRequest)),
        input_pdb: Optional[UploadFile] = File(None),
        input_pdb_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """profile as a single atomic task (blocks until completion)."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["pdb"] = resolve_input(
                input_pdb, input_pdb_uri, input_dir / "input.pdb", settings,
                field_name="input_pdb",
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return profile_argv(req, job_dir=job_dir, input_pdb=paths["pdb"], settings=settings)

        return execute_task(
            request, job_id=job_id, label="profile", params=params,
            build_argv=_build, save_inputs=_save,
        )


# Mount MCP — after all POST routes so auto-discovery sees the full surface.
attach_mcp(app)

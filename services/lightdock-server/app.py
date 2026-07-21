"""FastAPI app for lightdock-server.

Exposes a single /api/dock endpoint (full LightDock GSO docking protocol) plus
its async task counterpart. Job lifecycle endpoints (/healthz, /api/jobs/*,
/api/manifest, /openapi.json) come from `bioq_service.create_app`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from bioq_service import (
    JobInfo,
    attach_mcp,
    create_app,
    execute_task,
    model_form_depends,
    read_version_file,
    resolve_task_id,
)
from bioq_service.uris import maybe_resolve_input, resolve_input
from fastapi import Depends, File, Form, Header, HTTPException, Request, UploadFile

from .adapter import LightdockAdapter
from .docking import list_scoring_functions, lightdock_version
from .models import DockRequest
from .settings import LightdockSettings
from .tools import dock_argv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = LightdockSettings()
adapter = LightdockAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="LightDock Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---- /healthz/detail override: report lightdock version + scoring functions ----

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
    """Extended health: installed LightDock version + available scoring functions.

    LightDock has no NN weights (scoring params ship in the package), so there
    is no `weights_loaded` field — the analogous readiness signal is that the
    package imports and its scoring functions are discoverable.
    """
    scoring = list_scoring_functions()
    version = lightdock_version()
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "lightdock_version": version,
        "lightdock_available": version is not None,
        "scoring_functions": scoring,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


@app.post("/api/dock", response_model=JobInfo)
def post_dock(
    params: DockRequest = Depends(model_form_depends(DockRequest)),
    receptor: Optional[UploadFile] = File(None),
    ligand: Optional[UploadFile] = File(None),
    restraints: Optional[UploadFile] = File(None),
    receptor_uri: Optional[str] = Form(None),
    ligand_uri: Optional[str] = Form(None),
    restraints_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Run the full LightDock docking protocol; returns ranked docked complexes."""
    if not receptor and not receptor_uri:
        raise HTTPException(422, "Provide `receptor` upload or `receptor_uri`.")
    if not ligand and not ligand_uri:
        raise HTTPException(422, "Provide `ligand` upload or `ligand_uri`.")

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        receptor_path = resolve_input(receptor, receptor_uri, input_dir / "receptor.pdb", settings)
        ligand_path = resolve_input(ligand, ligand_uri, input_dir / "ligand.pdb", settings)
        restraints_path = maybe_resolve_input(
            restraints, restraints_uri, input_dir / "restraints.list", settings
        )
        return dock_argv(
            params,
            job_dir=job_dir,
            receptor_path=receptor_path,
            ligand_path=ligand_path,
            restraints_path=restraints_path,
            settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build, label="dock",
        input_params=params.model_dump(mode="json"),
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/dock", response_model=JobInfo)
    def post_dock_task(
        request: Request,
        params: DockRequest = Depends(model_form_depends(DockRequest)),
        receptor: Optional[UploadFile] = File(None),
        ligand: Optional[UploadFile] = File(None),
        restraints: Optional[UploadFile] = File(None),
        receptor_uri: Optional[str] = Form(None),
        ligand_uri: Optional[str] = Form(None),
        restraints_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Run the full LightDock docking protocol as a single atomic task.

        Blocks until docking completes. For the submit/poll interface, use POST /api/dock.
        """
        if not receptor and not receptor_uri:
            raise HTTPException(422, "Provide `receptor` upload or `receptor_uri`.")
        if not ligand and not ligand_uri:
            raise HTTPException(422, "Provide `ligand` upload or `ligand_uri`.")

        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Optional[Path]] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["receptor"] = resolve_input(
                receptor, receptor_uri, input_dir / "receptor.pdb", settings
            )
            paths["ligand"] = resolve_input(
                ligand, ligand_uri, input_dir / "ligand.pdb", settings
            )
            paths["restraints"] = maybe_resolve_input(
                restraints, restraints_uri, input_dir / "restraints.list", settings
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return dock_argv(
                req,
                job_dir=job_dir,
                receptor_path=paths["receptor"],
                ligand_path=paths["ligand"],
                restraints_path=paths["restraints"],
                settings=settings,
            )

        return execute_task(
            request,
            job_id=job_id,
            label="dock",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)

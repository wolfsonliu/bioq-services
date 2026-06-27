"""FastAPI app for rfdiffusion-server.

Five POST endpoints, all driving `scripts/run_inference.py` with different
Hydra overrides:

  * `/api/generate/unconditional` — length-only monomer / macrocycle
  * `/api/generate/motif`         — motif scaffolding (input PDB required)
  * `/api/generate/binder`        — PPI binder design with hotspots
  * `/api/generate/symmetry`      — cyclic / dihedral / tetrahedral oligomers
  * `/api/generate`               — raw contig + freeform Hydra overrides

The framework supplies `/healthz`, `/api/manifest`, `/api/jobs/{id}`, file
listing/download/deletion, and the MCP tool surface (attached after the POST
routes are registered).
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
    register_task_endpoint,
    resolve_task_id,
)
from fastapi import Depends, File, Form, Header, Request, UploadFile

from .adapter import RFdiffusionAdapter
from .models import (
    BinderRequest,
    CustomRequest,
    MotifRequest,
    SymmetryRequest,
    UnconditionalRequest,
)
from .settings import RFdiffusionSettings
from .tools import (
    binder_argv,
    custom_argv,
    motif_argv,
    symmetry_argv,
    unconditional_argv,
)
from .uris import resolve_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = RFdiffusionSettings()
adapter = RFdiffusionAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="RFdiffusion Server",
    version=read_version_file(__file__, default="0.1.0"),
)


# Remove framework's generic /healthz/detail so our rfdiffusion-specific
# weights probe takes over. FastAPI >=0.115 wraps included routers in
# `_IncludedRouter`; descend to find the framework route.
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
    """Extended health: report whether NAS-mounted weights are reachable.

    RFdiffusion checkpoints live on NAS at `RFDIFFUSION_MODELS_DIR` (default
    `/data/models/rfdiffusion/models/`).  We probe for the directory and any
    *.pt files; `weights_loaded=false` lets the agent detect a misconfigured
    FC mount / unbound SIF without crashing the service.
    """
    models_dir = settings.models_dir
    if not models_dir.exists():
        weights_loaded = False
        ckpts_found = 0
    else:
        ckpts_found = sum(1 for _ in models_dir.glob("*.pt"))
        weights_loaded = ckpts_found > 0
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(models_dir),
        "weights_loaded": weights_loaded,
        "ckpts_found": ckpts_found,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


# ---------------------------------------------------------------------------
# Service-specific endpoints
# ---------------------------------------------------------------------------


@app.post("/api/generate/unconditional", response_model=JobInfo)
def generate_unconditional(
    params: UnconditionalRequest = Depends(model_form_depends(UnconditionalRequest)),
) -> JobInfo:
    """Unconditional monomer (or macrocycle with `cyclic=true`)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return unconditional_argv(params, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="unconditional",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/generate/motif", response_model=JobInfo)
def generate_motif(
    input_pdb: Optional[UploadFile] = File(
        None, description="Input PDB carrying the motif. Mutually exclusive with `input_uri`."
    ),
    input_uri: Optional[str] = Form(
        None,
        description=(
            "URI to fetch the input PDB instead of uploading. Schemes: "
            "job://<id>/<file>, file:///path, oss://<bucket>/<key>, http(s)://..."
        ),
    ),
    params: MotifRequest = Depends(model_form_depends(MotifRequest)),
) -> JobInfo:
    """Motif scaffolding — contig references chain+residue ranges in the input PDB."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        pdb_path = resolve_input(input_pdb, input_uri, job_dir / "input" / "motif.pdb", settings)
        return motif_argv(params, pdb_path, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="motif",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/generate/binder", response_model=JobInfo)
def generate_binder(
    input_pdb: Optional[UploadFile] = File(
        None, description="Target PDB. Mutually exclusive with `input_uri`."
    ),
    input_uri: Optional[str] = Form(None, description="URI to fetch the target PDB (see /motif)."),
    params: BinderRequest = Depends(model_form_depends(BinderRequest)),
) -> JobInfo:
    """PPI binder design vs a target PDB; supply hotspots for site control."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        pdb_path = resolve_input(input_pdb, input_uri, job_dir / "input" / "target.pdb", settings)
        return binder_argv(params, pdb_path, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="binder",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/generate/symmetry", response_model=JobInfo)
def generate_symmetry(
    params: SymmetryRequest = Depends(model_form_depends(SymmetryRequest)),
) -> JobInfo:
    """Symmetric oligomer (cyclic / dihedral / tetrahedral)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return symmetry_argv(params, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="symmetry",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/generate", response_model=JobInfo)
def generate_custom(
    input_pdb: Optional[UploadFile] = File(
        None, description="Optional input PDB (motif / partial diffusion). Or pass `input_uri`."
    ),
    input_uri: Optional[str] = Form(None, description="URI to fetch the input PDB (see /motif)."),
    params: CustomRequest = Depends(model_form_depends(CustomRequest)),
) -> JobInfo:
    """Raw contig + arbitrary Hydra overrides — partial diffusion, fold conditioning, ..."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        # input PDB is optional here; resolve only if the caller asked for one.
        pdb_path = None
        if input_pdb is not None or input_uri:
            pdb_path = resolve_input(input_pdb, input_uri, job_dir / "input" / "input.pdb", settings)
        return custom_argv(params, pdb_path, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="custom",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Task endpoints (synchronous; FC Async Task Mode-friendly)
# ---------------------------------------------------------------------------

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/generate/motif", response_model=JobInfo)
    def generate_motif_task(
        request: Request,
        input_pdb: Optional[UploadFile] = File(None),
        input_uri: Optional[str] = Form(None),
        params: MotifRequest = Depends(model_form_depends(MotifRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Motif scaffolding as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["pdb"] = resolve_input(input_pdb, input_uri, input_dir / "motif.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return motif_argv(req, paths["pdb"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="motif", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/generate/binder", response_model=JobInfo)
    def generate_binder_task(
        request: Request,
        input_pdb: Optional[UploadFile] = File(None),
        input_uri: Optional[str] = Form(None),
        params: BinderRequest = Depends(model_form_depends(BinderRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """PPI binder design as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["pdb"] = resolve_input(input_pdb, input_uri, input_dir / "target.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return binder_argv(req, paths["pdb"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="binder", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/generate", response_model=JobInfo)
    def generate_custom_task(
        request: Request,
        input_pdb: Optional[UploadFile] = File(None),
        input_uri: Optional[str] = Form(None),
        params: CustomRequest = Depends(model_form_depends(CustomRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Custom Hydra-override generation as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Optional[Path]] = {"pdb": None}

        def _save(_req, input_dir: Path) -> None:
            if input_pdb is not None or input_uri:
                paths["pdb"] = resolve_input(
                    input_pdb, input_uri, input_dir / "input.pdb", settings
                )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return custom_argv(req, paths["pdb"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="custom", params=params,
            build_argv=_build, save_inputs=_save,
        )


# No-upload endpoints — use the simpler register_task_endpoint helper.
# It internally honors settings.task_endpoints_enabled.

def _unconditional_build(req, _job_id: str, job_dir: Path) -> list[str]:
    return unconditional_argv(req, job_dir, settings)


def _symmetry_build(req, _job_id: str, job_dir: Path) -> list[str]:
    return symmetry_argv(req, job_dir, settings)


register_task_endpoint(
    app,
    path="/api/tasks/generate/unconditional",
    label="unconditional",
    request_model=UnconditionalRequest,
    build_argv=_unconditional_build,
)
register_task_endpoint(
    app,
    path="/api/tasks/generate/symmetry",
    label="symmetry",
    request_model=SymmetryRequest,
    build_argv=_symmetry_build,
)


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)

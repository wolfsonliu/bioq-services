"""FastAPI app for diffusion-hopping-server.

Exposes `/api/generate` (submit/poll) + `/api/tasks/generate` (FC async
task mode).  Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest,
/openapi.json) come from `bioagent_service.create_app`.
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

from .adapter import DiffusionHoppingAdapter
from .models import GenerateRequest
from .settings import DiffusionHoppingSettings
from .tools import generate_argv

logger = logging.getLogger(__name__)

settings = DiffusionHoppingSettings()
adapter = DiffusionHoppingAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="DiffHopp Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — surface NAS-mounted weight presence.
# ---------------------------------------------------------------------------
# FastAPI >=0.115 wraps included routers in `_IncludedRouter`; we recurse
# into them to drop the framework's generic /healthz/detail before our
# diffhopp-specific one runs.

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
    """Probe whether the 4 DiffHopp checkpoints are reachable on NAS.

    Weights live at `DIFFUSION_HOPPING_WEIGHTS_DIR` (default
    `/data/models/diffusion-hopping/checkpoints/`).  Service starts even
    when weights are missing — the failure is surfaced here so an agent can
    detect a misconfigured FC mount / unbound SIF without crashing imports.
    """
    expected = {
        v: settings.weights_dir / f"{v}.ckpt"
        for v in ("gvp_conditional", "gvp_unconditional",
                  "egnn_conditional", "egnn_unconditional")
    }
    missing = {k: str(p) for k, p in expected.items() if not p.exists()}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": not missing,
        "weights_missing": missing,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


# ---------------------------------------------------------------------------
# /api/generate (submit/poll)
# ---------------------------------------------------------------------------


def _save_inputs(
    protein: Optional[UploadFile],
    protein_uri: Optional[str],
    reference_ligand: Optional[UploadFile],
    reference_ligand_uri: Optional[str],
    input_dir: Path,
) -> tuple[Path, Path]:
    """Persist + URI-resolve the two required input files."""
    # Import here so test discovery isn't blocked by missing optional deps
    # when these closures aren't reached.
    from .uris import resolve_input

    input_dir.mkdir(parents=True, exist_ok=True)

    protein_dest = input_dir / (protein.filename if protein and protein.filename
                                else "protein.pdb")
    protein_path = resolve_input(protein, protein_uri, protein_dest, settings)

    ligand_dest = input_dir / (reference_ligand.filename
                               if reference_ligand and reference_ligand.filename
                               else "reference_ligand.sdf")
    ligand_path = resolve_input(
        reference_ligand, reference_ligand_uri, ligand_dest, settings,
    )
    return protein_path, ligand_path


@app.post("/api/generate", response_model=JobInfo)
def post_generate(
    params: GenerateRequest = Depends(model_form_depends(GenerateRequest)),
    protein: Optional[UploadFile] = File(None),
    protein_uri: Optional[str] = Form(None),
    reference_ligand: Optional[UploadFile] = File(None),
    reference_ligand_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Run scaffold-hopping generation. Returns a JobInfo; poll until completed."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        protein_path, ligand_path = _save_inputs(
            protein, protein_uri,
            reference_ligand, reference_ligand_uri,
            job_dir / "input",
        )
        return generate_argv(
            params,
            job_dir=job_dir,
            input_molecule=ligand_path,
            input_protein=protein_path,
            settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="generate",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# /api/tasks/generate (FC async task mode)
# ---------------------------------------------------------------------------


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/generate", response_model=JobInfo)
    def post_generate_task(
        request: Request,
        params: GenerateRequest = Depends(model_form_depends(GenerateRequest)),
        protein: Optional[UploadFile] = File(None),
        protein_uri: Optional[str] = Form(None),
        reference_ligand: Optional[UploadFile] = File(None),
        reference_ligand_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"
        ),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"
        ),
    ) -> JobInfo:
        """FC async task mode — blocks until done, HTTP request returns 202."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        # closure-shared dict bridges _save → _build
        paths: dict[str, Path] = {}

        def _save(_req: GenerateRequest, input_dir: Path) -> None:
            protein_path, ligand_path = _save_inputs(
                protein, protein_uri,
                reference_ligand, reference_ligand_uri,
                input_dir,
            )
            paths["protein"] = protein_path
            paths["ligand"] = ligand_path

        def _build(req: GenerateRequest, _job_id: str, job_dir: Path) -> list[str]:
            return generate_argv(
                req,
                job_dir=job_dir,
                input_molecule=paths["ligand"],
                input_protein=paths["protein"],
                settings=settings,
            )

        return execute_task(
            request,
            job_id=job_id,
            label="generate",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)

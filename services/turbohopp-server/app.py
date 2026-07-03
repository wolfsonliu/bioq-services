"""FastAPI app for turbohopp-server.

Exposes ``/api/generate`` (submit/poll) + ``/api/tasks/generate`` (FC async
task mode).  Job lifecycle endpoints (``/healthz``, ``/api/jobs/*``,
``/api/manifest``, ``/openapi.json``) come from ``bioagent_service.create_app``.

FC deployment notes:
  - Listen on 0.0.0.0:CAPort (default 9000)
  - Respond to /healthz within 120 s of start
  - Keep-alive >= 15 min for GPU consistency sampling
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

from .adapter import TurboHoppAdapter
from .models import GenerateRequest
from .settings import TurboHoppSettings
from .tools import generate_argv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = TurboHoppSettings()
adapter = TurboHoppAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="TurboHopp Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — surface NAS-mounted checkpoint presence.
# ---------------------------------------------------------------------------
# FastAPI >=0.115 wraps included routers in `_IncludedRouter`; recurse into
# them to drop the framework's generic /healthz/detail before ours runs.


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
    """Extended health: report whether NAS-mounted TurboHopp weights exist.

    The consistency-model .ckpt lives at ``TURBOHOPP_WEIGHTS_DIR`` (default
    ``/data/models/turbohopp/checkpoints/v1/``).  Upstream does not publish
    a public checkpoint — the deployer must supply one (train via
    upstream ``train_consistency.py`` or obtain from authors).

    ``weights_loaded=false`` lets the agent detect a "service alive but no
    model" state without crashing at import time.
    """
    wdir = settings.weights_dir
    if not wdir.exists():
        files_found, weights_loaded = 0, False
    else:
        files_found = sum(1 for p in wdir.rglob("*.ckpt") if p.is_file())
        weights_loaded = files_found > 0
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(wdir),
        "weights_loaded": weights_loaded,
        "files_found": files_found,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


# ---------------------------------------------------------------------------
# Shared: URI + upload resolution for the two input files.
# ---------------------------------------------------------------------------


def _save_inputs(
    protein: Optional[UploadFile],
    protein_uri: Optional[str],
    reference_ligand: Optional[UploadFile],
    reference_ligand_uri: Optional[str],
    input_dir: Path,
) -> tuple[Path, Path]:
    """Persist + URI-resolve the two required input files.

    Imports done inline so test discovery isn't blocked by missing
    optional deps when these closures aren't reached.
    """
    from .uris import resolve_input

    input_dir.mkdir(parents=True, exist_ok=True)

    protein_dest = input_dir / (
        protein.filename if protein and protein.filename else "protein.pdb"
    )
    protein_path = resolve_input(protein, protein_uri, protein_dest, settings)

    ligand_dest = input_dir / (
        reference_ligand.filename
        if reference_ligand and reference_ligand.filename
        else "reference_ligand.sdf"
    )
    ligand_path = resolve_input(
        reference_ligand, reference_ligand_uri, ligand_dest, settings,
    )
    return protein_path, ligand_path


# ---------------------------------------------------------------------------
# /api/generate (submit/poll)
# ---------------------------------------------------------------------------


@app.post("/api/generate", response_model=JobInfo)
def post_generate(
    params: GenerateRequest = Depends(model_form_depends(GenerateRequest)),
    protein: Optional[UploadFile] = File(
        None, description="Protein pocket PDB. Mutually exclusive with `protein_uri`.",
    ),
    protein_uri: Optional[str] = Form(
        None,
        description=(
            "URI to fetch the protein pocket instead of uploading. Schemes: "
            "job://<id>/<file>, file:///path, oss://<bucket>/<key>, http(s)://..."
        ),
    ),
    reference_ligand: Optional[UploadFile] = File(
        None,
        description=(
            "Reference ligand (.sdf / .mol2 / .pdb). Mutually exclusive with "
            "`reference_ligand_uri`."
        ),
    ),
    reference_ligand_uri: Optional[str] = Form(
        None, description="URI to fetch the reference ligand instead of uploading.",
    ),
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
            input_protein=protein_path,
            input_molecule=ligand_path,
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
            default=None, alias="X-Bioagent-Job-Id",
        ),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id",
        ),
    ) -> JobInfo:
        """FC async task mode — blocks until done, HTTP returns 202."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
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
                input_protein=paths["protein"],
                input_molecule=paths["ligand"],
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


# Mount MCP server — AFTER all POST routes so auto-discovery sees the full surface.
attach_mcp(app)

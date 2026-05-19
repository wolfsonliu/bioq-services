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

from bioagent_service import JobInfo, attach_mcp, create_app, model_form_depends, read_version_file
from fastapi import Depends, File, Form, UploadFile

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

    return app.state.runner.submit(build_argv=_build, label="unconditional")


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

    return app.state.runner.submit(build_argv=_build, label="motif")


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

    return app.state.runner.submit(build_argv=_build, label="binder")


@app.post("/api/generate/symmetry", response_model=JobInfo)
def generate_symmetry(
    params: SymmetryRequest = Depends(model_form_depends(SymmetryRequest)),
) -> JobInfo:
    """Symmetric oligomer (cyclic / dihedral / tetrahedral)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return symmetry_argv(params, job_dir, settings)

    return app.state.runner.submit(build_argv=_build, label="symmetry")


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

    return app.state.runner.submit(build_argv=_build, label="custom")


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)

"""FastAPI app for rfdiffusion2-server.

Three POST endpoints, all driving `rf_diffusion/run_inference.py` with
different Hydra overrides:

  * `/api/generate/active_site`           — atomic motif + ligand scaffolding
  * `/api/generate/small_molecule_binder` — RASA-conditioned binder design
  * `/api/generate`                       — raw contig + freeform Hydra overrides

The framework supplies `/healthz`, `/api/manifest`, `/api/jobs/{id}`, file
listing/download/deletion, and the MCP tool surface (attached after the POST
routes are registered).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from bioagent_service import JobInfo, attach_mcp, create_app, model_form_depends
from fastapi import Depends, File, Form, UploadFile

from .adapter import RFdiffusion2Adapter
from .models import (
    ActiveSiteRequest,
    CustomRequest,
    SmallMoleculeBinderRequest,
)
from .settings import RFdiffusion2Settings
from .tools import (
    active_site_argv,
    custom_argv,
    small_molecule_binder_argv,
)
from .uris import resolve_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = RFdiffusion2Settings()
adapter = RFdiffusion2Adapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="RFdiffusion2 Server",
    version="0.0.1",
)


# ---------------------------------------------------------------------------
# Service-specific endpoints
# ---------------------------------------------------------------------------


@app.post("/api/generate/active_site", response_model=JobInfo)
def generate_active_site(
    input_pdb: Optional[UploadFile] = File(
        None,
        description="Input PDB carrying the motif + ligand. Mutually exclusive with `input_uri`.",
    ),
    input_uri: Optional[str] = Form(
        None,
        description=(
            "URI to fetch the input PDB instead of uploading. Schemes: "
            "job://<id>/<file>, file:///path, oss://<bucket>/<key>, http(s)://..."
        ),
    ),
    params: ActiveSiteRequest = Depends(model_form_depends(ActiveSiteRequest)),
) -> JobInfo:
    """Active-site scaffolding around an atomic motif + ligand."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        pdb_path = resolve_input(input_pdb, input_uri, job_dir / "input" / "motif.pdb", settings)
        return active_site_argv(params, pdb_path, job_dir, settings)

    return app.state.runner.submit(build_argv=_build, label="active_site")


@app.post("/api/generate/small_molecule_binder", response_model=JobInfo)
def generate_small_molecule_binder(
    input_pdb: Optional[UploadFile] = File(
        None,
        description="Input PDB carrying the small molecule. Mutually exclusive with `input_uri`.",
    ),
    input_uri: Optional[str] = Form(None, description="URI to fetch the input PDB (see /active_site)."),
    params: SmallMoleculeBinderRequest = Depends(model_form_depends(SmallMoleculeBinderRequest)),
) -> JobInfo:
    """Small-molecule binder design, optionally RASA-conditioned."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        pdb_path = resolve_input(input_pdb, input_uri, job_dir / "input" / "ligand.pdb", settings)
        return small_molecule_binder_argv(params, pdb_path, job_dir, settings)

    return app.state.runner.submit(build_argv=_build, label="small_molecule_binder")


@app.post("/api/generate", response_model=JobInfo)
def generate_custom(
    input_pdb: Optional[UploadFile] = File(
        None, description="Optional input PDB. Required iff `input_pdb_required=true`."
    ),
    input_uri: Optional[str] = Form(None, description="URI to fetch the input PDB."),
    params: CustomRequest = Depends(model_form_depends(CustomRequest)),
) -> JobInfo:
    """Raw contig + freeform Hydra overrides — any config under config/inference/."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        pdb_path = None
        if input_pdb is not None or input_uri:
            pdb_path = resolve_input(input_pdb, input_uri, job_dir / "input" / "input.pdb", settings)
        return custom_argv(params, pdb_path, job_dir, settings)

    return app.state.runner.submit(build_argv=_build, label="custom")


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)

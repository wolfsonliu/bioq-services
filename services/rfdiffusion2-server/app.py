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
from bioagent_service.uris import resolve_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = RFdiffusion2Settings()
adapter = RFdiffusion2Adapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="RFdiffusion2 Server",
    version=read_version_file(__file__, default="0.0.1"),
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

    return app.state.runner.submit(
        build_argv=_build, label="active_site",
        input_params=params.model_dump(mode="json"),
    )


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

    return app.state.runner.submit(
        build_argv=_build, label="small_molecule_binder",
        input_params=params.model_dump(mode="json"),
    )


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

    return app.state.runner.submit(
        build_argv=_build, label="custom",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Task endpoints (synchronous; FC Async Task Mode-friendly)
# ---------------------------------------------------------------------------

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/generate/active_site", response_model=JobInfo)
    def generate_active_site_task(
        request: Request,
        input_pdb: Optional[UploadFile] = File(None),
        input_uri: Optional[str] = Form(None),
        params: ActiveSiteRequest = Depends(model_form_depends(ActiveSiteRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Active-site scaffolding as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["pdb"] = resolve_input(input_pdb, input_uri, input_dir / "motif.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return active_site_argv(req, paths["pdb"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="active_site", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/generate/small_molecule_binder", response_model=JobInfo)
    def generate_small_molecule_binder_task(
        request: Request,
        input_pdb: Optional[UploadFile] = File(None),
        input_uri: Optional[str] = Form(None),
        params: SmallMoleculeBinderRequest = Depends(model_form_depends(SmallMoleculeBinderRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Small-molecule binder design as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["pdb"] = resolve_input(input_pdb, input_uri, input_dir / "ligand.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return small_molecule_binder_argv(req, paths["pdb"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="small_molecule_binder", params=params,
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


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)

"""FastAPI app for rfantibody-server.

Exposes three single-tool endpoints (`/api/rfdiffusion`, `/api/proteinmpnn`,
`/api/rf2`). Pipeline-style orchestration belongs to clients — shared NAS makes
`job://<id>/<file>` cheap enough that chaining outputs is no worse than a local
script.

The framework handles `/healthz`, `/api/jobs/{id}`, `/files`, `/log`, `/download`,
`/file/{path}`, and `DELETE /api/jobs/{id}`. We only register the
service-specific POST routes here.

FC deployment notes:
  - Listen on 0.0.0.0:CAPort (default 9000)
  - Respond to /healthz within 120 s of start
  - Keep-alive must be >= 15 min for long-running RF2 jobs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from bioagent_service import JobInfo, attach_mcp, create_app, model_form_depends, read_version_file
from fastapi import Depends, File, Form, UploadFile

from .adapter import RFantibodyAdapter
from .models import ProteinMPNNRequest, RF2Request, RFdiffusionRequest
from .settings import RFantibodySettings
from .tools import proteinmpnn_argv, rf2_argv, rfdiffusion_argv
from .uris import resolve_input, save_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = RFantibodySettings()
adapter = RFantibodyAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="RFantibody Server",
    version=read_version_file(__file__, default="0.2.0"),
)


# ---------------------------------------------------------------------------
# Service-specific endpoints
# ---------------------------------------------------------------------------


@app.post("/api/rfdiffusion", response_model=JobInfo)
def run_rfdiffusion(
    target: UploadFile = File(..., description="Target antigen PDB file"),
    framework: UploadFile = File(..., description="Antibody framework PDB file"),
    params: RFdiffusionRequest = Depends(model_form_depends(RFdiffusionRequest)),
) -> JobInfo:
    """RFdiffusion antibody-framework backbone design."""

    def _build(job_id: str, job_dir: Path) -> list[str]:
        target_path = save_upload(target, job_dir / "input" / "target.pdb")
        framework_path = save_upload(framework, job_dir / "input" / "framework.pdb")
        return rfdiffusion_argv(params, target_path, framework_path, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="rfdiffusion",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/proteinmpnn", response_model=JobInfo)
def run_proteinmpnn(
    input_quiver: Optional[UploadFile] = File(
        None, description="Input Quiver (from RFdiffusion). Mutually exclusive with `input_uri`."
    ),
    input_uri: Optional[str] = Form(
        None,
        description=(
            "URI to fetch the input Quiver instead of uploading. Schemes: "
            "job://<id>/<file>, file:///path, oss://<bucket>/<key>, http(s)://..."
        ),
    ),
    params: ProteinMPNNRequest = Depends(model_form_depends(ProteinMPNNRequest)),
) -> JobInfo:
    """ProteinMPNN CDR sequence design over an RFdiffusion-generated backbone set."""

    def _build(job_id: str, job_dir: Path) -> list[str]:
        qv_path = resolve_input(input_quiver, input_uri, job_dir / "input" / "input.qv", settings)
        return proteinmpnn_argv(params, qv_path, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="proteinmpnn",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/rf2", response_model=JobInfo)
def run_rf2(
    input_quiver: Optional[UploadFile] = File(
        None, description="Input Quiver (from ProteinMPNN). Mutually exclusive with `input_uri`."
    ),
    input_uri: Optional[str] = Form(None, description="URI to fetch input Quiver from (see /proteinmpnn)."),
    params: RF2Request = Depends(model_form_depends(RF2Request)),
) -> JobInfo:
    """RF2 structure prediction + filtering over MPNN-designed sequences."""

    def _build(job_id: str, job_dir: Path) -> list[str]:
        qv_path = resolve_input(input_quiver, input_uri, job_dir / "input" / "input.qv", settings)
        return rf2_argv(params, qv_path, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="rf2",
        input_params=params.model_dump(mode="json"),
    )


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)

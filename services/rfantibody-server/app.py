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

from bioq_service import (
    JobInfo,
    attach_mcp,
    create_app,
    execute_task,
    model_form_depends,
    read_version_file,
    resolve_task_id,
)
from fastapi import Depends, File, Form, Header, Request, UploadFile

from .adapter import RFantibodyAdapter
from .models import ProteinMPNNRequest, RF2Request, RFdiffusionRequest
from .settings import RFantibodySettings
from .tools import proteinmpnn_argv, rf2_argv, rfdiffusion_argv
from bioq_service.uris import resolve_input

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


# Remove framework's generic /healthz/detail so our rfantibody-specific
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

    RFantibody weights live on NAS at `RFANTIBODY_WEIGHTS_DIR` (default
    `/data/models/rfantibody/weights/`).  We report presence of the dir and
    file count; `weights_loaded=false` lets the agent detect a misconfigured
    FC mount / unbound SIF without crashing the service.
    """
    wdir = settings.weights_dir
    if not wdir.exists():
        weights_loaded = False
        files_found = 0
    else:
        files_found = sum(1 for p in wdir.rglob("*") if p.is_file())
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
# Service-specific endpoints
# ---------------------------------------------------------------------------


@app.post("/api/rfdiffusion", response_model=JobInfo)
def run_rfdiffusion(
    target: Optional[UploadFile] = File(
        None, description="Target antigen PDB. Mutually exclusive with `target_uri`."
    ),
    target_uri: Optional[str] = Form(
        None,
        description=(
            "URI to fetch the target instead of uploading. Schemes: "
            "job://<id>/<file>, file:///path, oss://<bucket>/<key>, http(s)://..."
        ),
    ),
    framework: Optional[UploadFile] = File(
        None, description="Antibody framework PDB. Mutually exclusive with `framework_uri`."
    ),
    framework_uri: Optional[str] = Form(
        None, description="URI to fetch the framework instead of uploading."
    ),
    params: RFdiffusionRequest = Depends(model_form_depends(RFdiffusionRequest)),
) -> JobInfo:
    """RFdiffusion antibody-framework backbone design.

    Each input can be supplied EITHER as a multipart upload OR as a URI.
    URIs are required when invoking via FC async task mode (``X-Fc-Invocation-Type:
    Async``) for any PDB larger than ~120 KiB — FC's async event payload is
    capped at 128 KiB.
    """

    def _build(job_id: str, job_dir: Path) -> list[str]:
        target_path = resolve_input(target, target_uri, job_dir / "input" / "target.pdb", settings, field_name="target")
        framework_path = resolve_input(framework, framework_uri, job_dir / "input" / "framework.pdb", settings, field_name="framework")
        return rfdiffusion_argv(params, target_path, framework_path, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="rfdiffusion",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/proteinmpnn", response_model=JobInfo)
def run_proteinmpnn(
    input_quiver: Optional[UploadFile] = File(
        None, description="Input Quiver (from RFdiffusion). Mutually exclusive with `input_quiver_uri`."
    ),
    input_quiver_uri: Optional[str] = Form(
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
        qv_path = resolve_input(input_quiver, input_quiver_uri, job_dir / "input" / "input.qv", settings, field_name="input_quiver")
        return proteinmpnn_argv(params, qv_path, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="proteinmpnn",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/rf2", response_model=JobInfo)
def run_rf2(
    input_quiver: Optional[UploadFile] = File(
        None, description="Input Quiver (from ProteinMPNN). Mutually exclusive with `input_quiver_uri`."
    ),
    input_quiver_uri: Optional[str] = Form(None, description="URI to fetch input Quiver from (see /proteinmpnn)."),
    params: RF2Request = Depends(model_form_depends(RF2Request)),
) -> JobInfo:
    """RF2 structure prediction + filtering over MPNN-designed sequences."""

    def _build(job_id: str, job_dir: Path) -> list[str]:
        qv_path = resolve_input(input_quiver, input_quiver_uri, job_dir / "input" / "input.qv", settings, field_name="input_quiver")
        return rf2_argv(params, qv_path, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="rf2",
        input_params=params.model_dump(mode="json"),
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/rfdiffusion", response_model=JobInfo)
    def run_rfdiffusion_task(
        request: Request,
        target: Optional[UploadFile] = File(None),
        target_uri: Optional[str] = Form(None),
        framework: Optional[UploadFile] = File(None),
        framework_uri: Optional[str] = Form(None),
        params: RFdiffusionRequest = Depends(model_form_depends(RFdiffusionRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """RFdiffusion antibody-framework backbone design as a single atomic task.

        Same upload-or-URI duality as the sync endpoint.  Required when payloads
        exceed FC's 128 KiB async invocation cap.
        """
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["target"] = resolve_input(target, target_uri, input_dir / "target.pdb", settings, field_name="target")
            paths["framework"] = resolve_input(framework, framework_uri, input_dir / "framework.pdb", settings, field_name="framework")

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return rfdiffusion_argv(req, paths["target"], paths["framework"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="rfdiffusion", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/proteinmpnn", response_model=JobInfo)
    def run_proteinmpnn_task(
        request: Request,
        input_quiver: Optional[UploadFile] = File(None),
        input_quiver_uri: Optional[str] = Form(None),
        params: ProteinMPNNRequest = Depends(model_form_depends(ProteinMPNNRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """ProteinMPNN CDR sequence design as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["qv"] = resolve_input(input_quiver, input_quiver_uri, input_dir / "input.qv", settings, field_name="input_quiver")

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return proteinmpnn_argv(req, paths["qv"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="proteinmpnn", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/rf2", response_model=JobInfo)
    def run_rf2_task(
        request: Request,
        input_quiver: Optional[UploadFile] = File(None),
        input_quiver_uri: Optional[str] = Form(None),
        params: RF2Request = Depends(model_form_depends(RF2Request)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """RF2 structure prediction as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["qv"] = resolve_input(input_quiver, input_quiver_uri, input_dir / "input.qv", settings, field_name="input_quiver")

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return rf2_argv(req, paths["qv"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="rf2", params=params,
            build_argv=_build, save_inputs=_save,
        )


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)

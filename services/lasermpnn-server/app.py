"""FastAPI app for lasermpnn-server.

Exposes /api/design and /api/design_ligandmpnn (+ matching /api/tasks/*).
Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest, /openapi.json)
come from bioq_service.create_app.
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
from bioq_service.uris import resolve_input
from fastapi import Depends, File, Form, Header, Request, UploadFile

from .adapter import LASErMPNNAdapter
from .models import DesignLigandMPNNRequest, DesignRequest
from .settings import LASErMPNNSettings
from .tools import design_argv, design_ligandmpnn_argv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = LASErMPNNSettings()
adapter = LASErMPNNAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="LASErMPNN Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---- /healthz/detail override: report NAS weights presence ----
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
    """Weights probe: report whether the NAS checkpoints are mounted.

    Does not raise when the mount is missing so the service can still start and
    surface the state here rather than crashing on first inference.
    """
    expected = {
        "nothing_heldout": settings.weights_dir / "laser_weights_0p1A_nothing_heldout.pt",
        "ligandmpnn_split": settings.weights_dir / "laser_weights_0p1A_noise_ligandmpnn_split.pt",
        "soluble": settings.weights_dir / "soluble_weights_no_heldout_drop_clusters_optstep_65000.pt",
        "ligand_encoder": settings.weights_dir / "pretrained_ligand_encoder_weights.pt",
    }
    missing = {k: str(p) for k, p in expected.items() if not p.exists()}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": not missing,
        "weights_missing": missing,
        "lasermpnn_root_present": (settings.root / "LASErMPNN").is_dir(),
        "device": settings.device,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


@app.post("/api/design", response_model=JobInfo)
def post_design(
    params: DesignRequest = Depends(model_form_depends(DesignRequest)),
    pdb: Optional[UploadFile] = File(None),
    pdb_uri: Optional[str] = Form(None),
) -> JobInfo:
    """LASErMPNN ligand-conditioned batch sequence design."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_pdb = resolve_input(pdb, pdb_uri, job_dir / "input" / "input.pdb", settings)
        return design_argv(params, input_pdb=input_pdb, job_dir=job_dir, settings=settings)

    return app.state.runner.submit(
        build_argv=_build, label="design",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/design_ligandmpnn", response_model=JobInfo)
def post_design_ligandmpnn(
    params: DesignLigandMPNNRequest = Depends(model_form_depends(DesignLigandMPNNRequest)),
    pdb: Optional[UploadFile] = File(None),
    pdb_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Retrained-LigandMPNN variant batch design."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_pdb = resolve_input(pdb, pdb_uri, job_dir / "input" / "input.pdb", settings)
        return design_ligandmpnn_argv(params, input_pdb=input_pdb, job_dir=job_dir, settings=settings)

    return app.state.runner.submit(
        build_argv=_build, label="design_ligandmpnn",
        input_params=params.model_dump(mode="json"),
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/design", response_model=JobInfo)
    def post_design_task(
        request: Request,
        params: DesignRequest = Depends(model_form_depends(DesignRequest)),
        pdb: Optional[UploadFile] = File(None),
        pdb_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """LASErMPNN design as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["pdb"] = resolve_input(pdb, pdb_uri, input_dir / "input.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return design_argv(req, input_pdb=paths["pdb"], job_dir=job_dir, settings=settings)

        return execute_task(
            request, job_id=job_id, label="design", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/design_ligandmpnn", response_model=JobInfo)
    def post_design_ligandmpnn_task(
        request: Request,
        params: DesignLigandMPNNRequest = Depends(model_form_depends(DesignLigandMPNNRequest)),
        pdb: Optional[UploadFile] = File(None),
        pdb_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Retrained-LigandMPNN design as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["pdb"] = resolve_input(pdb, pdb_uri, input_dir / "input.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return design_ligandmpnn_argv(req, input_pdb=paths["pdb"], job_dir=job_dir, settings=settings)

        return execute_task(
            request, job_id=job_id, label="design_ligandmpnn", params=params,
            build_argv=_build, save_inputs=_save,
        )


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)

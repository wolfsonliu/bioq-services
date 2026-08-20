"""FastAPI app for immunebuilder-server.

Exposes /api/predict_antibody, /api/predict_nanobody, /api/predict_tcr.
Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest, /openapi.json)
come from `bioq_service.create_app`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from bioq_service import (
    JobInfo,
    attach_mcp,
    create_app,
    model_form_depends,
    read_version_file,
    register_task_endpoint,
)
from fastapi import Depends, Request

from .adapter import ImmuneBuilderAdapter
from .models import AntibodyRequest, NanobodyRequest, TCRRequest
from .settings import ImmuneBuilderSettings
from .tools import (
    predict_antibody_argv,
    predict_nanobody_argv,
    predict_tcr_argv,
    write_fasta,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = ImmuneBuilderSettings()
adapter = ImmuneBuilderAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="ImmuneBuilder Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# Remove framework's generic /healthz/detail so our immunebuilder-specific
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

    Trained model weights live on NAS at `IMMUNEBUILDER_WEIGHTS_DIR`
    (default `/data/models/immunebuilder/trained_model/`).  The image
    contains a symlink at the upstream package's expected path pointing
    here; if NAS is unmounted, the symlink target is missing and the upstream
    code crashes at first inference.  `weights_loaded=false` lets the agent
    detect the misconfiguration before that.
    """
    wdir = settings.weights_dir
    if not wdir.exists():
        weights_loaded = False
        files_found = 0
    else:
        # Count any tracked file (16 expected: 4 antibody + 4 nanobody +
        # 4 tcr + 4 tcr2).
        files_found = sum(1 for p in wdir.iterdir() if p.is_file())
        # ≥12 of the 16 typically indicates a healthy stage; tolerate partial.
        weights_loaded = files_found >= 12
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


@app.post("/api/predict_antibody", response_model=JobInfo)
def post_predict_antibody(
    params: AntibodyRequest = Depends(model_form_depends(AntibodyRequest)),
) -> JobInfo:
    """Predict antibody structure from heavy + light chain sequences."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        fasta_path = write_fasta(
            {"H": params.heavy_sequence, "L": params.light_sequence},
            job_dir / "input" / "input.fasta",
        )
        return predict_antibody_argv(
            params, job_dir=job_dir, fasta_path=fasta_path, settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="predict_antibody",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/predict_nanobody", response_model=JobInfo)
def post_predict_nanobody(
    params: NanobodyRequest = Depends(model_form_depends(NanobodyRequest)),
) -> JobInfo:
    """Predict nanobody structure from heavy chain sequence."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        fasta_path = write_fasta(
            {"H": params.heavy_sequence},
            job_dir / "input" / "input.fasta",
        )
        return predict_nanobody_argv(
            params, job_dir=job_dir, fasta_path=fasta_path, settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="predict_nanobody",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/predict_tcr", response_model=JobInfo)
def post_predict_tcr(
    params: TCRRequest = Depends(model_form_depends(TCRRequest)),
) -> JobInfo:
    """Predict TCR structure from alpha + beta chain sequences."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        fasta_path = write_fasta(
            {"A": params.alpha_sequence, "B": params.beta_sequence},
            job_dir / "input" / "input.fasta",
        )
        return predict_tcr_argv(
            params, job_dir=job_dir, fasta_path=fasta_path, settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="predict_tcr",
        input_params=params.model_dump(mode="json"),
    )


# Task endpoints (synchronous; FC Async Task Mode-friendly).
# No file uploads, so we use the framework's simpler register_task_endpoint helper.

def _antibody_build(req, _job_id: str, job_dir: Path) -> list[str]:
    fasta_path = write_fasta(
        {"H": req.heavy_sequence, "L": req.light_sequence},
        job_dir / "input" / "input.fasta",
    )
    return predict_antibody_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)


def _nanobody_build(req, _job_id: str, job_dir: Path) -> list[str]:
    fasta_path = write_fasta(
        {"H": req.heavy_sequence},
        job_dir / "input" / "input.fasta",
    )
    return predict_nanobody_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)


def _tcr_build(req, _job_id: str, job_dir: Path) -> list[str]:
    fasta_path = write_fasta(
        {"A": req.alpha_sequence, "B": req.beta_sequence},
        job_dir / "input" / "input.fasta",
    )
    return predict_tcr_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)


register_task_endpoint(
    app,
    path="/api/tasks/predict_antibody",
    label="predict_antibody",
    request_model=AntibodyRequest,
    build_argv=_antibody_build,
    summary="Predict antibody structure from heavy + light chain sequences (single atomic task).",
)
register_task_endpoint(
    app,
    path="/api/tasks/predict_nanobody",
    label="predict_nanobody",
    request_model=NanobodyRequest,
    build_argv=_nanobody_build,
    summary="Predict nanobody structure from heavy chain sequence (single atomic task).",
)
register_task_endpoint(
    app,
    path="/api/tasks/predict_tcr",
    label="predict_tcr",
    request_model=TCRRequest,
    build_argv=_tcr_build,
    summary="Predict TCR structure from alpha + beta chain sequences (single atomic task).",
)

attach_mcp(app)

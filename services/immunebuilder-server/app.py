"""FastAPI app for immunebuilder-server.

Exposes /api/predict_antibody, /api/predict_nanobody, /api/predict_tcr.
Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest, /openapi.json)
come from `bioagent_service.create_app`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from bioagent_service import (
    JobInfo,
    attach_mcp,
    create_app,
    model_form_depends,
    read_version_file,
    register_task_endpoint,
)
from fastapi import Depends

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
)
register_task_endpoint(
    app,
    path="/api/tasks/predict_nanobody",
    label="predict_nanobody",
    request_model=NanobodyRequest,
    build_argv=_nanobody_build,
)
register_task_endpoint(
    app,
    path="/api/tasks/predict_tcr",
    label="predict_tcr",
    request_model=TCRRequest,
    build_argv=_tcr_build,
)

attach_mcp(app)

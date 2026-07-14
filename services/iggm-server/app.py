"""FastAPI app for iggm-server.

Endpoints:
  POST /api/design               design / inverse_design / fr_design
  POST /api/affinity-maturation  affinity maturation (needs fasta_origin)
  POST /api/epitope              interface epitope calculation (tool)
  + /api/tasks/<name> async twins for FC Async Task Mode.

Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest, /openapi.json)
come from `bioagent_service.create_app`.
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
from fastapi import Depends, File, Form, Header, HTTPException, Request, UploadFile

from .adapter import IgGMAdapter
from .models import AffinityMaturationRequest, DesignRequest, EpitopeRequest
from .settings import IgGMSettings
from .tools import design_argv, epitope_argv
from bioagent_service.uris import resolve_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = IgGMSettings()
adapter = IgGMAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="IgGM Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---- /healthz/detail override: report NAS checkpoint presence ----
def _strip_route(router, path: str, method: str) -> None:
    router.routes = [
        r
        for r in router.routes
        if not (
            getattr(r, "path", None) == path
            and method in getattr(r, "methods", set())
        )
    ]
    for r in router.routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            _strip_route(inner, path, method)


_strip_route(app.router, "/healthz/detail", "GET")


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Surface whether the five NAS-mounted checkpoints are reachable.

    Weights live on NAS at IGGM_WEIGHTS_DIR (default /data/models/iggm/),
    symlinked to /opt/iggm/checkpoints so upstream finds them without a runtime
    torch.hub download. Reports missing .pth so the agent can detect a
    misconfigured mount before a job crashes.
    """
    status = settings.checkpoints_status()
    missing = {n: str(settings.checkpoint_path(n)) for n, ok in status.items() if not ok}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": not missing,
        "weights_missing": missing,
        "checkpoints": status,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


def _require_checkpoints(run_task: str) -> None:
    """422 if a checkpoint the task needs is absent (avoids runtime download)."""
    missing = settings.missing_checkpoints(run_task)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Checkpoints missing for run_task={run_task}: {missing}. "
                f"Expected under {settings.weights_dir}. See /healthz/detail."
            ),
        )


def _save_design_inputs(
    fasta: Optional[UploadFile],
    fasta_uri: Optional[str],
    antigen: Optional[UploadFile],
    antigen_uri: Optional[str],
    input_dir: Path,
    fasta_origin: Optional[UploadFile] = None,
    fasta_origin_uri: Optional[str] = None,
) -> dict[str, Path]:
    """Persist uploaded/URI inputs into the job input dir."""
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fasta": resolve_input(fasta, fasta_uri, input_dir / "input.fasta", settings),
        "antigen": resolve_input(antigen, antigen_uri, input_dir / "antigen.pdb", settings),
    }
    if fasta_origin is not None or fasta_origin_uri:
        paths["fasta_origin"] = resolve_input(
            fasta_origin, fasta_origin_uri, input_dir / "origin.fasta", settings
        )
    return paths


# ---- /api/design (design / inverse_design / fr_design) ----


@app.post("/api/design", response_model=JobInfo)
def post_design(
    params: DesignRequest = Depends(model_form_depends(DesignRequest)),
    fasta: Optional[UploadFile] = File(None),
    fasta_uri: Optional[str] = Form(None),
    antigen: Optional[UploadFile] = File(None),
    antigen_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Antibody design: CDR sequence + structure co-design (design), sequence
    on a fixed backbone (inverse_design), or framework-region redesign
    (fr_design), selected via run_task."""
    _require_checkpoints(params.run_task)

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        p = _save_design_inputs(fasta, fasta_uri, antigen, antigen_uri, job_dir / "input")
        return design_argv(
            params, job_dir=job_dir, fasta_path=p["fasta"], antigen_path=p["antigen"],
            settings=settings, run_task=params.run_task,
        )

    return app.state.runner.submit(
        build_argv=_build, label="design",
        input_params=params.model_dump(mode="json"),
    )


# ---- /api/affinity-maturation ----


@app.post("/api/affinity-maturation", response_model=JobInfo)
def post_affinity_maturation(
    params: AffinityMaturationRequest = Depends(model_form_depends(AffinityMaturationRequest)),
    fasta: Optional[UploadFile] = File(None),
    fasta_uri: Optional[str] = Form(None),
    antigen: Optional[UploadFile] = File(None),
    antigen_uri: Optional[str] = Form(None),
    fasta_origin: Optional[UploadFile] = File(None),
    fasta_origin_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Affinity maturation: per-position mutation scan of an existing antibody.

    Requires fasta_origin (the original sequence to mature from)."""
    _require_checkpoints("affinity_maturation")
    if fasta_origin is None and not fasta_origin_uri:
        raise HTTPException(
            status_code=422,
            detail="affinity-maturation requires fasta_origin (upload or URI).",
        )

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        p = _save_design_inputs(
            fasta, fasta_uri, antigen, antigen_uri, job_dir / "input",
            fasta_origin, fasta_origin_uri,
        )
        return design_argv(
            params, job_dir=job_dir, fasta_path=p["fasta"], antigen_path=p["antigen"],
            settings=settings, run_task="affinity_maturation",
            fasta_origin_path=p["fasta_origin"],
        )

    return app.state.runner.submit(
        build_argv=_build, label="affinity_maturation",
        input_params=params.model_dump(mode="json"),
    )


# ---- /api/epitope (tool; no model weights needed) ----


@app.post("/api/epitope", response_model=JobInfo)
def post_epitope(
    params: EpitopeRequest = Depends(model_form_depends(EpitopeRequest)),
    fasta: Optional[UploadFile] = File(None),
    fasta_uri: Optional[str] = Form(None),
    antigen: Optional[UploadFile] = File(None),
    antigen_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Compute the antigen interface epitope from a complex (fasta + antigen)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        p = _save_design_inputs(fasta, fasta_uri, antigen, antigen_uri, job_dir / "input")
        return epitope_argv(
            job_dir=job_dir, fasta_path=p["fasta"], antigen_path=p["antigen"],
            settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build, label="epitope",
        input_params=params.model_dump(mode="json"),
    )


# ---- FC async task twins ----

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/design", response_model=JobInfo)
    def post_design_task(
        request: Request,
        params: DesignRequest = Depends(model_form_depends(DesignRequest)),
        fasta: Optional[UploadFile] = File(None),
        fasta_uri: Optional[str] = Form(None),
        antigen: Optional[UploadFile] = File(None),
        antigen_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """design/inverse_design/fr_design as a single blocking task (FC Async)."""
        _require_checkpoints(params.run_task)
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths.update(
                _save_design_inputs(fasta, fasta_uri, antigen, antigen_uri, input_dir)
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return design_argv(
                req, job_dir=job_dir, fasta_path=paths["fasta"],
                antigen_path=paths["antigen"], settings=settings, run_task=req.run_task,
            )

        return execute_task(
            request, job_id=job_id, label="design", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/affinity-maturation", response_model=JobInfo)
    def post_affinity_maturation_task(
        request: Request,
        params: AffinityMaturationRequest = Depends(model_form_depends(AffinityMaturationRequest)),
        fasta: Optional[UploadFile] = File(None),
        fasta_uri: Optional[str] = Form(None),
        antigen: Optional[UploadFile] = File(None),
        antigen_uri: Optional[str] = Form(None),
        fasta_origin: Optional[UploadFile] = File(None),
        fasta_origin_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Affinity maturation as a single blocking task (FC Async)."""
        _require_checkpoints("affinity_maturation")
        if fasta_origin is None and not fasta_origin_uri:
            raise HTTPException(
                status_code=422,
                detail="affinity-maturation requires fasta_origin (upload or URI).",
            )
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths.update(
                _save_design_inputs(
                    fasta, fasta_uri, antigen, antigen_uri, input_dir,
                    fasta_origin, fasta_origin_uri,
                )
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return design_argv(
                req, job_dir=job_dir, fasta_path=paths["fasta"],
                antigen_path=paths["antigen"], settings=settings,
                run_task="affinity_maturation", fasta_origin_path=paths["fasta_origin"],
            )

        return execute_task(
            request, job_id=job_id, label="affinity_maturation", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/epitope", response_model=JobInfo)
    def post_epitope_task(
        request: Request,
        params: EpitopeRequest = Depends(model_form_depends(EpitopeRequest)),
        fasta: Optional[UploadFile] = File(None),
        fasta_uri: Optional[str] = Form(None),
        antigen: Optional[UploadFile] = File(None),
        antigen_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Epitope calculation as a single blocking task (FC Async)."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths.update(
                _save_design_inputs(fasta, fasta_uri, antigen, antigen_uri, input_dir)
            )

        def _build(_req, _job_id: str, job_dir: Path) -> list[str]:
            return epitope_argv(
                job_dir=job_dir, fasta_path=paths["fasta"],
                antigen_path=paths["antigen"], settings=settings,
            )

        return execute_task(
            request, job_id=job_id, label="epitope", params=params,
            build_argv=_build, save_inputs=_save,
        )


attach_mcp(app)

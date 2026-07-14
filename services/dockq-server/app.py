"""FastAPI app for dockq-server.

Exposes /api/score (single pair) and /api/score_batch (1 native + N models).
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

from .adapter import DockQAdapter
from .models import ScoreBatchRequest, ScoreRequest
from .settings import DockQSettings
from .tools import batch_argv, score_argv
from bioagent_service.uris import resolve_input, save_upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = DockQSettings()
adapter = DockQAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="DockQ Server",
    version=read_version_file(__file__, default="0.0.1"),
)


@app.post("/api/score", response_model=JobInfo)
def post_score(
    params: ScoreRequest = Depends(model_form_depends(ScoreRequest)),
    model: Optional[UploadFile] = File(None),
    native: Optional[UploadFile] = File(None),
    model_uri: Optional[str] = Form(None),
    native_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Score a single (model, native) pair via DockQ; returns DockQ's JSON output."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        model_path = resolve_input(model, model_uri, input_dir / "model.pdb", settings)
        native_path = resolve_input(native, native_uri, input_dir / "native.pdb", settings)
        return score_argv(
            params,
            job_dir=job_dir,
            model_path=model_path,
            native_path=native_path,
            settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build, label="score",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/score_batch", response_model=JobInfo)
def post_score_batch(
    params: ScoreBatchRequest = Depends(model_form_depends(ScoreBatchRequest)),
    native: Optional[UploadFile] = File(None),
    native_uri: Optional[str] = Form(None),
    models: Optional[list[UploadFile]] = File(None),
) -> JobInfo:
    """Score N candidate models against 1 reference native.

    Per-model JSONs land in `output/per_model/<basename>.json`; the sorted
    summary is `output/scores.csv`. Models that errored show up in
    `output/failed.csv` (job still completes successfully if at least one
    model produced a valid score).
    """
    if not models:
        raise HTTPException(
            status_code=422,
            detail="At least one `models` upload is required (use repeated -F models=@...).",
        )
    if len(models) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(models)} exceeds max_batch_size={settings.max_batch_size}.",
        )

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        native_path = resolve_input(native, native_uri, input_dir / "native.pdb", settings)
        models_dir = input_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for i, upload in enumerate(models):
            # Sanitize filename: keep stem, force .pdb / .cif / .cif.gz suffix.
            basename = Path(upload.filename or f"model_{i:04d}.pdb").name
            save_upload(upload, models_dir / basename)
        return batch_argv(
            params,
            job_dir=job_dir,
            native_path=native_path,
            models_dir=models_dir,
            settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build, label="score_batch",
        input_params={**params.model_dump(mode="json"), "num_models": len(models)},
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/score", response_model=JobInfo)
    def post_score_task(
        request: Request,
        params: ScoreRequest = Depends(model_form_depends(ScoreRequest)),
        model: Optional[UploadFile] = File(None),
        native: Optional[UploadFile] = File(None),
        model_uri: Optional[str] = Form(None),
        native_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Score a single (model, native) pair as a single atomic task.

        Blocks until DockQ completes.  For the submit/poll interface, use POST /api/score.
        """
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["model"] = resolve_input(model, model_uri, input_dir / "model.pdb", settings)
            paths["native"] = resolve_input(native, native_uri, input_dir / "native.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return score_argv(
                req, job_dir=job_dir,
                model_path=paths["model"], native_path=paths["native"],
                settings=settings,
            )

        return execute_task(
            request,
            job_id=job_id,
            label="score",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )

    @app.post("/api/tasks/score_batch", response_model=JobInfo)
    def post_score_batch_task(
        request: Request,
        params: ScoreBatchRequest = Depends(model_form_depends(ScoreBatchRequest)),
        native: Optional[UploadFile] = File(None),
        native_uri: Optional[str] = Form(None),
        models: Optional[list[UploadFile]] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Score N candidate models against 1 native as a single atomic task."""
        if not models:
            raise HTTPException(
                status_code=422,
                detail="At least one `models` upload is required (use repeated -F models=@...).",
            )
        if len(models) > settings.max_batch_size:
            raise HTTPException(
                status_code=422,
                detail=f"Batch size {len(models)} exceeds max_batch_size={settings.max_batch_size}.",
            )

        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["native"] = resolve_input(native, native_uri, input_dir / "native.pdb", settings)
            models_dir = input_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            for i, upload in enumerate(models):
                basename = Path(upload.filename or f"model_{i:04d}.pdb").name
                save_upload(upload, models_dir / basename)
            paths["models_dir"] = models_dir

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return batch_argv(
                req, job_dir=job_dir,
                native_path=paths["native"], models_dir=paths["models_dir"],
                settings=settings,
            )

        return execute_task(
            request,
            job_id=job_id,
            label="score_batch",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)

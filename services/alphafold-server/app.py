"""FastAPI app for alphafold-server.

Exposes `/api/fold` for protein structure prediction. Job lifecycle endpoints
(`/healthz`, `/api/jobs/*`, `/api/manifest`, `/openapi.json`) come from
`bioagent_service.create_app`.
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

from .adapter import AlphaFoldAdapter
from .models import FoldRequest
from .settings import AlphaFoldSettings
from .tools import fold_argv
from .uris import resolve_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = AlphaFoldSettings()
adapter = AlphaFoldAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="AlphaFold Server",
    version=read_version_file(__file__, default="0.0.1"),
)


@app.post("/api/fold", response_model=JobInfo)
def post_fold(
    params: FoldRequest = Depends(model_form_depends(FoldRequest)),
    input_fasta: Optional[UploadFile] = File(None),
    input_fasta_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Predict protein structure using AlphaFold v2.3.2."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = resolve_input(
            input_fasta,
            input_fasta_uri,
            dest=input_dir / "input.fasta",
            settings=settings,
        )
        return fold_argv(
            params, job_dir=job_dir, fasta_path=fasta_path, settings=settings
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="fold",
        input_params=params.model_dump(mode="json"),
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/fold", response_model=JobInfo)
    def post_fold_task(
        request: Request,
        params: FoldRequest = Depends(model_form_depends(FoldRequest)),
        input_fasta: Optional[UploadFile] = File(None),
        input_fasta_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Predict protein structure using AlphaFold v2.3.2 as a single atomic task.

        Blocks until pipeline completion.  Designed to be invoked via FC Async
        Task Mode (X-Fc-Invocation-Type: Async): FC enqueues + dispatches, and
        the HTTP request stays active for the full subprocess lifetime so FC
        won't recycle the instance mid-run.

        For the legacy submit/poll interface, use POST /api/fold instead.
        """
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        fasta_paths: list[Path] = []

        def _save(_req, input_dir: Path) -> None:
            fasta_paths.append(
                resolve_input(
                    input_fasta,
                    input_fasta_uri,
                    dest=input_dir / "input.fasta",
                    settings=settings,
                )
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return fold_argv(
                req, job_dir=job_dir, fasta_path=fasta_paths[0], settings=settings
            )

        return execute_task(
            request,
            job_id=job_id,
            label="fold",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


attach_mcp(app)

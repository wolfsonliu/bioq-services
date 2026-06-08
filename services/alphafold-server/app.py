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
    model_form_depends,
    read_version_file,
)
from fastapi import Depends, File, Form, UploadFile

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


attach_mcp(app)

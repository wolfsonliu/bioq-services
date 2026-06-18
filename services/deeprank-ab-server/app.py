"""FastAPI app for deeprank-ab-server.

Exposes /api/score for scoring antibody-antigen docking complexes.
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
from fastapi import Depends, File, Form, Header, Request, UploadFile

from .adapter import DeepRankAbAdapter
from .models import ScoreRequest
from .settings import DeepRankAbSettings
from .argv import score_argv
from .uris import resolve_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = DeepRankAbSettings()
adapter = DeepRankAbAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="DeepRank-Ab Server",
    version=read_version_file(__file__, default="0.0.1"),
)


@app.post("/api/score", response_model=JobInfo)
def post_score(
    params: ScoreRequest = Depends(model_form_depends(ScoreRequest)),
    input_pdb: Optional[UploadFile] = File(None),
    input_pdb_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Score an antibody-antigen docking complex.

    Runs the DeepRank-Ab EGNN pipeline: PDB processing, ESM-2 embeddings,
    ANARCI CDR annotation, atom-level graph construction, MCL clustering,
    and EGNN inference. Returns predicted DockQ scores as a CSV.
    """

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        pdb_path = resolve_input(input_pdb, input_pdb_uri, input_dir / "input.pdb", settings)
        return score_argv(
            params,
            job_dir=job_dir,
            pdb_path=pdb_path,
            settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="score",
        input_params=params.model_dump(mode="json"),
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/score", response_model=JobInfo)
    def post_score_task(
        request: Request,
        params: ScoreRequest = Depends(model_form_depends(ScoreRequest)),
        input_pdb: Optional[UploadFile] = File(None),
        input_pdb_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Score an antibody-antigen docking complex as a single atomic task.

        Blocks until the DeepRank-Ab pipeline (PDB processing → ESM-2 embeddings →
        ANARCI → graph → MCL → EGNN) completes.  Designed for FC Async Task Mode
        invocation; the submit/poll interface is at POST /api/score.
        """
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        pdb_paths: list[Path] = []

        def _save(_req, input_dir: Path) -> None:
            pdb_paths.append(
                resolve_input(input_pdb, input_pdb_uri, input_dir / "input.pdb", settings)
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return score_argv(req, job_dir=job_dir, pdb_path=pdb_paths[0], settings=settings)

        return execute_task(
            request,
            job_id=job_id,
            label="score",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


attach_mcp(app)

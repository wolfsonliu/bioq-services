"""FastAPI app for deeprank-ab-server.

Exposes /api/score for scoring antibody-antigen docking complexes.
Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest, /openapi.json)
come from `bioagent_service.create_app`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from bioagent_service import JobInfo, attach_mcp, create_app, model_form_depends, read_version_file
from fastapi import Depends, File, Form, UploadFile

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


attach_mcp(app)

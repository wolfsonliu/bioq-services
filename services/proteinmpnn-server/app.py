"""FastAPI app for proteinmpnn-server.

Exposes /api/design, /api/score, /api/probs. Job lifecycle endpoints
(/health, /api/jobs/*, /api/manifest, /openapi.json) come from
`bioagent_service.create_app`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

from bioagent_service import JobInfo, create_app
from fastapi import File, Form, UploadFile

from .adapter import ProteinMPNNAdapter
from .models import DesignRequest, ProbsRequest, ScoreRequest
from .settings import ProteinMPNNSettings
from .tools import design_argv, prepare_inputs, probs_argv, score_argv
from .uris import resolve_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = ProteinMPNNSettings()
adapter = ProteinMPNNAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="ProteinMPNN Server",
    version="0.0.1",
)


@app.post("/api/design", response_model=JobInfo)
def post_design(
    params: Annotated[DesignRequest, Form()],
    pdb: Optional[UploadFile] = File(None),
    pdb_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Sequence design (FASTA output) over the input PDB."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        resolve_input(pdb, pdb_uri, job_dir / "input" / "input.pdb", settings)
        paths = prepare_inputs(
            job_dir,
            settings=settings,
            ca_only=(params.model_variant == "ca_only"),
            chains_to_design=params.chains_to_design,
            fixed_positions=params.fixed_positions,
            tied_positions=params.tied_positions,
            homooligomer=params.homooligomer,
            bias_AA=params.bias_AA,
            bias_by_res=params.bias_by_res,
            omit_AA_per_chain=params.omit_AA_per_chain,
        )
        return design_argv(params, job_dir=job_dir, paths=paths, settings=settings)

    return app.state.runner.submit(build_argv=_build, label="design")


@app.post("/api/score", response_model=JobInfo)
def post_score(
    params: Annotated[ScoreRequest, Form()],
    pdb: Optional[UploadFile] = File(None),
    pdb_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Score a (structure, sequence) pair via --score_only 1."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        resolve_input(pdb, pdb_uri, job_dir / "input" / "input.pdb", settings)
        paths = prepare_inputs(
            job_dir,
            settings=settings,
            ca_only=(params.model_variant == "ca_only"),
            chains_to_design=params.chains_to_design,
            fixed_positions=None,
            tied_positions=None,
            homooligomer=False,
            bias_AA=None,
            bias_by_res=None,
            omit_AA_per_chain=None,
        )
        return score_argv(params, job_dir=job_dir, paths=paths, settings=settings)

    return app.state.runner.submit(build_argv=_build, label="score")


@app.post("/api/probs", response_model=JobInfo)
def post_probs(
    params: Annotated[ProbsRequest, Form()],
    pdb: Optional[UploadFile] = File(None),
    pdb_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Per-residue AA probability output (conditional / unconditional)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        resolve_input(pdb, pdb_uri, job_dir / "input" / "input.pdb", settings)
        paths = prepare_inputs(
            job_dir,
            settings=settings,
            ca_only=(params.model_variant == "ca_only"),
            chains_to_design=params.chains_to_design,
            fixed_positions=None,
            tied_positions=None,
            homooligomer=False,
            bias_AA=None,
            bias_by_res=None,
            omit_AA_per_chain=None,
        )
        return probs_argv(params, job_dir=job_dir, paths=paths, settings=settings)

    return app.state.runner.submit(build_argv=_build, label="probs")

"""CLI batch-mode entry point for deeprank-ab-server.

Usage::

    python -m server.cli score --input-pdb /data/complex.pdb \\
        --heavy-chain-id H --light-chain-id L --antigen-chain-id A \\
        --output-dir /scratch/results/
"""

from __future__ import annotations

import shutil
from pathlib import Path

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import DeepRankAbAdapter
from .models import ScoreRequest
from .settings import DeepRankAbSettings
from .argv import score_argv

settings = DeepRankAbSettings()
adapter = DeepRankAbAdapter(settings=settings)


def _score_build(req, inputs, job_dir, settings):
    return score_argv(
        req,
        job_dir=job_dir,
        pdb_path=inputs["input_pdb"],
        settings=settings,
    )


endpoints = {
    "score": CLIEndpoint(
        name="score",
        help="Score an antibody-antigen docking complex via DeepRank-Ab",
        request_model=ScoreRequest,
        build_argv=_score_build,
        inputs={"input_pdb": ("Input PDB file with antibody-antigen complex", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

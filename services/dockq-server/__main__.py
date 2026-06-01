"""CLI batch-mode entry point for dockq-server.

Usage::

    python -m server.cli score \\
        --model /data/model.pdb --native /data/native.pdb \\
        --output-dir /scratch/results/

    python -m server.cli score_batch \\
        --native /data/native.pdb --models-dir /data/candidates/ \\
        --output-dir /scratch/results/
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import DockQAdapter
from .models import ScoreBatchRequest, ScoreRequest
from .settings import DockQSettings
from .tools import batch_argv, score_argv

settings = DockQSettings()
adapter = DockQAdapter(settings=settings)


def _score_build(req, inputs, job_dir, settings):
    return score_argv(
        req,
        job_dir=job_dir,
        model_path=inputs["model"],
        native_path=inputs["native"],
        settings=settings,
    )


def _batch_build(req, inputs, job_dir, settings):
    return batch_argv(
        req,
        job_dir=job_dir,
        native_path=inputs["native"],
        models_dir=inputs["models_dir"],
        settings=settings,
    )


endpoints = {
    "score": CLIEndpoint(
        name="score",
        help="Score a single (model, native) pair via DockQ",
        request_model=ScoreRequest,
        build_argv=_score_build,
        inputs={
            "model": ("Model PDB/CIF file", True),
            "native": ("Native/reference PDB/CIF file", True),
        },
    ),
    "score_batch": CLIEndpoint(
        name="score_batch",
        help="Score N candidate models against 1 reference native",
        request_model=ScoreBatchRequest,
        build_argv=_batch_build,
        inputs={
            "native": ("Native/reference PDB/CIF file", True),
            "models_dir": ("Directory containing candidate model PDB/CIF files", True),
        },
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

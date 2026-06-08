"""CLI batch-mode entry point for alphafold-server.

Usage::

    python -m server fold \
        --input input.fasta \
        --output-dir /scratch/results/
"""

from __future__ import annotations

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import AlphaFoldAdapter
from .models import FoldRequest
from .settings import AlphaFoldSettings
from .tools import fold_argv

settings = AlphaFoldSettings()
adapter = AlphaFoldAdapter(settings=settings)


def _fold_build(req, inputs, job_dir, settings):
    fasta_path = inputs["input_fasta"]
    return fold_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)


endpoints = {
    "fold": CLIEndpoint(
        name="fold",
        help="Predict protein structure using AlphaFold v2.3.2",
        request_model=FoldRequest,
        build_argv=_fold_build,
        inputs={"input_fasta": ("Input FASTA file path", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

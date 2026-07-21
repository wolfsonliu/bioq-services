"""CLI batch-mode entry point for immunebuilder-server.

Usage::

    python -m server predict_antibody \\
        --heavy-sequence "EVQLVESGGGLVQPGG..." --light-sequence "DIQMTQSPSSLSA..." \\
        --output-dir /scratch/results/

    python -m server predict_nanobody \\
        --heavy-sequence "EVQLVESGGGLVQPGG..." \\
        --output-dir /scratch/results/

    python -m server predict_tcr \\
        --alpha-sequence "METLL..." --beta-sequence "MGIRL..." \\
        --output-dir /scratch/results/
"""

from __future__ import annotations

from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import ImmuneBuilderAdapter
from .models import AntibodyRequest, NanobodyRequest, TCRRequest
from .settings import ImmuneBuilderSettings
from .tools import (
    predict_antibody_argv,
    predict_nanobody_argv,
    predict_tcr_argv,
    write_fasta,
)

settings = ImmuneBuilderSettings()
adapter = ImmuneBuilderAdapter(settings=settings)


def _antibody_build(req, inputs, job_dir, settings):
    fasta_path = write_fasta(
        {"H": req.heavy_sequence, "L": req.light_sequence},
        job_dir / "input" / "input.fasta",
    )
    return predict_antibody_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)


def _nanobody_build(req, inputs, job_dir, settings):
    fasta_path = write_fasta(
        {"H": req.heavy_sequence},
        job_dir / "input" / "input.fasta",
    )
    return predict_nanobody_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)


def _tcr_build(req, inputs, job_dir, settings):
    fasta_path = write_fasta(
        {"A": req.alpha_sequence, "B": req.beta_sequence},
        job_dir / "input" / "input.fasta",
    )
    return predict_tcr_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)


endpoints = {
    "predict_antibody": CLIEndpoint(
        name="predict_antibody",
        help="Predict antibody structure from heavy + light chain sequences",
        request_model=AntibodyRequest,
        build_argv=_antibody_build,
    ),
    "predict_nanobody": CLIEndpoint(
        name="predict_nanobody",
        help="Predict nanobody structure from heavy chain sequence",
        request_model=NanobodyRequest,
        build_argv=_nanobody_build,
    ),
    "predict_tcr": CLIEndpoint(
        name="predict_tcr",
        help="Predict TCR structure from alpha + beta chain sequences",
        request_model=TCRRequest,
        build_argv=_tcr_build,
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

"""CLI batch-mode entry point for seqkit-server (SIF / sbatch).

Usage::

    python -m server stats --input-fasta reads.fasta --output-dir /scratch/out/
    python -m server revcomp --input-fasta reads.fasta \
        --params-json '{"seq_type": "dna"}' --output-dir out/
"""

from __future__ import annotations

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import SeqkitAdapter
from .models import RevcompRequest, StatsRequest
from .settings import SeqkitSettings
from .tools import revcomp_argv, stats_argv

settings = SeqkitSettings()
adapter = SeqkitAdapter(settings=settings)


def _stats_build(req, inputs, job_dir, settings):
    return stats_argv(req, job_dir=job_dir, input_fasta=inputs["input_fasta"], settings=settings)


def _revcomp_build(req, inputs, job_dir, settings):
    return revcomp_argv(req, job_dir=job_dir, input_fasta=inputs["input_fasta"], settings=settings)


endpoints = {
    "stats": CLIEndpoint(
        name="stats",
        help="Summary statistics for one FASTA/FASTQ file",
        request_model=StatsRequest,
        build_argv=_stats_build,
        inputs={"input_fasta": ("Input FASTA/FASTQ file", True)},
    ),
    "revcomp": CLIEndpoint(
        name="revcomp",
        help="Reverse-complement every record in one FASTA/FASTQ file",
        request_model=RevcompRequest,
        build_argv=_revcomp_build,
        inputs={"input_fasta": ("Input FASTA/FASTQ file", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

"""CLI batch-mode entry point for iggm-server.

Same Docker image runs the HTTP service (uvicorn) and this CLI batch mode
(`python -m server <endpoint> ...`).  See
engineering/decisions/2026-05-29-cli-batch-mode.md.  Large affinity-maturation
sweeps should use this mode under Slurm sbatch.

Usage::

    python -m server design --fasta ab.fasta --antigen ag.pdb \\
        --run-task design --num-samples 4 --output-dir /scratch/run1/

    python -m server affinity-maturation --fasta ab.fasta --antigen ag.pdb \\
        --fasta-origin native.fasta --num-samples 10 --output-dir /scratch/run2/

    python -m server epitope --fasta complex.fasta --antigen complex.pdb \\
        --output-dir /scratch/run3/
"""

from __future__ import annotations

import shutil
from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import IgGMAdapter
from .models import AffinityMaturationRequest, DesignRequest, EpitopeRequest
from .settings import IgGMSettings
from .tools import design_argv, epitope_argv

settings = IgGMSettings()
adapter = IgGMAdapter(settings=settings)


def _stage(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def _design_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    fasta = _stage(inputs["fasta"], input_dir / "input.fasta")
    antigen = _stage(inputs["antigen"], input_dir / "antigen.pdb")
    return design_argv(
        req, job_dir=job_dir, fasta_path=fasta, antigen_path=antigen,
        settings=settings, run_task=req.run_task,
    )


def _affinity_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    fasta = _stage(inputs["fasta"], input_dir / "input.fasta")
    antigen = _stage(inputs["antigen"], input_dir / "antigen.pdb")
    origin = _stage(inputs["fasta_origin"], input_dir / "origin.fasta")
    return design_argv(
        req, job_dir=job_dir, fasta_path=fasta, antigen_path=antigen,
        settings=settings, run_task="affinity_maturation", fasta_origin_path=origin,
    )


def _epitope_build(_req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    fasta = _stage(inputs["fasta"], input_dir / "input.fasta")
    antigen = _stage(inputs["antigen"], input_dir / "antigen.pdb")
    return epitope_argv(
        job_dir=job_dir, fasta_path=fasta, antigen_path=antigen, settings=settings,
    )


endpoints = {
    "design": CLIEndpoint(
        name="design",
        help="Antibody design (design / inverse_design / fr_design via --run-task)",
        request_model=DesignRequest,
        build_argv=_design_build,
        inputs={
            "fasta": ("Antibody FASTA (X = design region, antigen last)", True),
            "antigen": ("Antigen PDB", True),
        },
    ),
    "affinity-maturation": CLIEndpoint(
        name="affinity-maturation",
        help="Affinity maturation (per-position mutation scan)",
        request_model=AffinityMaturationRequest,
        build_argv=_affinity_build,
        inputs={
            "fasta": ("Antibody FASTA with X-masked design region", True),
            "antigen": ("Antigen PDB", True),
            "fasta_origin": ("Original antibody FASTA to mature from", True),
        },
    ),
    "epitope": CLIEndpoint(
        name="epitope",
        help="Compute the antigen interface epitope from a complex",
        request_model=EpitopeRequest,
        build_argv=_epitope_build,
        inputs={
            "fasta": ("Complex FASTA (antigen last)", True),
            "antigen": ("Complex/antigen PDB", True),
        },
    ),
}


create_cli(adapter, settings, endpoints, version="0.0.1")

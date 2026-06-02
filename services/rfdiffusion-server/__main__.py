"""CLI batch-mode entry point for rfdiffusion-server.

Usage::

    python -m server unconditional --num-designs 10 --min-length 100 --max-length 200 \\
        --output-dir /scratch/results/

    python -m server binder --input-pdb /data/target.pdb \\
        --contigs "A1-150/0 70-100" --hotspots "A146,A170" \\
        --output-dir /scratch/results/
"""

from __future__ import annotations

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import RFdiffusionAdapter
from .models import (
    BinderRequest,
    CustomRequest,
    MotifRequest,
    SymmetryRequest,
    UnconditionalRequest,
)
from .settings import RFdiffusionSettings
from .tools import (
    binder_argv,
    custom_argv,
    motif_argv,
    symmetry_argv,
    unconditional_argv,
)

settings = RFdiffusionSettings()
adapter = RFdiffusionAdapter(settings=settings)


def _unconditional_build(req, inputs, job_dir, settings):
    return unconditional_argv(req, job_dir, settings)


def _motif_build(req, inputs, job_dir, settings):
    return motif_argv(req, inputs["input_pdb"], job_dir, settings)


def _binder_build(req, inputs, job_dir, settings):
    return binder_argv(req, inputs["input_pdb"], job_dir, settings)


def _symmetry_build(req, inputs, job_dir, settings):
    return symmetry_argv(req, job_dir, settings)


def _custom_build(req, inputs, job_dir, settings):
    return custom_argv(req, inputs.get("input_pdb"), job_dir, settings)


endpoints = {
    "unconditional": CLIEndpoint(
        name="unconditional",
        help="Unconditional monomer backbone generation",
        request_model=UnconditionalRequest,
        build_argv=_unconditional_build,
    ),
    "motif": CLIEndpoint(
        name="motif",
        help="Motif scaffolding (input PDB + contig)",
        request_model=MotifRequest,
        build_argv=_motif_build,
        inputs={"input_pdb": ("Input PDB carrying the motif", True)},
    ),
    "binder": CLIEndpoint(
        name="binder",
        help="PPI binder design against a target PDB",
        request_model=BinderRequest,
        build_argv=_binder_build,
        inputs={"input_pdb": ("Target PDB file", True)},
    ),
    "symmetry": CLIEndpoint(
        name="symmetry",
        help="Symmetric oligomer generation",
        request_model=SymmetryRequest,
        build_argv=_symmetry_build,
    ),
    "custom": CLIEndpoint(
        name="custom",
        help="Raw contig + freeform Hydra overrides",
        request_model=CustomRequest,
        build_argv=_custom_build,
        inputs={"input_pdb": ("Optional input PDB", False)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.1.0")

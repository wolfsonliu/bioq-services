"""CLI batch-mode entry point for rfantibody-server.

Usage::

    python -m server rfdiffusion \\
        --target /data/target.pdb --framework /data/framework.pdb \\
        --output-dir /scratch/results/

    python -m server proteinmpnn \\
        --input-quiver /scratch/results/output/1_rfdiffusion.qv \\
        --output-dir /scratch/results_mpnn/

    python -m server rf2 \\
        --input-quiver /scratch/results_mpnn/output/2_proteinmpnn.qv \\
        --output-dir /scratch/results_rf2/
"""

from __future__ import annotations

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import RFantibodyAdapter
from .models import ProteinMPNNRequest, RF2Request, RFdiffusionRequest
from .settings import RFantibodySettings
from .tools import proteinmpnn_argv, rf2_argv, rfdiffusion_argv

settings = RFantibodySettings()
adapter = RFantibodyAdapter(settings=settings)


def _rfdiffusion_build(req, inputs, job_dir, settings):
    return rfdiffusion_argv(req, inputs["target"], inputs["framework"], job_dir, settings)


def _proteinmpnn_build(req, inputs, job_dir, settings):
    return proteinmpnn_argv(req, inputs["input_quiver"], job_dir, settings)


def _rf2_build(req, inputs, job_dir, settings):
    return rf2_argv(req, inputs["input_quiver"], job_dir, settings)


endpoints = {
    "rfdiffusion": CLIEndpoint(
        name="rfdiffusion",
        help="RFdiffusion antibody-framework backbone design",
        request_model=RFdiffusionRequest,
        build_argv=_rfdiffusion_build,
        inputs={
            "target": ("Target antigen PDB file", True),
            "framework": ("Antibody framework PDB file", True),
        },
    ),
    "proteinmpnn": CLIEndpoint(
        name="proteinmpnn",
        help="ProteinMPNN CDR sequence design over RFdiffusion backbones",
        request_model=ProteinMPNNRequest,
        build_argv=_proteinmpnn_build,
        inputs={
            "input_quiver": ("Input Quiver file (from RFdiffusion)", True),
        },
    ),
    "rf2": CLIEndpoint(
        name="rf2",
        help="RF2 structure prediction + filtering over MPNN-designed sequences",
        request_model=RF2Request,
        build_argv=_rf2_build,
        inputs={
            "input_quiver": ("Input Quiver file (from ProteinMPNN)", True),
        },
    ),
}

create_cli(adapter, settings, endpoints, version="0.2.0")

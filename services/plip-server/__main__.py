"""CLI batch-mode entry point for plip-server (SIF / sbatch).

Usage::

    python -m server profile --input-pdb complex.pdb --output-dir /scratch/out/
    python -m server profile --input-pdb complex.pdb \
        --params-json '{"mode": "peptide", "peptide_chains": ["I"]}' \
        --output-dir out/
"""

from __future__ import annotations

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import PlipAdapter
from .models import ProfileRequest
from .settings import PlipSettings
from .tools import profile_argv

settings = PlipSettings()
adapter = PlipAdapter(settings=settings)


def _profile_build(req, inputs, job_dir, settings):
    return profile_argv(req, job_dir=job_dir, input_pdb=inputs["input_pdb"], settings=settings)


endpoints = {
    "profile": CLIEndpoint(
        name="profile",
        help="Profile non-covalent interactions in one PDB complex",
        request_model=ProfileRequest,
        build_argv=_profile_build,
        inputs={"input_pdb": ("Input PDB complex (protein + ligand)", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

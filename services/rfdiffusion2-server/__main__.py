"""CLI batch-mode entry point for rfdiffusion2-server.

Usage::

    python -m server.cli active_site --input-pdb /data/motif.pdb \\
        --ligand NAD --contigs "46,A106-106" \\
        --output-dir /scratch/results/
"""

from __future__ import annotations

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import RFdiffusion2Adapter
from .models import ActiveSiteRequest, CustomRequest, SmallMoleculeBinderRequest
from .settings import RFdiffusion2Settings
from .tools import active_site_argv, custom_argv, small_molecule_binder_argv

settings = RFdiffusion2Settings()
adapter = RFdiffusion2Adapter(settings=settings)


def _active_site_build(req, inputs, job_dir, settings):
    return active_site_argv(req, inputs["input_pdb"], job_dir, settings)


def _sm_binder_build(req, inputs, job_dir, settings):
    return small_molecule_binder_argv(req, inputs["input_pdb"], job_dir, settings)


def _custom_build(req, inputs, job_dir, settings):
    return custom_argv(req, inputs.get("input_pdb"), job_dir, settings)


endpoints = {
    "active_site": CLIEndpoint(
        name="active_site",
        help="Active-site scaffolding around an atomic motif + ligand",
        request_model=ActiveSiteRequest,
        build_argv=_active_site_build,
        inputs={"input_pdb": ("Input PDB with motif + ligand", True)},
    ),
    "small_molecule_binder": CLIEndpoint(
        name="small_molecule_binder",
        help="Small-molecule binder design, optionally RASA-conditioned",
        request_model=SmallMoleculeBinderRequest,
        build_argv=_sm_binder_build,
        inputs={"input_pdb": ("Input PDB with small molecule", True)},
    ),
    "custom": CLIEndpoint(
        name="custom",
        help="Raw contig + freeform Hydra overrides",
        request_model=CustomRequest,
        build_argv=_custom_build,
        inputs={"input_pdb": ("Optional input PDB", False)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

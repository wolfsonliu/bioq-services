"""CLI batch-mode entry point for lightdock-server.

Usage::

    python -m server dock \\
        --receptor /data/receptor.pdb --ligand /data/ligand.pdb \\
        --output-dir /scratch/results/ \\
        --swarms 20 --glowworms 100 --steps 50 --top 10
"""

from __future__ import annotations

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import LightdockAdapter
from .models import DockRequest
from .settings import LightdockSettings
from .tools import dock_argv

settings = LightdockSettings()
adapter = LightdockAdapter(settings=settings)


def _dock_build(req, inputs, job_dir, settings):
    return dock_argv(
        req,
        job_dir=job_dir,
        receptor_path=inputs["receptor"],
        ligand_path=inputs["ligand"],
        restraints_path=inputs.get("restraints"),
        settings=settings,
    )


endpoints = {
    "dock": CLIEndpoint(
        name="dock",
        help="Run the full LightDock GSO docking protocol",
        request_model=DockRequest,
        build_argv=_dock_build,
        inputs={
            "receptor": ("Receptor PDB file", True),
            "ligand": ("Ligand PDB file", True),
            "restraints": ("Optional LightDock restraints file", False),
        },
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

"""CLI batch-mode entry point for diffdock-pp-server.

Same Docker image runs the HTTP service (via uvicorn) and this CLI batch
mode (via `python -m server dock ...`).  See
engineering/decisions/2026-05-29-cli-batch-mode.md.

Usage::

    python -m server dock \\
        --receptor /data/input/receptor.pdb \\
        --ligand /data/input/ligand.pdb \\
        --output-dir /scratch/results/ \\
        --params-json '{"num_samples": 40, "top_k": 5}'
"""

from __future__ import annotations

from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import DiffDockPPAdapter
from .models import DockRequest
from .settings import DiffDockPPSettings
from .tools import dock_argv

settings = DiffDockPPSettings()
adapter = DiffDockPPAdapter(settings=settings)


def _dock_build(
    req: DockRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings: DiffDockPPSettings,
) -> list[str]:
    return dock_argv(
        req,
        job_dir=job_dir,
        receptor=inputs["receptor"],
        ligand=inputs["ligand"],
        settings=settings,
    )


endpoints = {
    "dock": CLIEndpoint(
        name="dock",
        help="Run rigid protein-protein docking on a pair of PDBs",
        request_model=DockRequest,
        build_argv=_dock_build,
        inputs={
            "receptor": ("Receptor protein (.pdb)", True),
            "ligand": ("Ligand protein (.pdb) — the shape that gets docked", True),
        },
    ),
}


create_cli(adapter, settings, endpoints, version="0.0.1")

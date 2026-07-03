"""CLI batch-mode entry point for turbohopp-server.

Same Docker image runs the HTTP service (via uvicorn) and this CLI batch
mode (via ``python -m server generate ...``).  See
engineering/decisions/2026-05-29-cli-batch-mode.md.

Usage::

    python -m server generate \\
        --protein /data/input/pocket.pdb \\
        --reference-ligand /data/input/ref.sdf \\
        --output-dir /scratch/results/ \\
        --params-json '{"num_samples": 10, "num_sampling_steps": 40}'
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import TurboHoppAdapter
from .models import GenerateRequest
from .settings import TurboHoppSettings
from .tools import generate_argv

settings = TurboHoppSettings()
adapter = TurboHoppAdapter(settings=settings)


def _generate_build(
    req: GenerateRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings: TurboHoppSettings,
) -> list[str]:
    return generate_argv(
        req,
        job_dir=job_dir,
        input_protein=inputs["protein"],
        input_molecule=inputs["reference_ligand"],
        settings=settings,
    )


endpoints = {
    "generate": CLIEndpoint(
        name="generate",
        help="Generate scaffold-hopping candidates from a protein pocket + reference ligand",
        request_model=GenerateRequest,
        build_argv=_generate_build,
        inputs={
            "protein": ("Input protein pocket (.pdb)", True),
            "reference_ligand": (
                "Input reference ligand (.sdf / .mol2 / .pdb)", True,
            ),
        },
    ),
}


create_cli(adapter, settings, endpoints, version="0.0.1")

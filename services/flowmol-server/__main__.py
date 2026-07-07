"""CLI batch-mode entry point for flowmol-server.

Same Docker image runs the HTTP service (via uvicorn) and this CLI batch
mode (via `python -m server generate ...`).  See
engineering/decisions/2026-05-29-cli-batch-mode.md.

Unconditional generation has **no input files** — every parameter flows
through the `GenerateRequest` pydantic model.

Usage::

    python -m server generate \\
        --n-mols 100 --n-timesteps 250 \\
        --model-variant flowmol3 \\
        --output-dir /scratch/results/

    # via --params-json (scripting):
    python -m server generate \\
        --params-json '{"n_mols": 50, "n_timesteps": 100, "seed": 42}' \\
        --output-dir ./output/
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import FlowMolAdapter
from .models import GenerateRequest
from .settings import FlowMolSettings
from .tools import generate_argv

settings = FlowMolSettings()
adapter = FlowMolAdapter(settings=settings)


def _generate_build(
    req: GenerateRequest,
    inputs: dict[str, Path],  # empty — no file inputs
    job_dir: Path,
    settings: FlowMolSettings,
) -> list[str]:
    return generate_argv(req, job_dir=job_dir, settings=settings)


endpoints = {
    "generate": CLIEndpoint(
        name="generate",
        help="Generate 3D small molecules unconditionally with FlowMol3",
        request_model=GenerateRequest,
        build_argv=_generate_build,
        inputs={},
    ),
}


create_cli(adapter, settings, endpoints, version="0.0.1")

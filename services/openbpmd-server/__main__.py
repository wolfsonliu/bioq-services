"""CLI batch-mode entry point for openbpmd-server.

HPC / sbatch invocation (long-running metadynamics on a GPU node)::

    apptainer exec --nv openbpmd-server.sif \\
        python -m server score \\
        --structure  solvated.rst7 \\
        --parameters solvated.prm7 \\
        --output-dir /scratch/$SLURM_JOB_ID/ \\
        --lig-resname MOL --nreps 10 --hill-height 0.3

Complex-parameter path::

    python -m server score \\
        --structure solvated.gro --parameters solvated.top \\
        --output-dir out/ \\
        --params-json '{"lig_resname": "LIG", "nreps": 10, "hill_height": 0.3}'
"""

from __future__ import annotations

from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import OpenBPMDAdapter
from .models import ScoreRequest
from .settings import OpenBPMDSettings
from .tools import score_argv

settings = OpenBPMDSettings()
adapter = OpenBPMDAdapter(settings=settings)


_INPUTS: dict[str, tuple[str, bool]] = {
    "structure": ("Coordinate file (.rst7 Amber or .gro Gromacs)", True),
    "parameters": ("Topology/parameter file (.prm7 Amber or .top Gromacs)", True),
}


def _score_build(req, inputs, job_dir: Path, s: OpenBPMDSettings) -> list[str]:
    return score_argv(
        req,
        job_dir=job_dir,
        structure=inputs["structure"],
        parameters=inputs["parameters"],
        settings=s,
    )


endpoints = {
    "score": CLIEndpoint(
        name="score",
        help="Binding pose metadynamics stability scoring via OpenBPMD",
        request_model=ScoreRequest,
        build_argv=_score_build,
        inputs=_INPUTS,
    ),
}


create_cli(adapter, settings, endpoints, version="0.0.1")

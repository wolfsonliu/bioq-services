"""CLI batch-mode entry point for bindflow-server.

Primary HPC-primary invocation::

    apptainer exec --nv bindflow-server.sif \\
        python -m server fep \\
        --protein receptor.pdb \\
        --ligands-dir ligands/ \\
        --output-dir /scratch/$SLURM_JOB_ID/ \\
        --replicas 3 --threads 8 --num-jobs 6

Complex-parameter path::

    python -m server mmpbsa \\
        --protein rec.pdb --ligands-dir ligands/ \\
        --output-dir out/ \\
        --params-json '{"samples": 20, "replicas": 3, "solv_ion_conc": 0.15}'

`ligands_dir` is registered as an input (not a repeatable file flag) — the CLI
framework `Path.exists()` check accepts directories.  All ligand SDF/MOL files
under that dir become one BindFlow job.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import BindFlowAdapter
from .models import FepCalculateRequest, MmpbsaCalculateRequest
from .settings import BindFlowSettings
from .tools import calculate_argv

settings = BindFlowSettings()
adapter = BindFlowAdapter(settings=settings)


_INPUTS: dict[str, tuple[str, bool]] = {
    "protein": ("Protein PDB file", True),
    "ligands_dir": ("Directory containing ligand SDF/MOL/MOL2 files", True),
    "cofactor": ("Cofactor SDF/MOL file (optional)", False),
    "membrane": ("Membrane PDB file (optional; requires CRYST1)", False),
    "custom_ff_dir": ("Custom force-field directory (optional; contains *.ff subdirs)", False),
    "topology_dir": ("Per-ligand GROMACS topology directory (optional)", False),
}


def _fep_build(req, inputs, job_dir: Path, s: BindFlowSettings) -> list[str]:
    return calculate_argv(
        req,
        calculation_type="fep",
        job_dir=job_dir,
        protein_path=inputs["protein"],
        ligands_dir=inputs["ligands_dir"],
        cofactor_path=inputs.get("cofactor"),
        membrane_path=inputs.get("membrane"),
        custom_ff_dir=inputs.get("custom_ff_dir"),
        topology_dir=inputs.get("topology_dir"),
        settings=s,
    )


def _mmpbsa_build(req, inputs, job_dir: Path, s: BindFlowSettings) -> list[str]:
    return calculate_argv(
        req,
        calculation_type="mmpbsa",
        job_dir=job_dir,
        protein_path=inputs["protein"],
        ligands_dir=inputs["ligands_dir"],
        cofactor_path=inputs.get("cofactor"),
        membrane_path=inputs.get("membrane"),
        custom_ff_dir=inputs.get("custom_ff_dir"),
        topology_dir=inputs.get("topology_dir"),
        settings=s,
    )


endpoints = {
    "fep": CLIEndpoint(
        name="fep",
        help="Run FEP calculation on a set of ligands via BindFlow",
        request_model=FepCalculateRequest,
        build_argv=_fep_build,
        inputs=_INPUTS,
    ),
    "mmpbsa": CLIEndpoint(
        name="mmpbsa",
        help="Run MM(P/G)BSA calculation on a set of ligands via BindFlow",
        request_model=MmpbsaCalculateRequest,
        build_argv=_mmpbsa_build,
        inputs=_INPUTS,
    ),
}


create_cli(adapter, settings, endpoints, version="0.0.1")

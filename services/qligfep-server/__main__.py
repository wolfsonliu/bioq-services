"""CLI batch-mode entry for qligfep-server.

Usage::

    python -m server ligprep --ligand /data/17.mol2 --ligand-name 17 --output-dir /out
    python -m server run-fep --setup-zip setup.zip --window-idx 5 --leg protein \\
        --device gpu --output-dir /scratch/win5

Each subcommand mirrors the corresponding HTTP endpoint.  See engineering/
decisions/2026-05-29-cli-batch-mode.md for the design.
"""
from __future__ import annotations

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import QligfepAdapter
from .models import (
    AnalyzeFepRequest, AnalyzeLieRequest, CogRequest, LigprepRequest,
    ProtprepRequest, RunFepRequest, SetupLieRequest, SetupLigfepRequest,
    SetupResfepRequest,
)
from .settings import QligfepSettings
from .tools import (
    analyze_fep_argv, analyze_lie_argv, cog_argv, ligprep_argv,
    protprep_argv, run_fep_argv, setup_lie_argv, setup_ligfep_argv,
    setup_resfep_argv,
)

settings = QligfepSettings()
adapter = QligfepAdapter(settings=settings)


def _ligprep_build(req, inputs, job_dir, settings):
    return ligprep_argv(req, inputs["ligand"], job_dir, settings)


def _protprep_build(req, inputs, job_dir, settings):
    return protprep_argv(req, inputs["protein_pdb"], job_dir, settings)


def _cog_build(req, inputs, job_dir, settings):
    return cog_argv(req, inputs["pdb"], job_dir, settings)


def _setup_ligfep_build(req, inputs, job_dir, settings):
    return setup_ligfep_argv(req, inputs["ligprep_dir"], inputs["protprep_dir"],
                              job_dir, settings)


def _setup_resfep_build(req, inputs, job_dir, settings):
    return setup_resfep_argv(req, inputs["protprep_dir"], job_dir, settings)


def _setup_lie_build(req, inputs, job_dir, settings):
    return setup_lie_argv(req, inputs["ligprep_dir"], inputs["protprep_dir"],
                           job_dir, settings)


def _run_fep_build(req, inputs, job_dir, settings):
    return run_fep_argv(req, inputs["setup_dir"], job_dir, settings)


def _analyze_fep_build(req, inputs, job_dir, settings):
    return analyze_fep_argv(req, inputs["run_dir"], job_dir, settings)


def _analyze_lie_build(req, inputs, job_dir, settings):
    return analyze_lie_argv(req, inputs["run_dir"], job_dir, settings)


endpoints = {
    "ligprep": CLIEndpoint(
        name="ligprep", help="OpenFF-based ligand parameterization",
        request_model=LigprepRequest, build_argv=_ligprep_build,
        inputs={"ligand": ("Input ligand file (.mol2/.sdf/.pdb)", True)},
    ),
    "protprep": CLIEndpoint(
        name="protprep", help="Spherical boundary protein prep",
        request_model=ProtprepRequest, build_argv=_protprep_build,
        inputs={"protein_pdb": ("Input protein PDB", True)},
    ),
    "cog": CLIEndpoint(
        name="cog", help="Center of geometry helper",
        request_model=CogRequest, build_argv=_cog_build,
        inputs={"pdb": ("Input PDB (protein or ligand)", True)},
    ),
    "setup-ligfep": CLIEndpoint(
        name="setup-ligfep", help="QligFEP dual-topology setup",
        request_model=SetupLigfepRequest, build_argv=_setup_ligfep_build,
        inputs={
            "ligprep_dir": ("Directory with ligand param files", True),
            "protprep_dir": ("Directory with protprep output", True),
        },
    ),
    "setup-resfep": CLIEndpoint(
        name="setup-resfep", help="QresFEP residue mutation setup",
        request_model=SetupResfepRequest, build_argv=_setup_resfep_build,
        inputs={"protprep_dir": ("Directory with protprep output", True)},
    ),
    "setup-lie": CLIEndpoint(
        name="setup-lie", help="QLIE setup",
        request_model=SetupLieRequest, build_argv=_setup_lie_build,
        inputs={
            "ligprep_dir": ("Directory with ligand param files", True),
            "protprep_dir": ("Directory with protprep output", True),
        },
    ),
    "run-fep": CLIEndpoint(
        name="run-fep",
        help="Run one lambda window (qprep + eq + md sequence via qdyn/qdynp/qdyn_cuda)",
        request_model=RunFepRequest, build_argv=_run_fep_build,
        inputs={"setup_dir": ("Setup output directory (1.protein or 2.water)", True)},
    ),
    "analyze-fep": CLIEndpoint(
        name="analyze-fep", help="FEP DDG post-processing (Zwanzig / OS / BAR)",
        request_model=AnalyzeFepRequest, build_argv=_analyze_fep_build,
        inputs={"run_dir": ("Directory with run-fep outputs", True)},
    ),
    "analyze-lie": CLIEndpoint(
        name="analyze-lie", help="LIE post-processing",
        request_model=AnalyzeLieRequest, build_argv=_analyze_lie_build,
        inputs={"run_dir": ("Directory with LIE run outputs", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

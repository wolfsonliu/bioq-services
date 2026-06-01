"""CLI batch-mode entry point for ppiflow-server.

Usage::

    python -m server.cli binder --target /data/target.pdb \\
        --samples-per-target 10 --output-dir /scratch/results/

    python -m server.cli antibody --antigen /data/antigen.pdb \\
        --framework /data/framework.pdb --output-dir /scratch/results/

    python -m server.cli monomer --output-dir /scratch/results/
"""

from __future__ import annotations

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import PPIFlowAdapter
from .models import (
    AntibodyRequest,
    BinderRequest,
    MonomerRequest,
    NanobodyRequest,
    ScaffoldingRequest,
)
from .settings import PPIFlowSettings
from .tools import (
    antibody_argv,
    binder_argv,
    monomer_argv,
    nanobody_argv,
    scaffolding_argv,
)

settings = PPIFlowSettings()
adapter = PPIFlowAdapter(settings=settings)


def _binder_build(req, inputs, job_dir, settings):
    return binder_argv(req, inputs["target"], job_dir, settings)


def _antibody_build(req, inputs, job_dir, settings):
    return antibody_argv(req, inputs["antigen"], inputs["framework"], job_dir, settings)


def _nanobody_build(req, inputs, job_dir, settings):
    return nanobody_argv(req, inputs["antigen"], inputs["framework"], job_dir, settings)


def _monomer_build(req, inputs, job_dir, settings):
    return monomer_argv(req, job_dir, settings)


def _scaffolding_build(req, inputs, job_dir, settings):
    return scaffolding_argv(req, inputs["motif_csv"], job_dir, settings)


endpoints = {
    "binder": CLIEndpoint(
        name="binder",
        help="PPI binder design against a target PDB",
        request_model=BinderRequest,
        build_argv=_binder_build,
        inputs={"target": ("Target PDB file", True)},
    ),
    "antibody": CLIEndpoint(
        name="antibody",
        help="Antibody (heavy + light) CDR design",
        request_model=AntibodyRequest,
        build_argv=_antibody_build,
        inputs={
            "antigen": ("Antigen PDB file", True),
            "framework": ("Antibody framework PDB file", True),
        },
    ),
    "nanobody": CLIEndpoint(
        name="nanobody",
        help="VHH nanobody CDR design",
        request_model=NanobodyRequest,
        build_argv=_nanobody_build,
        inputs={
            "antigen": ("Antigen PDB file", True),
            "framework": ("Nanobody framework PDB file", True),
        },
    ),
    "monomer": CLIEndpoint(
        name="monomer",
        help="Unconditional monomer generation",
        request_model=MonomerRequest,
        build_argv=_monomer_build,
    ),
    "scaffolding": CLIEndpoint(
        name="scaffolding",
        help="Motif scaffolding from CSV + motif PDBs",
        request_model=ScaffoldingRequest,
        build_argv=_scaffolding_build,
        inputs={"motif_csv": ("Motif metadata CSV file", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

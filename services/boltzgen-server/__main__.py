"""CLI batch-mode entry point for boltzgen-server.

Usage::

    python -m server.cli design --design-yaml /data/spec.yaml \\
        --output-dir /scratch/results/

    python -m server.cli inverse_fold --design-yaml /data/spec.yaml \\
        --output-dir /scratch/results/

Note: Reference CIF/PDB files referenced in the YAML should be accessible
by absolute path from within the container.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import BoltzGenAdapter
from .models import DesignRequest, InverseFoldRequest
from .settings import BoltzGenSettings
from .tools import design_argv, inverse_fold_argv

settings = BoltzGenSettings()
adapter = BoltzGenAdapter(settings=settings)


def _design_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = input_dir / "design_spec.yaml"
    shutil.copy2(inputs["design_yaml"], yaml_path)
    return design_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)


def _inverse_fold_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = input_dir / "design_spec.yaml"
    shutil.copy2(inputs["design_yaml"], yaml_path)
    return inverse_fold_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)


endpoints = {
    "design": CLIEndpoint(
        name="design",
        help="Full BoltzGen binder design pipeline",
        request_model=DesignRequest,
        build_argv=_design_build,
        inputs={"design_yaml": ("Design specification YAML file", True)},
    ),
    "inverse_fold": CLIEndpoint(
        name="inverse_fold",
        help="BoltzGen inverse-fold-only mode",
        request_model=InverseFoldRequest,
        build_argv=_inverse_fold_build,
        inputs={"design_yaml": ("Design specification YAML file", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

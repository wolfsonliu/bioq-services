"""CLI batch-mode entry point for boltzgen-server.

Usage::

    python -m server design --design-yaml /data/spec.yaml \\
        --output-dir /scratch/results/

    python -m server inverse_fold --design-yaml /data/spec.yaml \\
        --output-dir /scratch/results/

Reference CIF/PDB files sitting alongside the design YAML are automatically
copied into the job directory so relative paths resolve correctly.
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


def _copy_yaml_with_refs(src_yaml: Path, input_dir: Path) -> Path:
    """Copy a design YAML and its sibling files into *input_dir*.

    boltzgen resolves relative ``path:`` entries against the YAML's parent
    directory.  When we relocate the YAML into a job-local input dir we must
    bring the referenced files along so those relative paths still resolve.
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_dest = input_dir / "design_spec.yaml"
    shutil.copy2(src_yaml, yaml_dest)

    src_dir = src_yaml.parent
    for item in src_dir.iterdir():
        if item.resolve() == src_yaml.resolve():
            continue
        dest = input_dir / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    return yaml_dest


def _design_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    yaml_path = _copy_yaml_with_refs(inputs["design_yaml"], input_dir)
    return design_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)


def _inverse_fold_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    yaml_path = _copy_yaml_with_refs(inputs["design_yaml"], input_dir)
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

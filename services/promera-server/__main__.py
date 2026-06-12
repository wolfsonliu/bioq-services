"""CLI batch-mode entry point for promera-server.

Usage::

    python -m server cofold \
        --input-schema /data/input.json \
        --output-dir /scratch/results/

    python -m server design \
        --target-schema /data/target.json \
        --output-dir /scratch/results/ \
        --design-type vhh --num-backbones 100
"""

from __future__ import annotations

import shutil

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import PromeraAdapter
from .models import CofoldRequest, DesignRequest
from .settings import PromeraSettings
from .tools import (
    build_design_config,
    cofold_argv,
    design_argv,
    write_design_config,
)

settings = PromeraSettings()
adapter = PromeraAdapter(settings=settings)


def _cofold_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    schema_path = input_dir / "input.json"
    shutil.copy2(inputs["input_schema"], schema_path)
    return cofold_argv(
        req, job_dir=job_dir, schema_path=schema_path, settings=settings
    )


def _design_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    target_dir = input_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(inputs["target_schema"], target_dir / "target.json")

    template_path = None
    if inputs.get("target_template"):
        template_path = input_dir / "target_template.cif"
        shutil.copy2(inputs["target_template"], template_path)

    cfg = build_design_config(
        req,
        target_dir=target_dir,
        output_dir=output_dir,
        template_path=template_path,
        settings=settings,
    )
    config_path = write_design_config(cfg, input_dir / "task_config.yaml")

    return design_argv(
        req, job_dir=job_dir, config_path=config_path, settings=settings
    )


endpoints = {
    "cofold": CLIEndpoint(
        name="cofold",
        help="Run structure prediction on a tinyprot JSON schema",
        request_model=CofoldRequest,
        build_argv=_cofold_build,
        inputs={"input_schema": ("Input JSON schema file", True)},
    ),
    "design": CLIEndpoint(
        name="design",
        help="Run de novo binder design on a target JSON schema",
        request_model=DesignRequest,
        build_argv=_design_build,
        inputs={
            "target_schema": ("Target JSON schema file", True),
            "target_template": ("Target template CIF file (optional)", False),
        },
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

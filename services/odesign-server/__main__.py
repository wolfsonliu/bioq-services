"""CLI batch-mode entry point for odesign-server.

Usage::

    python -m server.cli design --input-json /data/spec.json \\
        --model odesign_base_prot_flex --output-dir /scratch/results/

Note: Reference CIF/PDB files referenced in the JSON should be accessible
by absolute path from within the container.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import ODesignAdapter
from .models import DesignRequest
from .settings import ODesignSettings
from .tools import design_argv

settings = ODesignSettings()
adapter = ODesignAdapter(settings=settings)


def _design_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    json_path = input_dir / "input.json"
    shutil.copy2(inputs["input_json"], json_path)
    return design_argv(req, job_dir=job_dir, json_path=json_path, settings=settings)


endpoints = {
    "design": CLIEndpoint(
        name="design",
        help="ODesign biomolecular interaction design",
        request_model=DesignRequest,
        build_argv=_design_build,
        inputs={"input_json": ("JSON specification file", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

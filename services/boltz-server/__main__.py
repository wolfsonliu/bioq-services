"""CLI batch-mode entry point for boltz-server.

Usage::

    python -m server.cli predict_structure \\
        --raw-yaml /data/input.yaml \\
        --output-dir /scratch/results/

    python -m server.cli predict_structure \\
        --params-json '{"sequences": [{"type": "protein", "id": "A", "sequence": "MKTL..."}], "msa_mode": "empty"}' \\
        --output-dir /scratch/results/

Note: For CLI mode, use --raw-yaml to supply a pre-built Boltz YAML or
--params-json to provide the structured request parameters. MSA and template
files should be referenced by absolute path in the YAML.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import BoltzAdapter
from .models import PredictAffinityRequest, PredictStructureRequest
from .settings import BoltzSettings
from .tools import build_yaml, predict_argv

settings = BoltzSettings()
adapter = BoltzAdapter(settings=settings)


def _predict_build(req, inputs, job_dir, settings):
    yaml_path = inputs.get("raw_yaml")
    if yaml_path is not None:
        import shutil

        dest = job_dir / "input" / "input.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(yaml_path, dest)
        req.raw_yaml = dest.read_text(encoding="utf-8")

    yaml_path = build_yaml(
        req,
        job_dir=job_dir,
        settings=settings,
        saved_msa_paths={},
        saved_template_paths={},
    )
    return predict_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)


endpoints = {
    "predict_structure": CLIEndpoint(
        name="predict_structure",
        help="Predict 3D structure of a biomolecular complex",
        request_model=PredictStructureRequest,
        build_argv=_predict_build,
        inputs={"raw_yaml": ("Pre-built Boltz YAML input file", False)},
    ),
    "predict_affinity": CLIEndpoint(
        name="predict_affinity",
        help="Predict structure + ligand binding affinity",
        request_model=PredictAffinityRequest,
        build_argv=_predict_build,
        inputs={"raw_yaml": ("Pre-built Boltz YAML input file", False)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

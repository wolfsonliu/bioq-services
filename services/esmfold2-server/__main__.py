"""CLI batch-mode entry point for esmfold2-server.

Usage::

    python -m server fold \
        --params-json '{"sequences": [{"type": "protein", "id": "A", "sequence": "MKT..."}]}' \
        --output-dir /scratch/results/
"""

from __future__ import annotations

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import ESMFold2Adapter
from .models import FoldRequest
from .settings import ESMFold2Settings
from .tools import build_input_json, fold_argv

settings = ESMFold2Settings()
adapter = ESMFold2Adapter(settings=settings)


def _fold_build(req, inputs, job_dir, settings):
    input_json = build_input_json(
        req,
        job_dir=job_dir,
        saved_msa_paths={},
    )
    return fold_argv(req, job_dir=job_dir, input_json=input_json, settings=settings)


endpoints = {
    "fold": CLIEndpoint(
        name="fold",
        help="Predict 3D structure of a biomolecular complex",
        request_model=FoldRequest,
        build_argv=_fold_build,
        inputs={},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

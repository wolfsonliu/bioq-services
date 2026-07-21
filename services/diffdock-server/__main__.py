"""CLI batch-mode entry point for diffdock-server.

Same Docker image runs the HTTP service (via uvicorn) and this CLI batch
mode (via ``python -m server dock ...``).  See
engineering/decisions/2026-05-29-cli-batch-mode.md.

Usage::

    # PDB + SDF file
    python -m server dock \\
        --protein /data/target.pdb \\
        --ligand  /data/ligand.sdf \\
        --output-dir /scratch/results/ \\
        --params-json '{"complex_name": "1a0q", "samples_per_complex": 10}'

    # PDB + SMILES (SMILES goes in --params-json.ligand_description;
    # omit --ligand file when using SMILES)
    python -m server dock \\
        --protein /data/target.pdb \\
        --output-dir /scratch/results/ \\
        --params-json '{"ligand_description": "COc(cc1)ccc1C#N", "complex_name": "example"}'

    # Sequence + SMILES (both are text params in --params-json;
    # omit both file uploads)
    python -m server dock \\
        --output-dir /scratch/results/ \\
        --params-json '{"protein_sequence": "MKW...", "ligand_description": "CCO", "complex_name": "novel"}'

Build helpers live in ``cli_impl.py`` so tests can import them without
triggering ``create_cli`` (which parses ``sys.argv``).
"""

from __future__ import annotations

from bioq_service.cli import create_cli

from .adapter import DiffdockAdapter
from .cli_impl import build_endpoints
from .settings import DiffdockSettings

settings = DiffdockSettings()
adapter = DiffdockAdapter(settings=settings)

create_cli(adapter, settings, build_endpoints(), version="0.0.1")

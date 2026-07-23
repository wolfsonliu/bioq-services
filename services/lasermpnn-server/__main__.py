"""CLI batch-mode entry point for lasermpnn-server.

Usage::

    python -m server design \\
        --pdb /data/complex_with_ligand.pdb \\
        --designs-per-input 5 --sequence-temp 0.3 \\
        --output-dir /scratch/results/

Same argv builders as the HTTP app; no FastAPI/uvicorn. The input PDB must
carry hydrogens on the ligand (LASErMPNN was trained on protonated structures).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import LASErMPNNAdapter
from .models import DesignLigandMPNNRequest, DesignRequest
from .settings import LASErMPNNSettings
from .tools import design_argv, design_ligandmpnn_argv

settings = LASErMPNNSettings()
adapter = LASErMPNNAdapter(settings=settings)


def _copy_input(inputs, job_dir: Path) -> Path:
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / "input.pdb"
    shutil.copy2(inputs["pdb"], dest)
    return dest


def _build_design(req, inputs, job_dir, settings):
    input_pdb = _copy_input(inputs, job_dir)
    return design_argv(req, input_pdb=input_pdb, job_dir=job_dir, settings=settings)


def _build_design_ligandmpnn(req, inputs, job_dir, settings):
    input_pdb = _copy_input(inputs, job_dir)
    return design_ligandmpnn_argv(req, input_pdb=input_pdb, job_dir=job_dir, settings=settings)


endpoints = {
    "design": CLIEndpoint(
        name="design",
        help="LASErMPNN ligand-conditioned batch sequence design",
        request_model=DesignRequest,
        build_argv=_build_design,
        inputs={"pdb": ("Input PDB with a protonated ligand", True)},
    ),
    "design_ligandmpnn": CLIEndpoint(
        name="design_ligandmpnn",
        help="Retrained-LigandMPNN variant batch design",
        request_model=DesignLigandMPNNRequest,
        build_argv=_build_design_ligandmpnn,
        inputs={"pdb": ("Input PDB with a protonated ligand", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

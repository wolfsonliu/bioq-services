"""CLI batch-mode entry point for proteinmpnn-server.

Usage::

    python -m server design --pdb /data/structure.pdb \\
        --chains-to-design "A,B" --output-dir /scratch/results/

Note: The ProteinMPNN CLI runs helper scripts synchronously to prepare JSONL
inputs (parse_multiple_chains, assign_fixed_chains, etc.) before launching
the main protein_mpnn_run.py subprocess. The input PDB is copied into
<job_dir>/input/ and helper scripts are run from there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import ProteinMPNNAdapter
from .models import DesignRequest, ProbsRequest, ScoreRequest
from .settings import ProteinMPNNSettings
from .tools import design_argv, prepare_inputs, probs_argv, score_argv

settings = ProteinMPNNSettings()
adapter = ProteinMPNNAdapter(settings=settings)


def _prepare_and_design(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs["pdb"], input_dir / inputs["pdb"].name)
    paths = prepare_inputs(
        job_dir,
        settings=settings,
        ca_only=(req.model_variant == "ca_only"),
        chains_to_design=req.chains_to_design,
        fixed_positions=req.fixed_positions,
        tied_positions=req.tied_positions,
        homooligomer=req.homooligomer,
        bias_AA=req.bias_AA,
        bias_by_res=req.bias_by_res,
        omit_AA_per_chain=req.omit_AA_per_chain,
    )
    return design_argv(req, job_dir=job_dir, paths=paths, settings=settings)


def _prepare_and_score(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs["pdb"], input_dir / inputs["pdb"].name)
    paths = prepare_inputs(
        job_dir,
        settings=settings,
        ca_only=(req.model_variant == "ca_only"),
        chains_to_design=req.chains_to_design,
        fixed_positions=None,
        tied_positions=None,
        homooligomer=False,
        bias_AA=None,
        bias_by_res=None,
        omit_AA_per_chain=None,
    )
    return score_argv(req, job_dir=job_dir, paths=paths, settings=settings)


def _prepare_and_probs(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs["pdb"], input_dir / inputs["pdb"].name)
    paths = prepare_inputs(
        job_dir,
        settings=settings,
        ca_only=(req.model_variant == "ca_only"),
        chains_to_design=req.chains_to_design,
        fixed_positions=None,
        tied_positions=None,
        homooligomer=False,
        bias_AA=None,
        bias_by_res=None,
        omit_AA_per_chain=None,
    )
    return probs_argv(req, job_dir=job_dir, paths=paths, settings=settings)


endpoints = {
    "design": CLIEndpoint(
        name="design",
        help="Sequence design (FASTA output) over the input PDB",
        request_model=DesignRequest,
        build_argv=_prepare_and_design,
        inputs={"pdb": ("Input PDB file", True)},
    ),
    "score": CLIEndpoint(
        name="score",
        help="Score a (structure, sequence) pair",
        request_model=ScoreRequest,
        build_argv=_prepare_and_score,
        inputs={"pdb": ("Input PDB file", True)},
    ),
    "probs": CLIEndpoint(
        name="probs",
        help="Per-residue AA probability output",
        request_model=ProbsRequest,
        build_argv=_prepare_and_probs,
        inputs={"pdb": ("Input PDB file", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")

"""CLI batch-mode entry for reinvent-server.

Usage::

    python -m server sampling --generator reinvent --num-smiles 100 --output-dir /out
    python -m server staged-learning --params-json rl.json --output-dir /scratch/rl

See engineering/decisions/2026-05-29-cli-batch-mode.md.
"""
from __future__ import annotations

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import ReinventAdapter
from .models import (
    EnumerationRequest, SamplingRequest, ScoringRequest,
    StagedLearningRequest, TransferLearningRequest,
)
from .settings import ReinventSettings
from .tools import (
    enumeration_argv, sampling_argv, scoring_argv,
    staged_learning_argv, transfer_learning_argv,
)

settings = ReinventSettings()
adapter = ReinventAdapter(settings=settings)


def _sampling_build(req, inputs, job_dir, settings):
    return sampling_argv(req, {"smiles_file": inputs.get("smiles_file")}, job_dir, settings)


def _scoring_build(req, inputs, job_dir, settings):
    return scoring_argv(req, {"smiles_file": inputs.get("smiles_file")}, job_dir, settings)


def _enumeration_build(req, inputs, job_dir, settings):
    return enumeration_argv(req, {
        "smiles_file": inputs.get("peptide_smiles"),
        "amino_acid_library": inputs.get("amino_acid_library"),
    }, job_dir, settings)


def _tl_build(req, inputs, job_dir, settings):
    return transfer_learning_argv(req, {
        "smiles_file": inputs.get("smiles_file"),
        "validation_smiles_file": inputs.get("validation_smiles_file"),
        "model_file": inputs.get("input_model"),
    }, job_dir, settings)


def _rl_build(req, inputs, job_dir, settings):
    return staged_learning_argv(req, {
        "smiles_file": inputs.get("smiles_file"),
        "prior_file": inputs.get("prior"),
        "agent_file": inputs.get("agent"),
    }, job_dir, settings)


endpoints = {
    "sampling": CLIEndpoint(
        name="sampling", help="Sample molecules from a prior",
        request_model=SamplingRequest, build_argv=_sampling_build,
        inputs={"smiles_file": ("Seed SMILES (Lib/Link/Mol2Mol/Pep only)", False)},
    ),
    "scoring": CLIEndpoint(
        name="scoring", help="Score SMILES with a scoring function",
        request_model=ScoringRequest, build_argv=_scoring_build,
        inputs={"smiles_file": ("SMILES to score (.smi/CSV)", True)},
    ),
    "enumeration": CLIEndpoint(
        name="enumeration", help="Peptide enumeration",
        request_model=EnumerationRequest, build_argv=_enumeration_build,
        inputs={
            "peptide_smiles": ("Masked peptide template", True),
            "amino_acid_library": ("Amino acid library CSV", True),
        },
    ),
    "transfer-learning": CLIEndpoint(
        name="transfer-learning", help="Fine-tune a model on target SMILES",
        request_model=TransferLearningRequest, build_argv=_tl_build,
        inputs={
            "smiles_file": ("Target SMILES", True),
            "validation_smiles_file": ("Validation SMILES", False),
            "input_model": ("Explicit starting model (overrides input_model_file)", False),
        },
    ),
    "staged-learning": CLIEndpoint(
        name="staged-learning", help="Reinforcement / curriculum learning",
        request_model=StagedLearningRequest, build_argv=_rl_build,
        inputs={
            "smiles_file": ("Seed SMILES (Lib/Link/Mol2Mol/Pep only)", False),
            "prior": ("Explicit prior model", False),
            "agent": ("Explicit agent / checkpoint model", False),
        },
    ),
}

if __name__ == "__main__":
    create_cli(adapter, settings, endpoints, version="0.0.3")

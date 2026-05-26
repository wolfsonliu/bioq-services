"""Pydantic request models for odesign-server.

Single endpoint: POST /api/design — unified biomolecular interaction design.
Model variant and design modality are selected via request parameters.
"""

from __future__ import annotations

from typing import Literal, Optional

from bioagent_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exports)
from pydantic import BaseModel, Field

__all__ = [
    "DesignRequest",
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "MODELS",
]

MODELS = Literal[
    "odesign_base_prot_flex",
    "odesign_base_prot_rigid",
    "odesign_base_ligand_rigid",
    "odesign_base_na_rigid",
]

DESIGN_MODALITIES = Literal["protein", "ligand", "dna", "rna"]


class DesignRequest(BaseModel):
    """`POST /api/design` — unified biomolecular interaction design."""

    model: MODELS = Field(
        default="odesign_base_prot_flex",
        description=(
            "Model variant: "
            "odesign_base_prot_flex (protein, flex receptor), "
            "odesign_base_prot_rigid (protein, rigid receptor), "
            "odesign_base_ligand_rigid (ligand design), "
            "odesign_base_na_rigid (DNA/RNA design)."
        ),
    )
    design_modality: Optional[DESIGN_MODALITIES] = Field(
        default=None,
        description=(
            "Design modality: protein/ligand/dna/rna. "
            "Auto-inferred from model name for prot/ligand models. "
            "REQUIRED for na_rigid (must specify 'dna' or 'rna')."
        ),
    )
    n_sample: int = Field(
        default=5,
        ge=1,
        le=200,
        description="Number of backbone samples per seed.",
    )
    seeds: str = Field(
        default="[42]",
        description="Random seed list in JSON array format, e.g. '[42]' or '[42,101,777]'.",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        le=16,
        description="Number of dataloader workers.",
    )
    invfold_topk: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of inverse-folded sequence variants per backbone.",
    )
    invfold_temp: float = Field(
        default=1.0,
        gt=0.0,
        le=5.0,
        description="Inverse folding sampling temperature.",
    )
    enable_partial_diff: bool = Field(
        default=False,
        description=(
            "Enable partial diffusion mode. "
            "Requires 'partial_diff' field in the input JSON spec."
        ),
    )
    partial_diff_snr: float = Field(
        default=0.1,
        gt=0.0,
        le=10.0,
        description="Signal-to-noise ratio for partial diffusion.",
    )

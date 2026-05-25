"""Pydantic request models for boltzgen-server.

Two endpoints: POST /api/design (full pipeline) and POST /api/inverse_fold
(inverse-fold-only mode). Both share common fields via `_BoltzGenCommon`.
"""

from __future__ import annotations

from typing import Literal, Optional

from bioagent_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exports)
from pydantic import BaseModel, Field

__all__ = [
    "DesignRequest",
    "FailureKind",
    "InverseFoldRequest",
    "JobInfo",
    "JobStatus",
]

PROTOCOLS = Literal[
    "protein-anything",
    "peptide-anything",
    "protein-small_molecule",
    "nanobody-anything",
    "antibody-anything",
    "protein-redesign",
]


class _BoltzGenCommon(BaseModel):
    """Fields shared across /api/design and /api/inverse_fold."""

    name: str = Field(
        default="run",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$",
        description="Output name prefix.",
    )
    protocol: PROTOCOLS = Field(
        default="protein-anything",
        description="Design protocol — determines default settings for filtering/analysis.",
    )
    num_designs: int = Field(
        default=100,
        ge=1,
        le=100000,
        description="Number of designs to generate (upstream default 10000; FC recommended 50-500).",
    )
    budget: int = Field(
        default=30,
        ge=1,
        le=10000,
        description="Number of final designs after diversity-quality filtering.",
    )
    diffusion_batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Diffusion samples per trunk run. None=auto (1 if num_designs<100, else 10).",
    )
    step_scale: Optional[str] = Field(
        default=None,
        description="Fixed step scale (e.g. '1.8'). Default uses a schedule.",
    )
    noise_scale: Optional[str] = Field(
        default=None,
        description="Fixed noise scale (e.g. '0.98'). Default uses a schedule.",
    )
    use_kernels: Literal["auto", "true", "false"] = Field(
        default="auto",
        description="cuequivariance kernel usage. 'auto' detects GPU capability >= 8.",
    )
    skip_inverse_folding: bool = Field(
        default=False,
        description="Skip inverse folding step.",
    )
    inverse_fold_num_sequences: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of sequences per backbone in inverse fold step.",
    )
    inverse_fold_avoid: Optional[str] = Field(
        default=None,
        description="Disallowed residues as one-letter AA codes (e.g. 'C' to avoid Cys). "
        "Default: 'C' for peptide/nanobody/antibody protocols, none for others.",
    )
    alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Diversity/quality trade-off for filtering (0=quality-only, 1=diversity-only).",
    )
    filter_biased: Literal["true", "false"] = Field(
        default="true",
        description="Remove amino-acid composition outliers.",
    )
    reuse: bool = Field(
        default=False,
        description="Reuse existing results (resume interrupted pipeline).",
    )


class DesignRequest(_BoltzGenCommon):
    """`POST /api/design` — full binder design pipeline."""
    pass


class InverseFoldRequest(_BoltzGenCommon):
    """`POST /api/inverse_fold` — inverse folding only mode."""
    pass

"""Per-endpoint pydantic request models for immunebuilder-server.

Three request models for the three predictors (antibody / nanobody / tcr),
sharing common fields via `_PredictCommon`.
"""

from __future__ import annotations

from typing import Literal

from bioagent_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exports)
from pydantic import BaseModel, Field

__all__ = [
    "AntibodyRequest",
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "NanobodyRequest",
    "TCRRequest",
]

_AA_PATTERN = r"^[ACDEFGHIKLMNPQRSTVWY]+$"


class _PredictCommon(BaseModel):
    """Fields shared across all three predict endpoints."""

    name: str = Field(
        default="prediction",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$",
    )
    numbering_scheme: Literal[
        "imgt", "chothia", "kabat", "aho", "wolfguy", "martin", "raw"
    ] = "imgt"
    save_all_models: bool = Field(
        default=True,
        description="Save all 4 ensemble models + error estimates (--to_directory)",
    )
    no_sidechain_bond_check: bool = Field(
        default=False,
        description="Skip strained-bond check during refinement (-u flag)",
    )
    n_threads: int = Field(
        default=-1,
        ge=-1,
        description="OpenMM refinement threads; >0 forces CPU refinement; -1 uses GPU",
    )


class AntibodyRequest(_PredictCommon):
    """`POST /api/predict_antibody` — antibody structure prediction (H + L)."""

    heavy_sequence: str = Field(min_length=20, pattern=_AA_PATTERN)
    light_sequence: str = Field(min_length=20, pattern=_AA_PATTERN)


class NanobodyRequest(_PredictCommon):
    """`POST /api/predict_nanobody` — nanobody structure prediction (H only)."""

    heavy_sequence: str = Field(min_length=20, pattern=_AA_PATTERN)


class TCRRequest(_PredictCommon):
    """`POST /api/predict_tcr` — TCR structure prediction (A + B)."""

    alpha_sequence: str = Field(min_length=20, pattern=_AA_PATTERN)
    beta_sequence: str = Field(min_length=20, pattern=_AA_PATTERN)

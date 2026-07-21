"""Pydantic request models for deeprank-ab-server.

Single endpoint: POST /api/score — score an antibody-antigen docking complex.
"""

from __future__ import annotations

from typing import Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exports)
from pydantic import BaseModel, Field

__all__ = [
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "ScoreRequest",
]


class ScoreRequest(BaseModel):
    """`POST /api/score` — score an antibody-antigen complex PDB."""

    heavy_chain_id: str = Field(
        default="H",
        min_length=1,
        max_length=2,
        description="PDB chain ID for the heavy chain.",
    )
    light_chain_id: str = Field(
        default="L",
        max_length=2,
        description=(
            "PDB chain ID for the light chain. "
            "Use '-' (or 'none'/'null') for nanobodies / VHH without a light chain."
        ),
    )
    antigen_chain_id: str = Field(
        default="A",
        min_length=1,
        max_length=2,
        description="PDB chain ID for the antigen.",
    )

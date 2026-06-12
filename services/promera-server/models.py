"""Per-endpoint pydantic request models for promera-server."""

from __future__ import annotations

from typing import Literal

from bioagent_service import FailureKind, JobInfo, JobStatus  # noqa: F401
from pydantic import BaseModel, Field


class CofoldRequest(BaseModel):
    """Structure prediction (cofolding) request parameters."""

    num_seeds: int = Field(default=1, ge=1, le=10)
    diffusion_samples: int = Field(default=5, ge=1, le=25)
    diffusion_steps: int = Field(default=200, ge=10, le=1000)
    recycling_steps: int = Field(default=4, ge=1, le=20)
    save_trajectory: bool = Field(default=False)
    save_full_confidence: bool = Field(default=False)
    save_distogram: bool = Field(default=False)


class DesignRequest(BaseModel):
    """De novo binder design request parameters."""

    design_type: Literal["minibinder", "vhh"] = Field(default="minibinder")
    num_backbones: int = Field(default=10, ge=1, le=10000)
    diffusion_steps: int = Field(default=200, ge=10, le=1000)
    recycling_steps: int = Field(default=4, ge=1, le=20)
    binder_chain: str = Field(default="B", max_length=2)
    binder_length_min: int = Field(default=40, ge=10, le=300)
    binder_length_max: int = Field(default=120, ge=10, le=300)
    epitope_chain: str = Field(default="A", max_length=2)
    epitope_residues: str = Field(default="")
    target_chains: str = Field(default="")
    inverse_folder_type: Literal[
        "proteinmpnn", "solublempnn", "ligandmpnn", "none"
    ] = Field(default="solublempnn")
    inverse_folder_num_seqs: int = Field(default=1, ge=1, le=100)
    save_full_confidence: bool = Field(default=False)
    target_template_chain: str = Field(default="A", max_length=2)
    target_template_subsample_frac: float = Field(default=1.0, ge=0.0, le=1.0)

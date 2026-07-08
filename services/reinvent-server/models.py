"""Pydantic request models for reinvent-server (one per run mode).

Complex fields (scoring, stages, diversity_filter, inception, pairs,
learning_strategy) are dict/list — over multipart form they arrive as JSON
strings and are decoded by bioagent_service.forms.model_form_depends.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Generator = Literal["reinvent", "libinvent", "linkinvent", "mol2mol", "pepinvent"]
SampleStrategy = Literal["multinomial", "beamsearch"]


class SamplingRequest(BaseModel):
    generator: Generator = "reinvent"
    model_file: Optional[str] = Field(
        default=None,
        description="Prior: registry dot-key (.reinvent/.libinvent/...) or path "
                    "relative to prior_base. None → default for generator.",
    )
    num_smiles: int = Field(default=100, ge=1)
    unique_molecules: bool = True
    randomize_smiles: bool = True
    temperature: float = 1.0
    sample_strategy: SampleStrategy = "multinomial"
    device: Optional[str] = None


class ScoringRequest(BaseModel):
    smiles_column: str = "SMILES"
    standardize_smiles: bool = True
    parallel: int = Field(default=1, ge=1)
    scoring: dict = Field(..., description="[scoring] section (JSON): type + component list.")
    device: Optional[str] = None


class EnumerationRequest(BaseModel):
    batch_size: int = Field(default=10, ge=1)
    amino_acid_name_column: str = "Name"
    smiles_column: str = "Smiles"
    scoring: dict = Field(..., description="[scoring] section (JSON).")
    device: Optional[str] = None


class TransferLearningRequest(BaseModel):
    generator: Generator = "reinvent"
    input_model_file: Optional[str] = None
    output_model_name: str = "TL_model.model"
    num_epochs: int = Field(default=3, ge=1)
    save_every_n_epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=50, ge=1)
    num_refs: int = Field(default=100, ge=0)
    sample_batch_size: int = Field(default=100, ge=1)
    pairs: Optional[dict] = Field(
        default=None, description="Mol2Mol similarity pairing (JSON), → pairs.* in [parameters].",
    )
    device: Optional[str] = None


class StageSpec(BaseModel):
    chkpt_name: str
    termination: Literal["simple"] = "simple"
    max_score: float = 0.6
    min_steps: int = Field(default=25, ge=0)
    max_steps: int = Field(default=100, ge=1)
    scoring: dict = Field(..., description="[stage.scoring] section (JSON).")


def _default_learning_strategy() -> dict:
    return {"type": "dap", "sigma": 128, "rate": 0.0001}


class StagedLearningRequest(BaseModel):
    generator: Generator = "reinvent"
    prior_file: Optional[str] = None
    agent_file: Optional[str] = None
    batch_size: int = Field(default=64, ge=1)
    summary_csv_prefix: str = "staged_learning"
    use_checkpoint: bool = False
    purge_memories: bool = False
    randomize_smiles: bool = True
    learning_strategy: dict = Field(default_factory=_default_learning_strategy)
    diversity_filter: Optional[dict] = None
    inception: Optional[dict] = None
    stages: list[StageSpec] = Field(..., min_length=1)
    device: Optional[str] = None

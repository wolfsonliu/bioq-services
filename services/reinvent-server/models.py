"""Pydantic request models for reinvent-server (one per run mode).

Complex fields (scoring, stages, diversity_filter, inception, pairs,
learning_strategy) are dict/list — over multipart form they arrive as JSON
strings and are decoded by bioq_service.forms.model_form_depends.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from bioq_service import default_semantics

Generator = Literal["reinvent", "libinvent", "linkinvent", "mol2mol", "pepinvent"]
SampleStrategy = Literal["multinomial", "beamsearch"]

_URI_DESC = (
    "URI reference to the {what} (scheme in {{job, oss, file, http, https}}); "
    "alternative to the multipart upload. With oss_mount the gateway rewrites "
    "oss:// to a /mnt/oss path read straight off the mounted bucket."
)


class SamplingRequest(BaseModel):
    generator: Generator = "reinvent"
    model_file: Optional[str] = Field(
        default=None,
        description="Prior: registry dot-key (.reinvent/.libinvent/...) or path "
                    "relative to prior_base. None → default for generator.",
        json_schema_extra=default_semantics("auto", "use the tool's default model when omitted"),
    )
    num_smiles: int = Field(default=100, ge=1)
    unique_molecules: bool = True
    randomize_smiles: bool = True
    temperature: float = 1.0
    sample_strategy: SampleStrategy = "multinomial"
    device: Optional[str] = Field(default=None, json_schema_extra=default_semantics("auto", "auto-select CUDA if available"))
    smiles_file_uri: Optional[str] = Field(
        default=None,
        description=_URI_DESC.format(what="input SMILES file"),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )


class ScoringRequest(BaseModel):
    smiles_column: str = "SMILES"
    standardize_smiles: bool = True
    parallel: int = Field(default=1, ge=1)
    scoring: dict = Field(..., description="[scoring] section (JSON): type + component list.")
    device: Optional[str] = Field(default=None, json_schema_extra=default_semantics("auto", "auto-select CUDA if available"))
    smiles_file_uri: Optional[str] = Field(
        default=None,
        description=_URI_DESC.format(what="SMILES file to score"),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )


class EnumerationRequest(BaseModel):
    batch_size: int = Field(default=10, ge=1)
    amino_acid_name_column: str = "Name"
    smiles_column: str = "Smiles"
    scoring: dict = Field(..., description="[scoring] section (JSON).")
    device: Optional[str] = Field(default=None, json_schema_extra=default_semantics("auto", "auto-select CUDA if available"))
    peptide_smiles_uri: Optional[str] = Field(
        default=None,
        description=_URI_DESC.format(what="peptide SMILES file"),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    amino_acid_library_uri: Optional[str] = Field(
        default=None,
        description=_URI_DESC.format(what="amino-acid library file"),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )


class TransferLearningRequest(BaseModel):
    generator: Generator = "reinvent"
    input_model_file: Optional[str] = Field(default=None, json_schema_extra=default_semantics("auto", "use the tool's default model when omitted"))
    output_model_name: str = "TL_model.model"
    num_epochs: int = Field(default=3, ge=1)
    save_every_n_epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=50, ge=1)
    num_refs: int = Field(default=100, ge=0)
    # Upstream TL/validation.py::SectionParameters enforces ge=100; reject smaller
    # at the API boundary rather than failing deep in the run (FC async masks that
    # as "completed", leaving an empty-output job).
    sample_batch_size: int = Field(default=100, ge=100)
    pairs: Optional[dict] = Field(
        default=None,
        description="Mol2Mol similarity pairing (JSON), → pairs.* in [parameters].",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    device: Optional[str] = Field(default=None, json_schema_extra=default_semantics("auto", "auto-select CUDA if available"))
    smiles_file_uri: Optional[str] = Field(
        default=None,
        description=_URI_DESC.format(what="training SMILES file"),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    validation_smiles_file_uri: Optional[str] = Field(
        default=None,
        description=_URI_DESC.format(what="validation SMILES file"),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    input_model_uri: Optional[str] = Field(
        default=None,
        description=_URI_DESC.format(what="input model (.model) file"),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )


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
    prior_file: Optional[str] = Field(default=None, json_schema_extra=default_semantics("auto", "use the tool's default model when omitted"))
    agent_file: Optional[str] = Field(default=None, json_schema_extra=default_semantics("auto", "use the tool's default model when omitted"))
    batch_size: int = Field(default=64, ge=1)
    summary_csv_prefix: str = "staged_learning"
    use_checkpoint: bool = False
    purge_memories: bool = False
    randomize_smiles: bool = True
    learning_strategy: dict = Field(default_factory=_default_learning_strategy, json_schema_extra=default_semantics("auto", "use the tool's default when omitted"))
    diversity_filter: Optional[dict] = Field(default=None, json_schema_extra=default_semantics("unset", "only used when explicitly provided"))
    inception: Optional[dict] = Field(default=None, json_schema_extra=default_semantics("unset", "only used when explicitly provided"))
    stages: list[StageSpec] = Field(..., min_length=1)
    device: Optional[str] = Field(default=None, json_schema_extra=default_semantics("auto", "auto-select CUDA if available"))
    smiles_file_uri: Optional[str] = Field(
        default=None,
        description=_URI_DESC.format(what="input SMILES file"),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    prior_file_uri: Optional[str] = Field(
        default=None,
        description=_URI_DESC.format(what="prior (.model) file"),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    agent_file_uri: Optional[str] = Field(
        default=None,
        description=_URI_DESC.format(what="agent (.model) file"),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )

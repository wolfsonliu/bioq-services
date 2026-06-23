"""Folding ensemble Pydantic schemas — public API surface.

Input/output schemas shared across all folding methods.  Method-specific
options are nested in `method_options` (free-form dict, validated per-method
by each adapter's `method_options_schema`).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SequenceEntry(BaseModel):
    """One chain in the folding input."""

    id: str                              # chain ID, e.g. "A"
    sequence: str = Field(min_length=1)  # FASTA-style residues
    type: Literal["protein"] = "protein"


class FoldingInput(BaseModel):
    """Normalized input shared by all folding methods.

    Method-specific options (e.g. AlphaFold's db_preset) go in the API
    request's `method_options` field, validated per-method by each adapter.
    """

    sequences: list[SequenceEntry] = Field(min_length=1)
    msa_mode: Literal["auto", "empty", "search"] = "auto"


class StructureFile(BaseModel):
    """One predicted structure from a single method.

    URLs use a canonical, upstream-version-stable layout
    (``rank_<i>.<ext>``).  The raw upstream filename — which leaks
    implementation details like AlphaFold's ``ranked_<N>.pdb`` ranking or
    Boltz's ``input_model_<N>.cif`` — is preserved in
    ``original_filename`` for clients that need to cross-reference upstream
    logs/metadata.
    """

    rank: int                            # 0 = best for this method
    url: str                             # /v1/jobs/<task_id>/structures/<method>/rank_<i>.<ext>
    format: Literal["cif", "pdb"]
    plddt: Optional[float] = None
    size_bytes: Optional[int] = None
    original_filename: Optional[str] = None


class FoldingMethodResult(BaseModel):
    """One method's contribution to the ensemble result."""

    method: str                          # "alphafold" / "esmfold2" / "boltz"
    status: Literal["pending", "running", "completed", "failed"] = "completed"
    runtime_seconds: Optional[float] = None
    fc_job_id: Optional[str] = None
    error_summary: Optional[str] = None
    structures: list[StructureFile] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RankedStructure(BaseModel):
    """One entry in the cross-method ensemble ranking."""

    method: str
    rank: int                            # within-method rank
    overall_rank: int                    # across-methods rank
    score: float                         # by ensemble criterion (e.g. plddt)
    url: str


class FoldingOutput(BaseModel):
    """Full ensemble response body.

    Returned by GET /v1/jobs/<task_id> once `aggregated_output` is populated.
    """

    task_id: str
    status: Literal["pending", "running", "completed", "partial", "failed"]
    input: FoldingInput
    results: list[FoldingMethodResult]
    ensemble_ranking: list[RankedStructure] = Field(default_factory=list)
    ensemble_score: Optional[float] = None

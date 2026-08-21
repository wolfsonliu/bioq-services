"""Per-endpoint pydantic request models for esmfold2-server.

Single endpoint: `/api/fold` — structure prediction for protein/DNA/RNA/ligand
complexes using ESMFold2.

The `sequences` field mirrors ESMFold2's StructurePredictionInput: each entry
carries a `type` discriminator (protein/dna/rna/ligand) and type-specific
fields. `tools.build_input_json` renders them into the JSON that `inference.py`
consumes.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401
from bioq_service import default_semantics
from pydantic import BaseModel, Field, model_validator

# ---- Sequence entry types ----

SequenceType = Literal["protein", "dna", "rna", "ligand"]


class Modification(BaseModel):
    position: int = Field(ge=0, description="Residue index, 0-based.")
    ccd: str = Field(description="CCD code of the modified residue.")


class SequenceEntry(BaseModel):
    """One entry in the `sequences` list.

    Polymer chains (protein/dna/rna) require `sequence`. Ligand chains require
    exactly one of `smiles` or `ccd`.
    """

    type: SequenceType
    id: Union[str, list[str]]
    sequence: Optional[str] = None
    smiles: Optional[str] = None
    ccd: Optional[list[str]] = None
    modifications: list[Modification] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistent(self) -> "SequenceEntry":
        if self.type == "ligand":
            if bool(self.smiles) == bool(self.ccd):
                raise ValueError(
                    f"ligand entry id={self.id!r} requires exactly one of `smiles` or `ccd`"
                )
            if self.sequence is not None:
                raise ValueError(
                    f"ligand entry id={self.id!r} must not have `sequence`"
                )
        else:
            if not self.sequence:
                raise ValueError(
                    f"{self.type} entry id={self.id!r} requires `sequence`"
                )
            if self.smiles or self.ccd:
                raise ValueError(
                    f"{self.type} entry id={self.id!r} cannot have `smiles` or `ccd`"
                )
        return self


# ---- Request model ----


class FoldRequest(BaseModel):
    """Request body for `/api/fold`."""

    sequences: list[SequenceEntry] = Field(min_length=1)

    num_loops: int = Field(default=3, ge=1, le=20)
    num_sampling_steps: int = Field(default=50, ge=1, le=1000)
    num_diffusion_samples: int = Field(default=1, ge=1, le=50)
    seed: Optional[int] = Field(
        default=None,
        json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"),
    )
    noise_scale: Optional[float] = Field(
        default=None, ge=0.0, le=10.0,
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    step_scale: Optional[float] = Field(
        default=None, ge=0.0, le=10.0,
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )


__all__ = [
    "FailureKind",
    "FoldRequest",
    "JobInfo",
    "JobStatus",
    "Modification",
    "SequenceEntry",
    "SequenceType",
]

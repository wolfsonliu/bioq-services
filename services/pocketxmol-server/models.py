"""Per-endpoint pydantic request models for pocketxmol-server.

Six endpoints share a common backbone (num_samples / batch_size / seed /
pocket-args) but each has task-specific fields.  Field docs mirror
engineering/decisions/2026-07-06-pocketxmol-server-design.md §Request Schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

# Re-export framework JobInfo / JobStatus for backwards compatibility with
# callers that used to import them from server.models.
from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401
from bioq_service import default_semantics
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------
class NoiseMode(str, Enum):
    """/api/dock — controls the two settings weights (`free` vs `flexible`)."""

    gaussian = "gaussian"
    flexible = "flexible"


class SbddMode(str, Enum):
    """/api/sbdd — one-shot (`simple`) vs auto-regressive (`ar`, refine loop)."""

    ar = "ar"
    simple = "simple"


class Part1Pert(str, Enum):
    """/api/linking — how to treat the fixed fragment (part1)."""

    fixed = "fixed"
    free = "free"
    small = "small"


class PocketCriterion(str, Enum):
    center_of_mass = "center_of_mass"
    min = "min"


class PepDesignMode(str, Enum):
    """/api/pepdesign — 4 sub-modes."""

    denovo_linear = "denovo_linear"
    denovo_cyclic = "denovo_cyclic"
    inverse_fold = "inverse_fold"
    sc_pack = "sc_pack"


class ConfidenceVariant(str, Enum):
    tuned_cfd = "tuned_cfd"
    flex_cfd = "flex_cfd"


# ---------------------------------------------------------------------------
# 1. DockRequest
# ---------------------------------------------------------------------------
class DockRequest(BaseModel):
    num_samples: int = Field(default=10, ge=1, le=200,
                             description="Number of docked poses to generate.")
    batch_size: int = Field(default=50, ge=1, le=200,
                            description="GPU batch size; reduce on OOM.")
    is_pep: bool = Field(default=False,
                         description="True for peptide docking (peptide PDB "
                         "input or `pep_sequence`).")
    noise_mode: NoiseMode = Field(default=NoiseMode.gaussian)
    pocket_radius: float = Field(default=10.0, ge=5.0, le=25.0)
    pocket_criterion: PocketCriterion = Field(default=PocketCriterion.center_of_mass)
    pocket_coord: Optional[list[float]] = Field(
        default=None,
        description="Explicit pocket center [x, y, z]. "
        "Mutually exclusive with providing `ref_ligand` as pocket source.",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    smiles: Optional[str] = Field(
        default=None,
        description="SMILES string of the small-molecule ligand "
        "(alternative to uploading `ligand` file).",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    pep_sequence: Optional[str] = Field(
        default=None, min_length=3, max_length=30,
        description="Peptide sequence for docking from sequence (only when "
        "is_pep=true); equivalent to upstream `pepseq_<seq>` shortcut.",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    seed: Optional[int] = Field(default=None, ge=0, json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"))

    @field_validator("pocket_coord")
    @classmethod
    def _check_coord_shape(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None and len(v) != 3:
            raise ValueError("pocket_coord must be a list of 3 floats [x, y, z].")
        return v


# ---------------------------------------------------------------------------
# 2. SbddRequest
# ---------------------------------------------------------------------------
class SbddRequest(BaseModel):
    num_samples: int = Field(default=50, ge=1, le=200)
    batch_size: int = Field(default=50, ge=1, le=200)
    mode: SbddMode = Field(default=SbddMode.ar,
                           description="`ar` = auto-regressive refinement "
                           "(better quality); `simple` = one-shot (fast).")
    pocket_coord: list[float] = Field(
        ...,
        description="Explicit pocket center [x, y, z].  Required — de novo "
        "SBDD has no reference ligand to derive it from.",
    )
    pocket_radius: float = Field(default=15.0, ge=5.0, le=25.0)
    mol_size_mean: int = Field(default=28, ge=10, le=60)
    mol_size_std: int = Field(default=2, ge=1, le=10)
    seed: Optional[int] = Field(default=None, ge=0, json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"))

    @field_validator("pocket_coord")
    @classmethod
    def _check_coord_shape(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("pocket_coord must be a list of 3 floats [x, y, z].")
        return v


# ---------------------------------------------------------------------------
# 3. LinkingRequest
# ---------------------------------------------------------------------------
class LinkingRequest(BaseModel):
    num_samples: int = Field(default=50, ge=1, le=200)
    batch_size: int = Field(default=50, ge=1, le=200)
    fragments: list[list[int]] = Field(
        ...,
        description="Fragment atom index groups (0-based) from `input_ligand`. "
        "1 group = growing, 2+ groups = linking / PROTAC.",
    )
    part1_pert: Part1Pert = Field(default=Part1Pert.fixed,
                                  description="Fragment perturbation: "
                                  "`fixed`=lock poses; `free`=allow movement; "
                                  "`small`=slight perturbation (opt_partial).")
    mol_size_mean: int = Field(default=40, ge=10, le=100,
                               description="Target atom count of the linked mol.")
    mol_size_std: int = Field(default=3, ge=1, le=10)
    use_input_center: bool = Field(default=True,
                                   description="Use input ligand centroid as "
                                   "denoising space center; False → pocket centroid.")
    pocket_radius: float = Field(default=10.0, ge=5.0, le=25.0)
    seed: Optional[int] = Field(default=None, ge=0, json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"))

    @field_validator("fragments")
    @classmethod
    def _check_fragments(cls, v: list[list[int]]) -> list[list[int]]:
        if len(v) < 1 or len(v) > 5:
            raise ValueError("Provide 1 to 5 fragment groups.")
        seen: set[int] = set()
        for group in v:
            if not group:
                raise ValueError("Fragment groups cannot be empty.")
            for idx in group:
                if idx < 0:
                    raise ValueError("Fragment atom indices must be non-negative.")
                if idx in seen:
                    raise ValueError(
                        f"Atom index {idx} appears in more than one fragment group."
                    )
                seen.add(idx)
        return v


# ---------------------------------------------------------------------------
# 4. OptimizeRequest
# ---------------------------------------------------------------------------
class OptimizeRequest(BaseModel):
    num_samples: int = Field(default=50, ge=1, le=200)
    batch_size: int = Field(default=50, ge=1, le=200)
    init_step: float = Field(default=0.5, ge=0.05, le=0.99,
                             description="Initial noise fraction; smaller = "
                             "closer to input, larger = more exploration.")
    num_steps: int = Field(default=50, ge=10, le=200)
    mol_size_mean: int = Field(default=38, ge=10, le=100)
    mol_size_std: int = Field(default=3, ge=1, le=10)
    pocket_radius: float = Field(default=10.0, ge=5.0, le=25.0)
    seed: Optional[int] = Field(default=None, ge=0, json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"))


# ---------------------------------------------------------------------------
# 5. PepDesignRequest
# ---------------------------------------------------------------------------
class PepDesignRequest(BaseModel):
    mode: PepDesignMode = Field(default=PepDesignMode.denovo_linear)
    pep_length: Optional[int] = Field(
        default=None, ge=5, le=30,
        description="Peptide residue count (required for de novo modes; "
        "ignored for inverse_fold / sc_pack — length taken from input PDB).",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    num_samples: int = Field(default=10, ge=1, le=50)
    batch_size: int = Field(default=50, ge=1, le=200)
    pocket_coord: Optional[list[float]] = Field(default=None, json_schema_extra=default_semantics("unset", "only used when explicitly provided"))
    pocket_radius: float = Field(default=20.0, ge=10.0, le=30.0,
                                 description="Peptide pockets are larger than "
                                 "small-molecule pockets; default = 20 Å.")
    fix_pos_res_bb: list[int] = Field(default_factory=list, json_schema_extra=default_semantics("unset", "empty when omitted"))
    fix_pos_res_sc: list[int] = Field(default_factory=list, json_schema_extra=default_semantics("unset", "empty when omitted"))
    fix_type_res_bb: list[int] = Field(default_factory=list, json_schema_extra=default_semantics("unset", "empty when omitted"))
    fix_type_res_sc: list[int] = Field(default_factory=list, json_schema_extra=default_semantics("unset", "empty when omitted"))
    seed: Optional[int] = Field(default=None, ge=0, json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"))

    @model_validator(mode="after")
    def _check_length_for_denovo(self) -> "PepDesignRequest":
        if self.mode in (PepDesignMode.denovo_linear, PepDesignMode.denovo_cyclic):
            if self.pep_length is None:
                raise ValueError(
                    f"pep_length is required for mode={self.mode.value}."
                )
        return self

    @field_validator("pocket_coord")
    @classmethod
    def _check_coord_shape(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None and len(v) != 3:
            raise ValueError("pocket_coord must be a list of 3 floats [x, y, z].")
        return v


# ---------------------------------------------------------------------------
# 6. ConfidenceRequest
# ---------------------------------------------------------------------------
class ConfidenceRequest(BaseModel):
    source_job_id: str = Field(
        ...,
        min_length=1,
        description="job_id of a previously completed generation job.  "
        "Its output/<exp_name>_<timestamp>_SDF/ dir is what the tuned "
        "ranker consumes.",
    )
    variant: ConfidenceVariant = Field(default=ConfidenceVariant.tuned_cfd)
    batch_size: int = Field(default=50, ge=1, le=200)


__all__ = [
    "NoiseMode",
    "SbddMode",
    "Part1Pert",
    "PocketCriterion",
    "PepDesignMode",
    "ConfidenceVariant",
    "DockRequest",
    "SbddRequest",
    "LinkingRequest",
    "OptimizeRequest",
    "PepDesignRequest",
    "ConfidenceRequest",
    "JobInfo",
    "JobStatus",
    "FailureKind",
]

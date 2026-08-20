"""Per-endpoint pydantic request models for dockq-server.

The two endpoints share a common set of DockQ CLI flags (`mapping`,
`small_molecule`, `capri_peptide`, `no_align`, `allowed_mismatches`,
`optDockQF1`, `n_cpu`). `_DockQCommon` captures them.
"""

from __future__ import annotations

from typing import Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exports)
from bioq_service import default_semantics
from pydantic import BaseModel, Field

__all__ = [
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "ScoreBatchRequest",
    "ScoreRequest",
]


class _DockQCommon(BaseModel):
    """Fields shared across /api/score and /api/score_batch."""

    name: str = Field(
        default="run",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$",
        description=(
            "Output basename; appears in result paths (output/<name>.json for the "
            "single endpoint, output/scores.csv key column for batch). Restricted "
            "to [A-Za-z0-9_.-] (no slashes/spaces)."
        ),
    )
    mapping: Optional[str] = Field(
        default=None,
        description=(
            "DockQ --mapping flag, format MODELCHAINS:NATIVECHAINS. "
            "Use `*` as a wildcard; e.g. ':HL' restricts the native interfaces to H-L."
        ),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    small_molecule: bool = Field(
        default=False,
        description="Pass --small_molecule. Required when scoring PDB/CIF inputs that contain HEM / cofactor / ligand chains.",
    )
    capri_peptide: bool = Field(
        default=False,
        description="Pass --capri_peptide. Use only for peptide-protein scoring; DockQ explicitly warns the score is unreliable in this mode.",
    )
    no_align: bool = Field(
        default=False,
        description="Pass --no_align. Skips sequence alignment; trusts residue numbering directly. Faster, but only safe when model + native share the exact residue indexing.",
    )
    allowed_mismatches: int = Field(
        default=0, ge=0, le=100,
        description="Number of allowed mismatches when mapping model→native sequences (--allowed_mismatches).",
    )
    optDockQF1: bool = Field(
        default=False,
        description="Pass --optDockQF1. Optimizes the chain mapping for DockQ_F1 instead of DockQ.",
    )
    n_cpu: Optional[int] = Field(
        default=None, ge=1, le=64,
        description="Override DOCKQ_DEFAULT_N_CPU for this call. Forwarded to DockQ's --n_cpu.",
        json_schema_extra=default_semantics("auto", "use all available cores"),
    )


class ScoreRequest(_DockQCommon):
    """`POST /api/score` — score a single (model, native) pair."""


class ScoreBatchRequest(_DockQCommon):
    """`POST /api/score_batch` — score N candidate models against 1 reference native.

    The N model files are uploaded as multipart fields named `models`
    (`-F models=@m1.pdb -F models=@m2.pdb ...`). The native is uploaded as
    `native` (one file). Per-model DockQ JSONs and a sorted `scores.csv`
    summary land under `output/`.
    """

    sort_by: str = Field(
        default="DockQ",
        pattern=r"^[A-Za-z_]+$",
        description=(
            "Column name in scores.csv to sort the summary by (descending). "
            "Common choices: DockQ, DockQ_F1, fnat, iRMSD (ascending preferred for RMSDs)."
        ),
    )

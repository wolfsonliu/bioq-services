"""Per-endpoint pydantic request models for ppiflow-server.

Five generation modes correspond to five endpoints, each backed by one of
PPIFlow's `sample_*.py` scripts. Field defaults mirror the upstream CLI
defaults so an agent's "minimal call" matches what a PPIFlow user would type
on the command line.

The framework's `JobInfo` / `JobStatus` / `FailureKind` are re-exported here so
clients can keep importing `server.models.JobInfo` unchanged.
"""

from __future__ import annotations

from typing import Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exports)
from pydantic import BaseModel, Field

__all__ = [
    "AntibodyRequest",
    "BinderRequest",
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "MonomerRequest",
    "NanobodyRequest",
    "ScaffoldingRequest",
]


class _CommonRequest(BaseModel):
    """Fields shared across all PPIFlow endpoints."""

    name: str = Field(
        default="run",
        description=(
            "Identifier used as a subfolder under the output directory "
            "(`output/<name>/`). Restricted to [A-Za-z0-9_.-] (no slashes / spaces) "
            "to prevent path injection — `name='..'` would write outside the job dir."
        ),
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$",
    )
    samples_per_target: int = Field(
        default=5,
        ge=1,
        le=10000,
        description="How many samples to generate per (target, length) combination.",
    )


class BinderRequest(_CommonRequest):
    """`POST /api/sample/binder` — design a binder against an uploaded target PDB.

    Wraps `sample_binder.py`. The `target` PDB upload is required; binder
    chain is freshly designed.
    """

    target_chain: str = Field(..., min_length=1, description="Chain ID of the target in target PDB.")
    binder_chain: str = Field(default="A", min_length=1, description="Chain ID assigned to generated binder.")
    specified_hotspots: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated hotspot residues on the target, e.g. 'B119,B141,B200'. "
            "Optional: if absent PPIFlow samples hotspots via "
            "`sample_hotspot_rate_{min,max}`."
        ),
    )
    sample_hotspot_rate_min: float = Field(default=0.05, ge=0.0, le=1.0)
    sample_hotspot_rate_max: float = Field(default=0.20, ge=0.0, le=1.0)
    samples_min_length: int = Field(default=75, ge=20, le=400)
    samples_max_length: int = Field(default=120, ge=20, le=400)


class _AntibodyLike(_CommonRequest):
    """Shared fields for `/api/sample/antibody` and `/api/sample/nanobody`."""

    antigen_chain: str = Field(..., min_length=1, description="Chain ID of antigen.")
    heavy_chain: str = Field(default="A", min_length=1)
    specified_hotspots: Optional[str] = Field(
        default=None,
        description="Hotspot residues on the antigen, e.g. 'C56,C58'.",
    )
    cdr_length: str = Field(
        default="CDRH1,5-12,CDRH2,4-17,CDRH3,5-26",
        description=(
            "CDR length spec, comma-separated `CDR<name>,min-max,...`. "
            "Antibody adds CDRL1/CDRL2/CDRL3; nanobody uses heavy-only by default."
        ),
    )


class AntibodyRequest(_AntibodyLike):
    """`POST /api/sample/antibody` — antibody CDR design (uses antibody.ckpt).

    Framework PDB must include heavy AND light chains, with CDR loops removed
    and IMGT numbering applied (per upstream README).
    """

    light_chain: str = Field(default="B", min_length=1, description="Chain ID of light chain in framework.")
    # Override default to add light-chain CDRs.
    cdr_length: str = Field(
        default="CDRH1,5-12,CDRH2,4-17,CDRH3,5-26,CDRL1,5-12,CDRL2,3-10,CDRL3,4-13",
        description="CDR length spec including all six CDRs.",
    )


class NanobodyRequest(_AntibodyLike):
    """`POST /api/sample/nanobody` — VHH CDR design (uses nanobody.ckpt). Heavy-only."""


class MonomerRequest(_CommonRequest):
    """`POST /api/sample/monomer` — unconditional monomer generation.

    Wraps `sample_monomer.py` (uncond mode). `length_subset` controls which
    lengths to draw from.
    """

    length_subset: list[int] = Field(
        default_factory=lambda: [80, 100, 120],
        description="List of target lengths. PPIFlow samples `samples_per_target` per length.",
    )


class ScaffoldingRequest(_CommonRequest):
    """`POST /api/sample/scaffolding` — motif scaffolding (uses monomer.ckpt).

    Requires a `motif_csv` upload + `motif_pdbs` zip with the structures referenced
    by the CSV's `motif_path` column. `motif_names` filters which CSV rows to run.
    """

    motif_names: list[str] = Field(
        ...,
        min_length=1,
        description="Subset of target names from the CSV to scaffold, e.g. ['01_1LDB'].",
    )

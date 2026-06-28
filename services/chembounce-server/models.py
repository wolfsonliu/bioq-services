"""Per-endpoint pydantic request models for chembounce-server."""

from __future__ import annotations

from typing import Literal, Optional

from bioagent_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exported)
from pydantic import BaseModel, Field


DatabaseChoice = Literal["250mw", "full"]


class ScaffoldHopRequest(BaseModel):
    """Inputs to ChemBounce scaffold hopping pipeline.

    Mirrors upstream `chembounce.py` CLI flags but enums / clamps the things
    that should be enums and exposes the threshold bag as Optional fields.
    `database` selects which fingerprint .npz to subprocess against.
    """

    # ---- Required ----
    input_smiles: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Input SMILES string of the target molecule to scaffold-hop.",
    )

    # ---- Database selection ----
    database: DatabaseChoice = Field(
        default="250mw",
        description=(
            "Which scaffold fingerprint DB to search:\n"
            "- '250mw': scaffolds with MW ≤ 250 (~tens of MB, fast, low recall)\n"
            "- 'full' : ~4M scaffolds (~GB, paper-grade, needs ≥64 GB RAM)"
        ),
    )

    # ---- Search control ----
    core_smiles: Optional[str] = Field(
        default=None,
        max_length=500,
        description="A core substructure SMILES that must be preserved in the "
        "final candidate; if not actually contained in `input_smiles`, "
        "ChemBounce raises an error.",
    )
    frag_max_n: int = Field(
        default=100, ge=1, le=10000,
        description="Maximum candidates per fragment.",
    )
    overall_max_n: Optional[int] = Field(
        default=None, ge=1,
        description="Hard cap on total candidates across all fragments.",
    )
    scaffold_top_n: Optional[int] = Field(
        default=None, ge=1,
        description="Number of scaffolds to test per fragment.",
    )
    cand_max_n__rplc: int = Field(
        default=10, ge=1, le=1000,
        description="Maximum candidates per replaced scaffold.",
    )
    tanimoto_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Tanimoto similarity threshold between original and "
        "generated SMILES.",
    )

    # ---- Property thresholds ----
    qed_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    qed_max: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sa_min: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    sa_max: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    logp_min: Optional[float] = None
    logp_max: Optional[float] = None
    mw_min: Optional[float] = Field(default=None, ge=0.0)
    mw_max: Optional[float] = Field(default=None, ge=0.0)
    h_donor_min: Optional[int] = Field(default=None, ge=0)
    h_donor_max: Optional[int] = Field(default=None, ge=0)
    h_acceptor_min: Optional[int] = Field(default=None, ge=0)
    h_acceptor_max: Optional[int] = Field(default=None, ge=0)

    wo_lipinski: bool = Field(
        default=False,
        description="Turn off Lipinski's rule of five.  Enable for "
        "macrocycles / peptides that naturally violate it.",
    )

    # ---- Runtime ----
    low_mem: bool = Field(default=False, description="Upstream's `-l` flag.")


__all__ = [
    "DatabaseChoice",
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "ScaffoldHopRequest",
]

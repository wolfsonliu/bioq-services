"""Request model for lightdock-server.

A single `/api/dock` endpoint runs the full LightDock GSO docking protocol.
File inputs (`receptor` / `ligand` / `restraints`) are handled as separate Form
fields in app.py, not on this model.
"""

from __future__ import annotations

from typing import Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exports)
from pydantic import BaseModel, Field

__all__ = ["FailureKind", "JobInfo", "JobStatus", "DockRequest"]


class DockRequest(BaseModel):
    """`POST /api/dock` — full LightDock docking protocol.

    Runs setup → GSO run → conformation generation → per-swarm clustering →
    ranking → top-N generation, producing ranked docked complexes.

    NOTE on runtime: the sampling cost scales with swarms x glowworms x steps.
    The upstream production preset (400 swarms, 200 glowworms, 100 steps) takes
    hours and is not suited to an interactive FC call. For interactive use set a
    small `swarms` (e.g. 10-40) and supply `restraints` to focus sampling.
    """

    swarms: int = Field(
        default=0,
        ge=0,
        le=2000,
        description=(
            "Number of GSO swarms. 0 = auto-estimate from the receptor surface "
            "(can be hundreds — use with care on FC). Set a small value for "
            "interactive runs."
        ),
    )
    glowworms: int = Field(
        default=200,
        ge=1,
        le=500,
        description="Number of glowworms (candidate poses) per swarm.",
    )
    steps: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of GSO optimization steps.",
    )
    scoring_function: str = Field(
        default="fastdfire",
        pattern=r"^[a-z0-9]+$",
        max_length=32,
        description=(
            "LightDock scoring function (e.g. fastdfire, dfire, cpydock, pisa, "
            "vdw, dna for protein-DNA). Validated against the installed set; see "
            "/healthz/detail for the available list."
        ),
    )
    cores: Optional[int] = Field(
        default=None,
        ge=1,
        le=64,
        description="Multiprocessing cores for the GSO run. None → service default.",
    )
    top: int = Field(
        default=10,
        ge=1,
        le=200,
        description="Number of top-ranked docked complexes to generate (top/top_N.pdb).",
    )
    swarm_seed: int = Field(
        default=324324,
        description="Random seed for starting-position (swarm) calculation.",
    )
    gso_seed: int = Field(
        default=324324,
        description="Random seed for the GSO algorithm.",
    )
    use_anm: bool = Field(
        default=False,
        description="Enable ANM backbone flexibility during setup + conformation generation.",
    )
    cluster_cutoff: float = Field(
        default=4.0,
        gt=0.0,
        le=50.0,
        description="Backbone-RMSD cutoff (A) for per-swarm BSAS clustering.",
    )
    noxt: bool = Field(default=False, description="Remove OXT atoms during setup.")
    noh: bool = Field(default=False, description="Remove hydrogen atoms during setup.")
    now: bool = Field(default=False, description="Remove water (H2O) during setup.")

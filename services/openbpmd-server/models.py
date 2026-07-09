"""Pydantic request models for openbpmd-server."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    """Binding Pose Metadynamics stability scoring.

    File inputs (structure + parameters) are handled separately as multipart
    uploads / URIs — see app.py.  This model carries only the scalar params.
    """

    lig_resname: str = Field(
        default="MOL",
        min_length=1,
        max_length=4,
        pattern=r"^[A-Za-z0-9]+$",
        description="Ligand residue name in the topology (upstream argparse "
        "default is 'MOL'; some systems use 'LIG'/'UNK' — match your file).",
    )

    nreps: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of independent metadynamics repeats (run in series). "
        "10 is the standard BPMD protocol.",
    )

    hill_height: float = Field(
        default=0.3,
        ge=0.05,
        le=2.0,
        description="Metadynamical hill height in kcal/mol. 0.3 is standard.",
    )

    system_format: Optional[Literal["amber", "gromacs"]] = Field(
        default=None,
        description="Force input format. None = auto-detect by extension "
        "(.gro -> gromacs, else amber).",
    )

    # ---- Advanced / testing knobs (default None -> upstream standard) ----
    # Changing these breaks BPMD score comparability with published values —
    # they exist so integration tests can run a short trajectory. Leave unset
    # for real scoring. See design doc §6.6.
    sim_ns: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=50.0,
        description="ADVANCED/TESTING: production metadynamics length in ns. "
        "None -> upstream standard 10 ns. Non-standard values break score "
        "comparability.",
    )

    equil_steps: Optional[int] = Field(
        default=None,
        ge=100,
        le=5_000_000,
        description="ADVANCED/TESTING: NVT equilibration steps (2 fs each). "
        "None -> upstream standard 250000 (500 ps).",
    )

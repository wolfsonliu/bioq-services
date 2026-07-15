"""Pydantic request models for haddock3-server.

File inputs (molecule PDBs, restraint .tbl, .actpass) are handled separately as
multipart uploads / URIs in app.py — these models carry only scalar params.
"""

from __future__ import annotations

from typing import Optional

from bioagent_service import JobInfo  # noqa: F401  (re-export for compat)
from pydantic import BaseModel, Field


class DockRequest(BaseModel):
    """General workflow runner (`haddock3 <config>`).

    The workflow body is supplied out-of-band as the `config` form field (HTTP)
    or a `--config` file (CLI). It MUST NOT set `run_dir`, `molecules`, `mode`
    or `ncores` — those are injected/overridden by the service. Reference
    uploaded molecules and .tbl files by their bare filename.
    """

    run_name: str = Field(
        default="run",
        pattern=r"^[A-Za-z0-9._-]+$",
        max_length=64,
        description="Name of the run subfolder created under output/.",
    )
    ncores: Optional[int] = Field(
        default=None, ge=1, le=128,
        description="CPU cores (mode='local'). None -> service default.",
    )


class ProteinProteinRequest(BaseModel):
    """Curated two-body protein-protein docking with sane defaults."""

    sampling: int = Field(
        default=200, ge=1, le=10000,
        description="Rigid-body models to generate ([rigidbody] sampling). "
        "1000 is the production default; lower is faster/cheaper.",
    )
    do_flexref: bool = Field(
        default=True, description="Run semi-flexible refinement ([flexref]).",
    )
    do_emref: bool = Field(
        default=True, description="Run final energy-minimisation refinement ([emref]).",
    )
    clustering: bool = Field(
        default=True,
        description="Cluster models by fraction of common contacts ([clustfcc]).",
    )
    top_models: int = Field(
        default=4, ge=1, le=1000,
        description="Models kept per cluster after selection ([seletopclusts]).",
    )
    ncores: Optional[int] = Field(default=None, ge=1, le=128)


class ScoreRequest(BaseModel):
    """Standalone HADDOCK scoring of a complex (`haddock3-score`)."""

    full: bool = Field(
        default=True,
        description="Emit the per-component energy breakdown (vdw/elec/desolv/air/bsa).",
    )
    params: Optional[dict[str, str]] = Field(
        default=None,
        description="Extra emscoring params forwarded as `-p key value` "
        "(e.g. {'nemsteps': '50', 'w_air': '1'}). Booleans as 'True'/'False'.",
    )


class RestrainBodiesRequest(BaseModel):
    """CNS-free: generate body-restraint .tbl from a multi-chain PDB."""

    exclude: Optional[str] = Field(
        default=None,
        description="Comma-separated chain IDs to exclude from the calculation.",
    )


class ActpassToAmbigRequest(BaseModel):
    """CNS-free: convert two active/passive residue files into an ambig .tbl."""

    segid1: str = Field(
        default="A", max_length=4, pattern=r"^[A-Za-z0-9]+$",
        description="Segment/chain id for the first molecule.",
    )
    segid2: str = Field(
        default="B", max_length=4, pattern=r"^[A-Za-z0-9]+$",
        description="Segment/chain id for the second molecule.",
    )

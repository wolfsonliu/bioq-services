"""Per-endpoint pydantic request models for qligfep-server.

See engineering/decisions/2026-07-06-qligfep-server-design.md §4.
"""
from __future__ import annotations

from typing import Literal

from bioq_service import JobInfo, JobStatus, FailureKind  # noqa: F401
from pydantic import BaseModel, Field

Forcefield = Literal["OPLS2005", "OPLS2015", "OPLSAAM", "AMBER14sb", "CHARMM36"]
Cluster = Literal["LOCAL", "CSB", "SLURM"]


class LigprepRequest(BaseModel):
    ligand_name: str = Field(..., description="Output file prefix, e.g. '17' → 17.lib/17.prm/17.pdb")
    forcefield: Literal["openff-2.1.0", "openff-2.0.0"] = "openff-2.1.0"
    net_charge: int | None = Field(default=None, description="Override rdkit-inferred formal charge.")


class ProtprepRequest(BaseModel):
    sphere_radius: float = Field(default=22.0, gt=0)
    sphere_center: str = Field(..., description="'cx:cy:cz' or 'resname:resnum'")
    forcefield: Forcefield = "OPLSAAM"
    mutchain: str | None = None
    nowater: bool = False
    noclean: bool = False
    preplocation: Cluster = "LOCAL"


class CogRequest(BaseModel):
    mode: Literal["all", "atomrange"] = "all"
    atom_range: str | None = Field(default=None, description="e.g. '1:25'; required when mode=atomrange")


class SetupLigfepRequest(BaseModel):
    lig1_name: str = Field(...)
    lig2_name: str = Field(...)
    forcefield: Forcefield = "OPLSAAM"
    system: Literal["water", "protein"] = "protein"
    start: Literal["0.5", "1.0"] = "0.5"
    temperature: float = 298.15
    replicates: int = Field(default=10, ge=1)
    windows: int = Field(default=51, ge=3)
    sampling: Literal["linear", "exponential", "reverse_exp"] = "linear"
    timestep: Literal["1fs", "2fs"] = "2fs"
    cluster: str = "LOCAL"
    to_clean: bool = True


class SetupResfepRequest(BaseModel):
    mutation: str = Field(..., description="e.g. 'A24V'")
    mutchain: str = Field(...)
    system: Literal["water", "protein"] = "protein"
    dual: bool = False
    shell_rest: float = 25.0
    tripeptide: bool = False
    cofactors: list[str] | None = None
    forcefield: Forcefield = "OPLSAAM"
    windows: int = 51
    sampling: Literal["linear", "exponential", "reverse_exp"] = "linear"
    timestep: Literal["1fs", "2fs"] = "2fs"
    temperature: float = 298.15
    replicates: int = 10
    start: Literal["0.5", "1.0"] = "0.5"
    cluster: str = "LOCAL"


class SetupLieRequest(BaseModel):
    ligand_name: str = Field(...)
    forcefield: Forcefield = "OPLSAAM"
    system: Literal["water", "protein"] = "protein"
    cofactor: list[str] | None = None
    radius: float = 22.0
    time_ns: float = 5.0
    temperature: float = 298.15
    replicates: int = 10
    cluster: str = "LOCAL"
    preplocation: Cluster = "LOCAL"


class RunFepRequest(BaseModel):
    window_idx: int = Field(..., ge=0)
    leg: Literal["protein", "water"]
    replicate_idx: int = Field(default=0, ge=0)
    device: Literal["cpu", "mpi", "gpu"] = "mpi"
    nprocs: int = Field(default=1, ge=1)
    stage: Literal["eq", "md", "both"] = "both"
    keep_dcd: bool = True


class AnalyzeFepRequest(BaseModel):
    temperature: float = 298.15
    start: Literal["0.5", "1.0"] = "0.5"
    end_state_catastrophe: float = 1000.0
    use_pdb: bool = False


class AnalyzeLieRequest(BaseModel):
    radius: float = 22.0
    cofactor: list[str] | None = None

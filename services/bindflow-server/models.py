"""Per-endpoint pydantic request models for bindflow-server.

Two calculation types share `BaseCalculateRequest`:

* `FepCalculateRequest`   → `/api/calculate/fep`   + CLI `python -m server fep ...`
* `MmpbsaCalculateRequest`→ `/api/calculate/mmpbsa`+ CLI `python -m server mmpbsa ...`

File inputs (protein / ligands / cofactor / membrane / custom-FF / topology
zip) are NOT part of these pydantic models — they arrive as multipart form
uploads or resolved URIs and are handled in `app.py` / `__main__.py`.
"""

from __future__ import annotations

from typing import Literal, Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exported)
from pydantic import BaseModel, Field


WaterModel = Literal[
    "amber/tip3p",
    "amber/tip4p",
    "amber/tip4pew",
    "amber/spce",
    "charmm/tip3p",
    "charmm/tip4p",
    "opc",
    "opc3",
]
"""Water force fields supported by the vendored `data/gmx_water_models/`.
The upstream list is broader; we surface only widely-used ones + charmm
variants so pydantic validation is a real gate.  Extend as needed.
"""

SolvBoxType = Literal["dodecahedron", "octahedron", "cubic", "triclinic"]

SchedulerChoice = Literal["frontend", "slurm"]

LigandFFType = Literal["openff", "gaff"]
"""Small-molecule force-field families we bundle in the image.

Espaloma is intentionally NOT included in v0.0.1 (it pulls tensorflow + dgl,
adding 3-6 GB).  Design doc §11.3 records this decision.
"""


class BaseCalculateRequest(BaseModel):
    """Shared inputs between FEP and MMPBSA endpoints.

    Field naming mirrors the kwargs of `bindflow.runners.calculate(...)` so
    the wrapper can pass most fields through unchanged.  Fields we do rename
    (e.g. multi-line `global_config`) get a comment explaining the mapping.
    """

    # ---- Force fields ----
    water_model: WaterModel = Field(
        default="amber/tip3p",
        description="Water force field.  Must match one of the vendored gmx_water_models entries.",
    )
    ligand_ff_type: LigandFFType = Field(
        default="openff",
        description="Small-molecule force-field family.  espaloma is NOT bundled in v0.0.1.",
    )
    ligand_ff_code: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Force-field code (e.g. `openff_unconstrained-2.0.0.offxml`).  None → upstream default for the type.",
    )
    protein_ff_code: str = Field(
        default="amber99sb-ildn",
        max_length=64,
        description="GROMACS protein force-field code.",
    )

    # ---- System / solvation ----
    hmr_factor: Optional[float] = Field(
        default=2.5, ge=0.5, le=4.0,
        description="Hydrogen Mass Repartitioning scaling.  Set null / None to disable.",
    )
    dt_max: float = Field(
        default=0.004, ge=0.001, le=0.005,
        description="Maximum MD integration step (ps).  Constrained by hmr_factor.",
    )
    solv_d: float = Field(
        default=1.5, ge=0.5, le=5.0,
        description="Solvent box edge distance (`gmx editconf -d`, nm).",
    )
    solv_bt: SolvBoxType = Field(default="dodecahedron")
    solv_rmin: float = Field(
        default=1.0, ge=0.1, le=3.0,
        description="Minimum ion-ion distance (`gmx genion -rmin`).",
    )
    solv_ion_conc: float = Field(
        default=0.15, ge=0.0, le=2.0,
        description="Electrolyte concentration (mol/L).  0.0 adds only neutralizing counterions.",
    )

    # ---- Host / restraint ----
    host_name: str = Field(default="Protein", min_length=1, max_length=32)
    host_selection: str = Field(
        default="protein and name CA",
        description="MDAnalysis selection used for Boresch restraint detection.",
    )
    fix_protein: bool = Field(
        default=True,
        description="Run pdbfixer + `gmx editconf -ignh` on the protein input.",
    )

    # ---- Cofactor placement ----
    cofactor_on_protein: bool = Field(
        default=True,
        description="If a cofactor is provided, whether it joins the protein thermostat group (True) or the solvent (False).",
    )
    cofactor_selection: str = Field(
        default="resname COF",
        description="GMX selection for cofactor when top/gro is user-provided.",
    )

    # ---- Concurrency / replicas ----
    threads: int = Field(
        default=12, ge=1, le=64,
        description="CPU threads per snakemake rule (`gmx mdrun -nt` effectively).",
    )
    num_jobs: int = Field(
        default=10, ge=1, le=100000,
        description="Max concurrent snakemake jobs.  FrontEnd: cap at (n_cpu / threads).",
    )
    replicas: int = Field(
        default=3, ge=1, le=10,
        description="Number of independent MD replicas per ligand.",
    )
    job_prefix: Optional[str] = Field(
        default=None, max_length=64,
        description="Slurm queue naming prefix.",
    )

    # ---- Scheduling ----
    scheduler: SchedulerChoice = Field(
        default="frontend",
        description=(
            "'frontend' runs snakemake in-container (default; self-contained). "
            "'slurm' invokes sbatch — requires apptainer --bind of slurm sockets."
        ),
    )
    submit: bool = Field(
        default=True,
        description="If False, only build snakefile + job.sh without executing (dry-run).",
    )

    # ---- Escape hatch: raw YAML for BindFlow global_config ----
    global_config_yaml: Optional[str] = Field(
        default=None,
        max_length=131072,  # 128 KiB soft cap
        description=(
            "Raw YAML text merged into BindFlow's global_config.  Fields set "
            "via pydantic override same keys in the YAML.  Use for cluster/mdrun/mdp/extra_directives blocks.  "
            "SECURITY: contents can inject shell via extra_directives.dependencies "
            "— trusted-client only (see design doc §11.5)."
        ),
    )


class FepCalculateRequest(BaseCalculateRequest):
    """FEP-specific overrides.

    Lambda-schedule window counts.  Upstream defaults are documented in
    `bindflow.orchestration.flow_builder.update_nwindows_config`:
      ligand.vdw=11, ligand.coul=11
      complex.vdw=21, complex.coul=11, complex.bonded=11
    """

    nwindows_ligand_vdw: int = Field(default=11, ge=3, le=41)
    nwindows_ligand_coul: int = Field(default=11, ge=3, le=41)
    nwindows_complex_vdw: int = Field(default=21, ge=3, le=41)
    nwindows_complex_coul: int = Field(default=11, ge=3, le=41)
    nwindows_complex_bonded: int = Field(default=11, ge=3, le=41)


class MmpbsaCalculateRequest(BaseCalculateRequest):
    """MMPBSA-specific overrides."""

    samples: int = Field(
        default=20, ge=1, le=200,
        description="Number of MD frames sampled for MM(P/G)BSA analysis per replica.",
    )
    mmpbsa_yaml: Optional[str] = Field(
        default=None, max_length=32768,
        description="Raw YAML overriding BindFlow's global_config['mmpbsa'] block.",
    )


__all__ = [
    "BaseCalculateRequest",
    "FailureKind",
    "FepCalculateRequest",
    "JobInfo",
    "JobStatus",
    "LigandFFType",
    "MmpbsaCalculateRequest",
    "SchedulerChoice",
    "SolvBoxType",
    "WaterModel",
]

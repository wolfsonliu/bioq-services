"""Per-endpoint pydantic request models.  Re-export framework's JobInfo."""

from __future__ import annotations

from typing import Literal, Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401
from pydantic import BaseModel, Field, field_validator, model_validator


class MolFilterParams(BaseModel):
    """Filter applied to generated molecules (upstream MolFilter kwargs).

    None fields are omitted from the YAML config so upstream `Hparams.get()`
    falls back to its own defaults.
    """

    ring_sizes: Optional[list[int]] = Field(
        default=None, description="Only allow rings of these sizes."
    )
    ring_system_max: Optional[int] = Field(
        default=None, description="Max size of any ring system.", ge=1
    )
    ring_loops_max: Optional[int] = Field(
        default=None, description="Max number of ring loops.", ge=0
    )
    dbl_bond_pairs: Optional[bool] = Field(
        default=None,
        description="Allow consecutive double bonds (False = disallow).",
    )
    n_atoms_min: Optional[int] = Field(
        default=None, description="Minimum number of atoms per molecule.", ge=0
    )


def _broadcast_to_4(v):
    """Accept a scalar or list; scalar → [v]*4 (matches upstream Hparams behavior)."""
    if isinstance(v, (int, float)):
        return [float(v)] * 4
    return v


class GenerateRequest(BaseModel):
    """De novo ligand generation (/api/generate) — MolGenerator mode.

    ``zbetas`` / ``temps`` are always stored as ``list[float]`` of length 4;
    a scalar form-post value is broadcast to a length-4 list on the fly.
    """

    pdb_id: str = Field(default="target", min_length=1, max_length=20)
    n_samples: int = Field(default=10, ge=1, le=5000)

    zbetas: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0],
        description="Prior↔posterior interpolation per latent resolution.  "
        "List of length 4 or scalar (broadcast to length 4).",
    )
    temps: list[float] = Field(
        default_factory=lambda: [0.5, 0.5, 0.5, 0.5],
        description="Sampling temperature per latent resolution.  "
        "List of length 4 or scalar (broadcast to length 4).",
    )

    random_rotate: bool = Field(default=True)
    random_translate: bool = Field(default=False)

    ffopt_mols: bool = Field(
        default=True,
        description="RDKit FF post-processing on generated molecules.",
    )

    mol_filter: MolFilterParams = Field(default_factory=MolFilterParams)

    _broadcast_zbetas = field_validator("zbetas", mode="before")(_broadcast_to_4)
    _broadcast_temps = field_validator("temps", mode="before")(_broadcast_to_4)

    @model_validator(mode="after")
    def _check_zbetas_temps(self) -> "GenerateRequest":
        for name in ("zbetas", "temps"):
            v = getattr(self, name)
            if len(v) != 4:
                raise ValueError(
                    f"{name} must be a scalar or a list of length 4 "
                    f"(got length {len(v)})"
                )
        return self


class GenerateSpatialRequest(GenerateRequest):
    """Substructure modification / scaffold hopping (/api/generate_spatial)."""

    substruct_modify_pattern: Optional[str] = Field(
        default=None,
        description="SMILES / SMARTS for the substructure to preserve.  "
        "Provide EITHER this field OR upload `substruct_modify` file "
        "(and matching `substruct_modify_uri`), never both.",
    )

    # spatial defaults lean toward posterior (upstream generate_spatial.yml).
    zbetas: list[float] = Field(default_factory=lambda: [0.3, 0.3, 0.3, 0.3])
    temps: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])


OptKey = Literal["affinity_qvina", "qed", "alogp", "sa"]


class OptimizeRequest(BaseModel):
    """Multi-cycle affinity / property optimization (/api/optimize).

    Uses QVina2 for docking when `key_opt=affinity_qvina`.  Long-running:
    the default 8 cycles × 1000 initial × 20 children can take 4-8 h.
    """

    pdb_id: str = Field(default="target", min_length=1, max_length=20)

    key_opt: OptKey = Field(
        default="affinity_qvina",
        description="Optimization target: affinity_qvina | qed | alogp | sa",
    )
    opt_increase: bool = Field(
        default=False,
        description="True = maximize; False = minimize (default matches "
        "affinity_qvina where lower kcal/mol is better).",
    )
    save_name: str = Field(default="optimization_run", max_length=100)

    n_cycles: int = Field(default=8, ge=1, le=20)
    n_samples_initial: int = Field(default=1000, ge=10, le=10000)
    n_samples: int = Field(
        default=20, ge=1, le=200,
        description="Children per parent per cycle.",
    )
    n_best_parents: int = Field(default=20, ge=1, le=200)
    affinity_quantile_thresh: float = Field(default=0.5, ge=0.0, le=1.0)

    cluster_parents: bool = Field(default=True)
    random_rotate: bool = Field(default=True)
    random_translate: bool = Field(default=False)

    zbetas_initial: float = Field(default=0.3, ge=0.0, le=1.0)
    temps_initial: float = Field(default=1.0, gt=0.0)
    # zbetas: list of length == n_cycles (or scalar, broadcast on the fly).
    # Default is a length-1 sentinel that gets broadcast to n_cycles below.
    # Paper's canonical 8-cycle schedule is [0.3, 0.3, 0.2×5, 0.1] — pass
    # explicitly when running the default n_cycles=8.
    zbetas: list[float] = Field(
        default_factory=lambda: [0.3],
        description="Per-cycle zbetas — list of length n_cycles, or scalar "
        "(broadcast to length n_cycles).",
    )
    temps: list[float] = Field(
        default_factory=lambda: [1.0],
        description="Per-cycle temperature — list of length n_cycles, or "
        "scalar (broadcast to length n_cycles).",
    )

    protonate: bool = Field(default=True, description="obabel protonation before QVina.")

    mol_filter: MolFilterParams = Field(default_factory=MolFilterParams)

    @field_validator("zbetas", "temps", mode="before")
    @classmethod
    def _broadcast_scalar(cls, v):
        # Broadcast to a placeholder length; the after-validator enforces
        # the real n_cycles match once both fields are set.
        if isinstance(v, (int, float)):
            return [float(v)]
        return v

    @model_validator(mode="after")
    def _check_zbetas_length_matches_cycles(self) -> "OptimizeRequest":
        # Scalar was broadcast to a length-1 list in the field validator;
        # expand it now to match n_cycles.  If the user supplied a real list,
        # its length must equal n_cycles.
        for name in ("zbetas", "temps"):
            v = getattr(self, name)
            if len(v) == 1:
                setattr(self, name, v * self.n_cycles)
            elif len(v) != self.n_cycles:
                raise ValueError(
                    f"{name} list length {len(v)} must equal "
                    f"n_cycles={self.n_cycles}"
                )
        return self

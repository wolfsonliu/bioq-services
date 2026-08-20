"""Per-endpoint pydantic request models.

RFdiffusion is a single Hydra-driven script (`scripts/run_inference.py`) that
covers many generation modes. We expose five structured endpoints that map onto
the modes documented in the upstream README, plus a freeform escape hatch:

  * `/api/generate/unconditional` — monomer / unconditional + optional macrocycle
  * `/api/generate/motif`         — motif scaffolding around a PDB
  * `/api/generate/binder`        — PPI binder design with hotspots
  * `/api/generate/symmetry`      — C/D/tetrahedral symmetric oligomers
  * `/api/generate`               — raw contig + extra Hydra overrides (advanced)

Each model carries only the fields a *typical* user of that mode needs. Power
users can fall through to `/api/generate` and set anything in the base.yaml.
"""

from __future__ import annotations

from typing import Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # re-exports
from bioq_service import default_semantics
from pydantic import BaseModel, Field

__all__ = [
    "BinderRequest",
    "CustomRequest",
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "MotifRequest",
    "SymmetryRequest",
    "UnconditionalRequest",
]


# Known checkpoint friendly-name → filename mapping. Used by `tools.py` to turn
# `model="complex_base"` into `inference.ckpt_override_path=<models_dir>/Complex_base_ckpt.pt`.
# Kept here so the OpenAPI schema can surface the choices in the field description.
MODEL_CHOICES: dict[str, str] = {
    "base": "Base_ckpt.pt",
    "epoch8": "Base_epoch8_ckpt.pt",
    "complex_base": "Complex_base_ckpt.pt",
    "complex_beta": "Complex_beta_ckpt.pt",
    "complex_fold": "Complex_Fold_base_ckpt.pt",
    "inpaint_seq": "InpaintSeq_ckpt.pt",
    "inpaint_seq_fold": "InpaintSeq_Fold_ckpt.pt",
    "active_site": "ActiveSite_ckpt.pt",
}

_MODEL_DESC = (
    "Override checkpoint. Leave unset to let run_inference.py auto-select based on "
    "the inputs (recommended). Choices: " + ", ".join(sorted(MODEL_CHOICES))
)


class _GenerationCommon(BaseModel):
    """Fields every generation endpoint accepts."""

    num_designs: int = Field(default=10, ge=1, le=10000, description="Number of designs to sample.")
    diffuser_t: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Reverse-diffusion timesteps. 50 is the upstream default; lowering speeds up but hurts quality.",
    )
    final_step: int = Field(
        default=1,
        ge=1,
        description="Stop the trajectory at step N (default 1 = run to completion).",
    )
    write_trajectory: bool = Field(
        default=False,
        description="Write the full reverse-diffusion trajectory under output/traj/. Default off to save disk.",
    )
    deterministic: bool = Field(
        default=False, description="Fix RNG so reruns reproduce exactly."
    )
    noise_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description=(
            "Sets both denoiser.noise_scale_ca and noise_scale_frame. "
            "Lower (e.g. 0.5 or 0.0) → higher quality, less diversity. Binder mode defaults to 0.0."
        ),
    )
    model: Optional[str] = Field(default=None, description=_MODEL_DESC, json_schema_extra=default_semantics("auto", "auto-select by the tool from the request inputs"))


class UnconditionalRequest(_GenerationCommon):
    """Params for `POST /api/generate/unconditional`. No input PDB needed."""

    min_length: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Lower bound (residues) for each design. Length sampled per trajectory.",
    )
    max_length: int = Field(default=100, ge=10, le=1000, description="Upper bound (residues).")
    cyclic: bool = Field(
        default=False,
        description="RFpeptides macrocycle mode (inference.cyclic=True, cyc_chains='a').",
    )


class MotifRequest(_GenerationCommon):
    """Params for `POST /api/generate/motif`. Input PDB carries the motif residues."""

    contigs: str = Field(
        description=(
            "Hydra contig string, e.g. `10-40/A163-181/10-40`. Letters reference chains/residues "
            "in the input PDB; bare ranges (`10-40`) are residues to be designed."
        ),
        examples=["10-40/A163-181/10-40"],
    )
    length: Optional[str] = Field(
        default=None,
        description="Total length constraint, e.g. `55-55`. Useful when contigs span a wide range.",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    inpaint_seq: Optional[str] = Field(
        default=None,
        description="Residues whose sequence should be masked (auto-switches model to InpaintSeq).",
        examples=["A1/A30-40"],
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )


class BinderRequest(_GenerationCommon):
    """Params for `POST /api/generate/binder`. Input PDB is the target; contig defines the binder."""

    contigs: str = Field(
        description=(
            "Contig string with the target chain + binder length range, e.g. "
            "`A1-150/0 70-100` (target A residues 1-150, chainbreak, 70-100 residue binder)."
        ),
        examples=["A1-150/0 70-100", "B1-100/0 100-100"],
    )
    hotspots: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated hotspot residues on the target the binder must contact. "
            "Format `<chain><resnum>`, e.g. `A59,A83,A91`. 3-6 recommended."
        ),
        examples=["A59,A83,A91"],
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    # Binder mode normally runs with zero noise (per the upstream README).
    noise_scale: float = Field(default=0.0, ge=0.0, le=2.0)


class SymmetryRequest(_GenerationCommon):
    """Params for `POST /api/generate/symmetry`. Uses `--config-name symmetry`."""

    symmetry: str = Field(
        description="Symmetry group: `c<N>` (cyclic), `d<N>` (dihedral), or `tetrahedral`.",
        examples=["c6", "d2", "tetrahedral"],
    )
    total_length: int = Field(
        ge=20, le=2000,
        description="Total length of the oligomer. Must be divisible by the number of chains.",
    )
    # Optional auxiliary potential — passed straight through to Hydra
    guiding_potentials: Optional[str] = Field(
        default=None,
        description=(
            "Raw Hydra list-string for `potentials.guiding_potentials`, "
            'e.g. `["type:olig_contacts,weight_intra:1,weight_inter:0.1"]`.'
        ),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    guide_scale: Optional[float] = Field(
        default=None,
        description="Strength of the guiding potential (default 10).",
        json_schema_extra=default_semantics("auto", "use the tool's default when omitted"),
    )
    guide_decay: Optional[str] = Field(
        default=None,
        description="`constant` / `linear` / `quadratic` / `cubic`.",
        json_schema_extra=default_semantics("auto", "use the tool's default when omitted"),
    )
    olig_inter_all: bool = Field(default=False, description="`potentials.olig_inter_all=True` if set.")
    olig_intra_all: bool = Field(default=False, description="`potentials.olig_intra_all=True` if set.")


class CustomRequest(_GenerationCommon):
    """Params for `POST /api/generate` — raw contig + freeform Hydra overrides."""

    contigs: str = Field(
        description="Raw contig string, passed as `contigmap.contigs=[<value>]`.",
        examples=["100-100", "10-40/A163-181/10-40", "A1-150/0 70-100"],
    )
    config_name: str = Field(
        default="base",
        description="Hydra `--config-name`. Use `symmetry` for symmetric inputs; otherwise `base`.",
    )
    extra_overrides: Optional[str] = Field(
        default=None,
        description=(
            "JSON object mapping dotted Hydra paths → values, e.g. "
            '`{"diffuser.partial_T": 10, "ppi.hotspot_res": "[A59,A83,A91]"}`. '
            "Each key/value is appended to argv as `key=value`."
        ),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )

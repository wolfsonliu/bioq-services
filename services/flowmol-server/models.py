"""Per-endpoint pydantic request models for flowmol-server."""

from __future__ import annotations

from typing import Literal, Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401  (re-exported)
from bioq_service import default_semantics
from pydantic import BaseModel, Field


# 22 pretrained variants published in upstream `flowmol/trained_models/readme.md`
# (GEOM-Drugs training; QM9-trained variants are not in v3.1 and are out of
# scope for v0.0.1 — see design doc §"v0.0.1 不在范围").
ModelVariant = Literal[
    # Main model + primary self-correction ablations
    "flowmol3",           # paper SOTA; default
    "fm3_nodistort",      # no geometry distortion
    "fm3_none",           # no self-correction features at all
    # Loss-weight ablations (a/c/e/x = atom-type / charge / bond / position)
    "fm3_ahigh", "fm3_alow",
    "fm3_chigh", "fm3_clow",
    "fm3_ehigh", "fm3_elow",
    "fm3_xhigh", "fm3_xlow",
    # Geometry distortion parameter ablations
    "fm3_distort_extreme",
    "fm3_distort_highp", "fm3_distort_hight",
    "fm3_distort_lowp", "fm3_distort_lowt",
    # Fake-atom ablations
    "fm3_fa_highp", "fm3_fa_highstd",
    "fm3_fa_lowp", "fm3_fa_lowstd",
    # Self-conditioning proportion ablations
    "fm3_scprop_high", "fm3_scprop_low",
]


class GenerateRequest(BaseModel):
    """Unconditional 3D small-molecule generation with FlowMol3.

    No file inputs — pure parameterised sampling from the flow-matching prior.
    Upstream reference: `test.py` CLI (in-process wrapped by our
    `server/inference.py`).
    """

    n_mols: int = Field(
        default=100, ge=1, le=1000,
        description="Number of molecules to generate. Batched sampling; "
        "runtime scales linearly in ceil(n_mols / max_batch_size) forward passes.",
    )

    n_timesteps: int = Field(
        default=250, ge=50, le=500,
        description="Number of Euler integration steps. 250 is the paper "
        "sweet spot; 100 already usable at slightly lower PB-Valid rate.",
    )

    n_atoms_per_mol: Optional[int] = Field(
        default=None, ge=5, le=100,
        description="If set, every molecule has exactly this many atoms. "
        "Default (null) samples atom counts from the training distribution "
        "per molecule — recommended.",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )

    model_variant: ModelVariant = Field(
        default="flowmol3",
        description=(
            "Which trained model to use. See flowmol3 wiki for the "
            "3 primary + 19 ablation variants. `flowmol3` is the paper "
            "SOTA (default). Only pre-staged variants on NAS are usable — "
            "check /healthz/detail.staged_variants."
        ),
    )

    seed: Optional[int] = Field(
        default=None,
        description="pytorch-lightning `seed_everything`; null → framework "
        "fills a random seed and records it in JobInfo.input_params.",
        json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"),
    )

    stochasticity: Optional[float] = Field(
        default=None, ge=0.0,
        description="CTMC sampling stochasticity η_t. Applies to `fm3_*` "
        "CTMC-parameterised variants (all of them). Null → upstream config "
        "default.",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )

    hc_thresh: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="High-confidence threshold for CTMC purity sampling. "
        "Null → upstream config default.",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )

    max_batch_size: int = Field(
        default=128, ge=1, le=512,
        description="Sampling batch size cap. n_mols > max_batch_size splits "
        "into ceil(n_mols / max_batch_size) forward passes.",
    )


__all__ = [
    "FailureKind",
    "GenerateRequest",
    "JobInfo",
    "JobStatus",
    "ModelVariant",
]

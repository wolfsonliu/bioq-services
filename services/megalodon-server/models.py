"""Per-endpoint pydantic request models + model registry for megalodon-server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401  (re-exported)
from pydantic import BaseModel, Field

# Megalodon ships 6 headline checkpoints: {qm9, drugs} x {diffusion, fm, quick}.
# `model_name` selects a (dataset, config, checkpoint) bundle — see
# MODEL_REGISTRY below and settings.MegalodonSettings.list_models().
ModelName = Literal[
    "drugs_diffusion",
    "drugs_fm",
    "drugs_quick",
    "qm9_diffusion",
    "qm9_fm",
    "qm9_quick",
]


@dataclass(frozen=True)
class ModelSpec:
    """Static wiring for one Megalodon variant.

    config_rel is relative to the vendored conf dir
    (`/opt/megalodon/scripts/conf`); ckpt_file lives under
    `<weights_dir>/ckpts/<dataset>/`; statistics under
    `<weights_dir>/stats/<dataset>/`.
    """

    dataset: str  # "qm9" | "drugs"
    objective: str  # "diffusion" | "fm" | "quick"
    config_rel: str
    ckpt_file: str


# NOTE: the diffusion checkpoint filename differs per dataset
# (qm9=megalodon_diffusion, drugs=megalodon_large_diffusion); fm/quick share
# names. The drugs flow-matching variant uses the *_inference config.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "drugs_diffusion": ModelSpec(
        "drugs", "diffusion", "drugs/megalodon_diffusion.yaml",
        "megalodon_large_diffusion.ckpt"),
    "drugs_fm": ModelSpec(
        "drugs", "fm", "drugs/megalodon_fm_inference.yaml",
        "megalodon_fm.ckpt"),
    "drugs_quick": ModelSpec(
        "drugs", "quick", "drugs/megalodon_quick_diffusion.yaml",
        "megalodon_small_diffusion.ckpt"),
    "qm9_diffusion": ModelSpec(
        "qm9", "diffusion", "qm9/megalodon_diffusion.yaml",
        "megalodon_diffusion.ckpt"),
    "qm9_fm": ModelSpec(
        "qm9", "fm", "qm9/megalodon_fm.yaml",
        "megalodon_fm.ckpt"),
    "qm9_quick": ModelSpec(
        "qm9", "quick", "qm9/megalodon_quick_diffusion.yaml",
        "megalodon_small_diffusion.ckpt"),
}


class GenerateRequest(BaseModel):
    """Unconditional 3D small-molecule generation with Megalodon.

    No file inputs — pure parameterised sampling from the diffusion / flow
    prior. Upstream reference: `scripts/sample.py`, wrapped in-process by our
    `server/inference.py` (which runs its own sampling loop so `num_atoms`
    can be fixed, then reuses the upstream metric components).
    """

    model_name: ModelName = Field(
        default="drugs_diffusion",
        description="Which pre-staged variant to use: {qm9,drugs}_"
        "{diffusion,fm,quick}. `drugs_*` are drug-like (GEOM-Drugs); `qm9_*` "
        "are small molecules. Only variants present on NAS are usable — check "
        "/api/models or /healthz/detail.",
    )

    n_molecules: int = Field(
        default=100, ge=1, le=10000,
        description="Number of molecules to generate. Batched internally; "
        "runtime scales with ceil(n_molecules / batch_size) sampling passes.",
    )

    n_atoms_per_mol: Optional[int] = Field(
        default=None, ge=5, le=125,
        description="If set, every molecule has exactly this many atoms. "
        "Default (null) samples atom counts from the training size "
        "distribution (train_n_h.pickle) — recommended. Model tested up to "
        "125 atoms.",
    )

    timesteps: int = Field(
        default=500, ge=10, le=1000,
        description="Number of sampling steps. Diffusion default 500; flow-"
        "matching variants work well at ~100. Fewer = faster, lower quality.",
    )

    batch_size: int = Field(
        default=100, ge=1, le=512,
        description="Internal sampling batch size. n_molecules > batch_size "
        "is split into multiple passes automatically.",
    )

    seed: Optional[int] = Field(
        default=None,
        description="Random seed (torch.manual_seed). null → unseeded; "
        "recorded in JobInfo.input_params.",
    )


__all__ = [
    "FailureKind",
    "GenerateRequest",
    "JobInfo",
    "JobStatus",
    "ModelName",
    "ModelSpec",
    "MODEL_REGISTRY",
]

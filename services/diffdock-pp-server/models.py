"""Per-endpoint pydantic request models for diffdock-pp-server."""

from __future__ import annotations

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401  (re-exported)
from bioq_service import default_semantics
from pydantic import BaseModel, Field


class DockRequest(BaseModel):
    """Rigid protein-protein docking via score + confidence diffusion.

    Terminology: `receptor` / `ligand` are both proteins (EquiDock convention;
    the ligand is the shape that gets rotated/translated, the receptor sits
    at the coordinate anchor). Not to be confused with small-molecule ligand.
    """

    num_samples: int = Field(
        default=40, ge=1, le=200,
        description="Reverse-diffusion samples drawn by the score model "
        "(paper/upstream default 40). Runtime scales roughly linearly.",
    )

    actual_steps: int = Field(
        default=40, ge=10, le=40,
        description="Denoising steps actually executed. Upstream uses "
        "num_steps=40 with an early-stop at actual_steps because the last "
        "few steps tend to overfit; usually keep this at 40.",
    )

    top_k: int = Field(
        default=5, ge=1, le=40,
        description="How many top-ranked poses to write out as PDB "
        "(`dock_pose_<rank>.pdb`). The full N samples are always preserved "
        "in `raw_samples.pkl` regardless of top_k.",
    )

    use_confidence_model: bool = Field(
        default=True,
        description="If True, run the confidence model to rank samples and "
        "emit dock_pose_<rank>.pdb by descending confidence. If False, skip "
        "the confidence pass (about 30 percent faster) and rank samples by "
        "their draw order — you lose the quality signal but save GPU time.",
    )

    seed: int | None = Field(
        default=None,
        description="Random seed for reverse diffusion. Leave None for a "
        "fresh sample each call; set explicitly for reproducibility. "
        "Framework echoes the resolved seed in JobInfo.input_params.",
        json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"),
    )

    mirror_ligand: bool = Field(
        default=False,
        description="Upstream option: mirror half of the samples (swap "
        "receptor/ligand roles) for extra diversity. Default off — enable "
        "only if you have a specific need for chirality-doubled ensembles.",
    )

    no_final_noise: bool = Field(
        default=True,
        description="Upstream default: skip the noise term on the final "
        "denoising step. Stabilizes output; usually leave True.",
    )


__all__ = [
    "DockRequest",
    "FailureKind",
    "JobInfo",
    "JobStatus",
]

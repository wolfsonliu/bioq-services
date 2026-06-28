"""Per-endpoint pydantic request models for diffusion-hopping-server."""

from __future__ import annotations

from typing import Literal

from bioagent_service import FailureKind, JobInfo, JobStatus  # noqa: F401  (re-exported)
from pydantic import BaseModel, Field


ModelVariant = Literal[
    "gvp_conditional",      # DiffHopp (paper main); conditioned on ref ligand functional group
    "gvp_unconditional",    # DiffHopp inpainting; no ref-ligand functional-group conditioning
    "egnn_conditional",     # DiffHopp-EGNN (EGNN variant, conditional)
    "egnn_unconditional",   # DiffHopp-EGNN inpainting
]


class GenerateRequest(BaseModel):
    """Scaffold-hopping generation given a protein pocket + reference ligand.

    Upstream entry point: `generate_scaffolds.py`.  Our wrapper
    `inference.py` re-exposes this with explicit `--variant` / `--checkpoint`
    flags instead of upstream's hardcoded `gvp_conditional` path.
    """

    num_samples: int = Field(
        default=10, ge=1, le=100,
        description="Number of scaffold candidates to generate (1–100). "
        "Sampling is a single batched forward pass — runtime scales sub-linearly.",
    )

    model_variant: ModelVariant = Field(
        default="gvp_conditional",
        description=(
            "Which trained model to use:\n"
            "- gvp_conditional   : DiffHopp (recommended); ligand-conditioned\n"
            "- gvp_unconditional : DiffHopp inpainting (no functional-group conditioning)\n"
            "- egnn_conditional  : DiffHopp-EGNN (EGNN backbone, conditional)\n"
            "- egnn_unconditional: DiffHopp-EGNN inpainting"
        ),
    )


__all__ = [
    "FailureKind",
    "GenerateRequest",
    "JobInfo",
    "JobStatus",
    "ModelVariant",
]

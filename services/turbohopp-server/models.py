"""Per-endpoint pydantic request models for turbohopp-server."""

from __future__ import annotations

from typing import Optional

from bioagent_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exported)
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Scaffold-hopping generation given a protein pocket + reference ligand.

    Upstream (``evaluate_consistency.py``) only supports dataset-mode
    inference over ``pdbbind_filtered`` / ``crossdocked``.  Our wrapper
    ``inference.py`` re-exposes the consistency-model sampler with an
    explicit single (pocket, reference_ligand) input path.
    """

    num_samples: int = Field(
        default=10, ge=1, le=100,
        description=(
            "Number of scaffold candidates to generate.  Sampling is a single "
            "batched forward pass — runtime scales sub-linearly."
        ),
    )
    num_sampling_steps: int = Field(
        default=40, ge=1, le=100,
        description=(
            "Consistency-model sampling steps.  Paper default 40 (find_best "
            "upper bound); 5-10 works well in practice.  Trades off quality "
            "for latency; consistency-model output stays coherent at very "
            "low step counts, unlike raw diffusion."
        ),
    )
    find_best: bool = Field(
        default=False,
        description=(
            "Post-hoc rescoring: take the last num_sampling_steps candidates "
            "and pick the highest QED + normalized-SA-score composite.  "
            "Doubles effective sample count."
        ),
    )
    seed: Optional[int] = Field(
        default=None, ge=0,
        description="Sampling RNG seed.  None → torch default (non-deterministic).",
    )


__all__ = [
    "FailureKind",
    "GenerateRequest",
    "JobInfo",
    "JobStatus",
]

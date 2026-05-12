"""Per-endpoint pydantic request models.

The framework's `JobInfo` / `JobStatus` / `FailureKind` are re-exported here so
existing clients can keep importing `server.models.JobInfo` unchanged.

The generation endpoints fall into three structured shapes (unconditional /
motif / binder) plus a freeform `/api/generate` that takes raw YAML — the
custom one is intentionally schema-less because genie3's full experiment YAML
is large and evolves with upstream releases.
"""

from __future__ import annotations

from typing import Optional

from bioagent_service import FailureKind, JobInfo, JobStatus  # re-exports
from pydantic import BaseModel, Field

__all__ = [
    "BinderRequest",
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "MotifRequest",
    "UnconditionalRequest",
]


class _GenerationCommon(BaseModel):
    """Fields shared by all three structured generation endpoints."""

    n_sample: int = Field(default=4, ge=1, le=10000)
    batch_size: int = Field(default=1, ge=1)
    num_devices: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override genie3's GPU auto-detect. Leave unset to use all visible devices.",
    )


class UnconditionalRequest(_GenerationCommon):
    """Params for `POST /api/generate/unconditional` — no dataset needed."""

    min_length: int = Field(default=100, ge=20, le=800)
    max_length: int = Field(default=100, ge=20, le=800)
    length_step: int = Field(default=50, ge=1)
    direction_scale: float = Field(
        default=0.8,
        description="Quality–diversity trade-off. Recommended 0.8 for length ≤ 300, 0.0 for longer.",
    )


class MotifRequest(_GenerationCommon):
    """Params for `POST /api/generate/motif`. A `dataset` zip is uploaded alongside."""

    selections: Optional[str] = Field(
        default=None,
        description="Comma-separated problem names from the dataset. Default: all problems.",
    )
    direction_scale: float = Field(default=0.1)


class BinderRequest(_GenerationCommon):
    """Params for `POST /api/generate/binder`. A `dataset` zip is uploaded alongside."""

    selections: Optional[str] = Field(
        default=None,
        description="Comma-separated problem names from the dataset. Default: all problems.",
    )
    direction_scale: float = Field(default=0.0)

"""Pydantic models for genie3-server API request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskKind(str, Enum):
    UNCONDITIONAL = "unconditional"
    MOTIF = "motif"
    BINDER = "binder"
    CUSTOM = "custom"


# --- Generation requests ---


class UnconditionalRequest(BaseModel):
    """Unconditional protein backbone generation."""

    min_length: int = Field(default=100, ge=20, le=800)
    max_length: int = Field(default=100, ge=20, le=800)
    length_step: int = Field(default=50, ge=1)
    n_sample: int = Field(default=4, ge=1, le=10000)
    direction_scale: float = Field(
        default=0.8,
        description="Quality–diversity trade-off. 0.8 for length<=300, 0.0 for length>300.",
    )
    batch_size: int = Field(default=1, ge=1)


class MotifRequest(BaseModel):
    """Motif scaffolding generation. Requires a problem-set zip upload."""

    selections: Optional[str] = Field(
        default=None,
        description="Comma-separated problem names. If omitted, all problems are used.",
    )
    n_sample: int = Field(default=4, ge=1, le=10000)
    direction_scale: float = Field(default=0.1)
    batch_size: int = Field(default=1, ge=1)


class BinderRequest(BaseModel):
    """Binder design generation. Requires a problem-set zip upload."""

    selections: Optional[str] = Field(
        default=None,
        description="Comma-separated problem names. If omitted, all problems are used.",
    )
    n_sample: int = Field(default=4, ge=1, le=10000)
    direction_scale: float = Field(default=0.0)
    batch_size: int = Field(default=1, ge=1)


# --- Job ---


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    task: Optional[TaskKind] = None
    message: Optional[str] = None
    progress: Optional[str] = None

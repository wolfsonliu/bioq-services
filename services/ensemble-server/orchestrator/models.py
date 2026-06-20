"""Internal models for ensemble job state.

Public API schemas (FoldingInput, FoldingOutput) live in their respective
TaskKind directories.  These types are persisted to NAS sidecar `job.json`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SubTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CACHED = "cached"


class SubTaskRecord(BaseModel):
    """One method invocation as part of an ensemble job."""

    method: str
    sub_task_id: str                  # = "<ensemble_task_id>__<method>"
    status: SubTaskStatus = SubTaskStatus.PENDING
    fc_invocation_id: Optional[str] = None
    cache_key: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    runtime_seconds: Optional[float] = None
    error_summary: Optional[str] = None
    # Normalized output dict (per-TaskKind schema), populated on SUCCESS
    output: Optional[dict[str, Any]] = None


class EnsembleJob(BaseModel):
    """Top-level ensemble job state."""

    task_id: str
    task_kind: str                    # TaskKind.value
    customer_id: str                  # from API key
    submitted_at: datetime
    completed_at: Optional[datetime] = None
    input: dict[str, Any]             # FoldingInput / DesignInput / ... raw
    requested_methods: list[str]
    sub_tasks: dict[str, SubTaskRecord] = Field(default_factory=dict)
    aggregated_output: Optional[dict[str, Any]] = None

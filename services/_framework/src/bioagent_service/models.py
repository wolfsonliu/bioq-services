"""Framework-wide pydantic models.

These are intentionally minimal — service-specific request/response models live in
each service's own `models.py`. Anything that lives here is part of the contract
between the framework and *every* service.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp for job lifecycle events."""
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    """Lifecycle state of a job. Strings so JSON serialization is human-readable."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FailureKind(str, Enum):
    """Coarse classification of why a job failed.

    Set by `finalize_job` based on subprocess return code + output detection,
    or by `reload_from_disk` when a job was RUNNING at the moment of restart.
    Lets the client distinguish recoverable from fatal failures without parsing
    `error_summary` strings.
    """

    SUBPROCESS_ERROR = "subprocess_error"   # rc != 0
    NO_OUTPUTS = "no_outputs"               # rc == 0 but adapter.detect_outputs() = False
    TIMEOUT = "timeout"                     # not yet enforced by framework; reserved
    DATASET_INVALID = "dataset_invalid"     # raised before subprocess (e.g., bad zip)
    INTERRUPTED = "interrupted"             # was RUNNING at process/container restart


class JobInfo(BaseModel):
    """Public job state — what `GET /api/jobs/{id}` returns."""

    job_id: str = Field(..., min_length=1)
    status: JobStatus
    message: Optional[str] = None
    progress: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    input_params: Optional[dict[str, Any]] = None
    output_count: Optional[int] = None
    output_total_bytes: Optional[int] = None
    # Populated on failure: one-line exception summary extracted from the subprocess log.
    error_summary: Optional[str] = None
    # Populated on failure: trailing slice of the subprocess log (~4 KB) so clients can
    # triage without a separate /log call.
    error_tail: Optional[str] = None
    # Populated on failure: classification (subprocess_error / no_outputs / ...).
    failure_kind: Optional[FailureKind] = None
    # Instance that created (and owns the subprocess for) this job.  Used by
    # reload_from_disk to avoid corrupting running jobs that belong to a
    # different FC instance sharing the same NAS.
    instance_id: Optional[str] = None
